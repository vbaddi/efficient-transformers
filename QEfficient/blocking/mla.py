# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Parallel and sparse MLA kernels shared by benchmarks and model integrations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from QEfficient.customop.utils import ctx_gather_3d, ctx_gather_blocked_kv, ctx_scatter_3d
from QEfficient.transformers.cache_utils import QEffDynamicCompressedKVRopeCache
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    unsqueeze_dim: int,
) -> torch.Tensor:
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    return tensor * cos + _rotate_half(tensor) * sin


class DSAIndexerKeyCache:
    """Retained DSA index keys with full and block-wise read APIs."""

    def __init__(self, key_cache: torch.Tensor):
        self.key_cache = key_cache

    def update(self, key_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        self.key_cache = ctx_scatter_3d(self.key_cache, position_ids.to(torch.int32), key_states)
        return self.key_cache

    def _read(self, indices: torch.Tensor, invalid_mask: torch.Tensor) -> torch.Tensor:
        invalid_index = torch.iinfo(torch.int32).max if torch.onnx.is_in_onnx_export() else 0
        safe_indices = torch.where(invalid_mask, invalid_index, indices).to(torch.int32)
        values = ctx_gather_3d(self.key_cache, safe_indices)
        return torch.where(invalid_mask.unsqueeze(-1), torch.zeros_like(values), values)

    def read_valid(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.arange(self.key_cache.shape[1], device=self.key_cache.device).unsqueeze(0)
        invalid_mask = indices > position_ids.max(1, keepdim=True).values
        return self._read(indices, invalid_mask), invalid_mask

    def read_block(
        self, start_index: int, end_index: int, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.arange(start_index, end_index, device=self.key_cache.device).unsqueeze(0)
        invalid_mask = indices > position_ids.max(1, keepdim=True).values
        return self._read(indices, invalid_mask), invalid_mask


def gather_dsa_sparse_cache(
    compressed_kvs: QEffDynamicCompressedKVRopeCache,
    layer_idx: int,
    topk_indices: torch.Tensor,
    valid_topk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = compressed_kvs.layers[layer_idx]
    batch, num_kv_heads, _, _ = layer.ckv.shape
    gather_indices = topk_indices[:, -1].unsqueeze(1).expand(batch, num_kv_heads, -1)
    invalid_index = torch.iinfo(torch.int32).max if torch.onnx.is_in_onnx_export() else 0
    gather_indices = torch.where(valid_topk[:, -1].unsqueeze(1), gather_indices, invalid_index).to(torch.int32)
    compressed = ctx_gather_blocked_kv(layer.ckv, gather_indices)
    rope = ctx_gather_blocked_kv(layer.k_pe, gather_indices)
    invalid_mask = ~valid_topk[:, -1].unsqueeze(1).unsqueeze(-1)
    return (
        torch.where(invalid_mask, torch.zeros_like(compressed), compressed),
        torch.where(invalid_mask, torch.zeros_like(rope), rope),
    )


def build_dsa_topk_indices(
    module: nn.Module,
    hidden_states: torch.Tensor,
    q_a_proj_out: torch.Tensor,
    position_ids: torch.Tensor,
    indexer_key_cache: DSAIndexerKeyCache,
    cache_kwargs: Dict[str, Any],
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, q_len, _ = hidden_states.shape
    index_n_heads = module.dsa_index_n_heads
    index_head_dim = module.dsa_index_head_dim
    rope_dim = module.qk_rope_head_dim

    q_idx = module.dsa_wq_b(q_a_proj_out).view(bsz, q_len, index_n_heads, index_head_dim)
    q_pe, q_nope = torch.split(q_idx, [rope_dim, index_head_dim - rope_dim], dim=-1)

    k_idx = module.dsa_k_norm(module.dsa_wk(hidden_states))
    k_pe, k_nope = torch.split(k_idx, [rope_dim, index_head_dim - rope_dim], dim=-1)

    q_pe = _apply_rotary(q_pe, module.cos_cached, module.sin_cached, position_ids, unsqueeze_dim=2)
    k_pe = _apply_rotary(
        k_pe.unsqueeze(2), module.cos_cached, module.sin_cached, position_ids, unsqueeze_dim=2
    ).squeeze(2)
    q_idx = torch.cat((q_pe, q_nope), dim=-1)
    k_idx = torch.cat((k_pe, k_nope), dim=-1)

    indexer_key_cache.update(k_idx, position_ids)
    k_cache, invalid_mask = indexer_key_cache.read_valid(position_ids)

    weights = module.dsa_weights_proj(hidden_states).to(torch.float32) * (index_n_heads**-0.5)
    scores = torch.einsum("bshd,btd->bsht", q_idx.to(torch.float32), k_cache.to(torch.float32))
    scores = F.relu(scores * module.dsa_softmax_scale)
    index_scores = torch.einsum("bsht,bsh->bst", scores, weights).to(hidden_states.dtype)
    masked_score = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=index_scores.dtype, device=index_scores.device)
    index_scores = torch.where(invalid_mask.unsqueeze(1), masked_score, index_scores)

    topk_indices = torch.topk(index_scores, k=topk, dim=-1).indices.to(torch.int32)
    valid_topk = topk_indices.to(position_ids.dtype) <= position_ids.max(1, keepdim=True).values.unsqueeze(-1)
    return topk_indices, valid_topk


def build_dsa_topk_indices_blocked(
    module: nn.Module,
    hidden_states: torch.Tensor,
    q_a_proj_out: torch.Tensor,
    position_ids: torch.Tensor,
    indexer_key_cache: DSAIndexerKeyCache,
    cache_kwargs: Dict[str, Any],
    topk: int,
    num_kv_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise variant of build_dsa_topk_indices.

    Splits the indexer key cache along the T-axis into num_kv_blocks contiguous blocks.
    Each block reads only its slice (saves indexer bandwidth at long context, low pos),
    runs local top-k of size `topk`, then a final merge stage picks global top-k from the
    concatenated candidates. Future-only blocks are skipped (eager) or forced to -inf scores
    (ONNX export). Approximation: a row that ranks below local-topk in its block is lost,
    even if globally top-k.
    """
    bsz, q_len, _ = hidden_states.shape
    index_n_heads = module.dsa_index_n_heads
    index_head_dim = module.dsa_index_head_dim
    rope_dim = module.qk_rope_head_dim

    q_idx = module.dsa_wq_b(q_a_proj_out).view(bsz, q_len, index_n_heads, index_head_dim)
    q_pe, q_nope = torch.split(q_idx, [rope_dim, index_head_dim - rope_dim], dim=-1)

    k_idx = module.dsa_k_norm(module.dsa_wk(hidden_states))
    k_pe, k_nope = torch.split(k_idx, [rope_dim, index_head_dim - rope_dim], dim=-1)

    q_pe = _apply_rotary(q_pe, module.cos_cached, module.sin_cached, position_ids, unsqueeze_dim=2)
    k_pe = _apply_rotary(
        k_pe.unsqueeze(2), module.cos_cached, module.sin_cached, position_ids, unsqueeze_dim=2
    ).squeeze(2)
    q_idx = torch.cat((q_pe, q_nope), dim=-1)
    k_idx = torch.cat((k_pe, k_nope), dim=-1)

    indexer_key_cache.update(k_idx, position_ids)

    weights = module.dsa_weights_proj(hidden_states).to(torch.float32) * (index_n_heads**-0.5)
    masked_score = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=hidden_states.dtype, device=hidden_states.device)

    ctx_len = indexer_key_cache.key_cache.shape[1]
    block_size = -(-ctx_len // num_kv_blocks)
    current_position = position_ids.max(dim=-1).values

    candidate_scores: List[torch.Tensor] = []
    candidate_indices: List[torch.Tensor] = []

    for j in range(num_kv_blocks):
        start_index = j * block_size
        if j == num_kv_blocks - 1:
            kv_len_block = ctx_len - start_index
        else:
            kv_len_block = block_size
        end_index = start_index + kv_len_block

        skip_future = (torch.tensor(start_index, device=hidden_states.device) > current_position).all()
        if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
            if skip_future.item():
                break

        k_block, invalid_block = indexer_key_cache.read_block(start_index, end_index, position_ids)

        scores_block = torch.einsum("bshd,btd->bsht", q_idx.to(torch.float32), k_block.to(torch.float32))
        scores_block = F.relu(scores_block * module.dsa_softmax_scale)
        scores_block = torch.einsum("bsht,bsh->bst", scores_block, weights).to(hidden_states.dtype)

        scores_block = torch.where(invalid_block.unsqueeze(1), masked_score, scores_block)
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
            scores_block = torch.where(skip_future, masked_score, scores_block)

        local_k = min(topk, kv_len_block)
        local = torch.topk(scores_block, k=local_k, dim=-1)
        candidate_scores.append(local.values)
        candidate_indices.append(local.indices.to(torch.int32) + start_index)

    merged_scores = torch.cat(candidate_scores, dim=-1)
    merged_indices = torch.cat(candidate_indices, dim=-1)

    final = torch.topk(merged_scores, k=topk, dim=-1)
    topk_indices = torch.gather(merged_indices, -1, final.indices.to(torch.int64)).to(torch.int32)
    valid_topk = topk_indices.to(position_ids.dtype) <= current_position.unsqueeze(-1)
    return topk_indices, valid_topk


def dsa_par_mla_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    per_head_v_up: torch.Tensor,
    per_head_k_up_normal: torch.Tensor,
    absorption: bool,
    scaling: float,
    par_num_split: int,
    layer_idx: int,
    compressed_kvs: QEffDynamicCompressedKVRopeCache,
    topk_indices: torch.Tensor,
    valid_topk: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    B, NQH, QL, D_abs = query.shape
    kv_lora_rank = module.config.kv_lora_rank
    split = par_num_split
    if not absorption:
        raise ValueError("dsa_par currently supports MLA absorption mode only.")

    ckv_sparse, k_pe_sparse = gather_dsa_sparse_cache(compressed_kvs, layer_idx, topk_indices, valid_topk)
    Hkv = ckv_sparse.shape[1]
    n_rep = NQH // Hkv
    T_orig = ckv_sparse.shape[2]

    k_block = torch.cat((ckv_sparse, k_pe_sparse), dim=-1)
    ckv_for_v = ckv_sparse
    T_blk = T_orig
    pad = 0
    if T_blk % split != 0:
        pad = split - (T_blk % split)
        k_block = F.pad(k_block, (0, 0, 0, pad))
        ckv_for_v = F.pad(ckv_for_v, (0, 0, 0, pad))
        valid_topk = F.pad(valid_topk[:, -1, :], (0, pad), value=False).unsqueeze(1)
        T_blk += pad
    else:
        valid_topk = valid_topk[:, -1, :].unsqueeze(1)
    T_h = T_blk // split

    q_fold = query.reshape(B, Hkv, QL * n_rep, D_abs)
    Q_5d = q_fold.unsqueeze(2).expand(B, Hkv, split, QL * n_rep, D_abs)
    K_5d = k_block.view(B, Hkv, split, T_h, D_abs)
    V_5d = ckv_for_v.view(B, Hkv, split, T_h, kv_lora_rank)

    attn = torch.matmul(Q_5d, K_5d.transpose(-1, -2)) * scaling
    valid_mask = valid_topk.view(B, 1, split, 1, T_h)
    attn = attn.masked_fill(~valid_mask, -3.0e4)

    m = attn.max(dim=-1).values
    exp_attn = torch.exp(attn - m.unsqueeze(-1))
    exp_attn = torch.where(valid_mask, exp_attn, torch.zeros(1, dtype=exp_attn.dtype, device=exp_attn.device))
    sum_split = exp_attn.sum(dim=-1)
    out_split = torch.matmul(exp_attn, V_5d)

    m2 = m.max(dim=2).values
    w2 = torch.exp(m - m2.unsqueeze(2))
    s2 = (w2 * sum_split).sum(dim=2)
    o2 = (w2.unsqueeze(-1) * out_split).sum(dim=2)
    output = o2 / s2.unsqueeze(-1)
    output = output.view(B, Hkv, n_rep, QL, kv_lora_rank).reshape(B, NQH, QL, kv_lora_rank)

    attn_output = torch.matmul(output, per_head_v_up)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def blocked_kv_par_mla_attention_forward(
    module: nn.Module,
    query: torch.Tensor,  # [B, NQH, QL, D_abs]  absorption-space Q
    per_head_v_up: torch.Tensor,  # [1, NQH, kv_lora_rank, v_head_dim]
    per_head_k_up_normal: torch.Tensor,  # [1, NQH, qk_nope_head_dim, kv_lora_rank] — for non-absorption K
    absorption: bool,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    par_num_split: int,  # T-dim split within each KV block (maps to NSP cores)
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    compressed_kvs,
    blocking_config,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    GQA headpar-style MLA attention.

    Layout matches qwen3_gqa_kv_blocking_microbench.py:
      q_fold = query.reshape(B, Hkv, QL*n_rep, D)            # simple reshape, no permute
      Q_5d   = q_fold.unsqueeze(2).expand(..., split, ...)   # broadcast over split
      K_5d   = k_block.view(B, Hkv, split, T_h, D)          # consecutive T split

    Merge is two-stage offline (buffer all blocks):
      Stage 1: max/exp/sum across KV blocks
      Stage 2: max/exp/sum across splits
    """
    B, NQH, QL, D_abs = query.shape
    kv_lora_rank = module.config.kv_lora_rank
    split = par_num_split

    # absorption=True : all n_rep heads in a group share the same K (= ckv||k_pe)
    #                   → fold: Hkv = module.num_key_value_heads, n_rep = NQH // Hkv
    # absorption=False: each query head has its own K = ckv @ k_up_h
    #                   → cannot fold across heads; treat as Hkv=NQH, n_rep=1
    if absorption:
        Hkv = getattr(module, "num_key_value_heads", 1)
        n_rep = NQH // Hkv
    else:
        Hkv = NQH
        n_rep = 1

    # ── Q fold: reshape + unsqueeze + expand (GQA style) ─────────────────────
    q_fold = query.reshape(B, Hkv, QL * n_rep, D_abs)
    Q_5d = q_fold.unsqueeze(2).expand(B, Hkv, split, QL * n_rep, D_abs)

    ctx_len = compressed_kvs.layers[layer_idx].ckv.shape[2]
    kv_block_size = -(-ctx_len // num_kv_blocks)
    T_h_nom = -(-kv_block_size // split)  # ceiling — nominal T per split chunk

    # kv_offsets: consecutive layout, offset of position within block
    # offsets[s, t] = s*T_h_nom + t
    kv_offsets = (
        torch.arange(split, device=query.device)[:, None] * T_h_nom
        + torch.arange(T_h_nom, device=query.device)[None, :]
    ).view(1, 1, split, 1, T_h_nom)  # [1, 1, split, 1, T_h_nom]

    current_position = position_ids.max(dim=-1).values
    skip_kv = getattr(blocking_config, "skip_kv", True)

    max_buf: list = []
    sum_buf: list = []
    out_buf: list = []

    for j in range(num_kv_blocks):
        start_index = j * kv_block_size
        kv_len_block = ctx_len - start_index if j == num_kv_blocks - 1 else kv_block_size
        end_index = start_index + kv_len_block
        T_orig = kv_len_block

        skip_future = None
        if skip_kv:
            skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
            if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                if skip_future.item():
                    break

        # Read KV block: [B, Hkv, T_orig, kv_lora_rank/qk_rope_head_dim]
        ckv_block = compressed_kvs.read_only_blocked_ckv(start_index, end_index, layer_idx, cache_kwargs)
        k_pe_block = compressed_kvs.read_only_blocked_k_pe(start_index, end_index, layer_idx, cache_kwargs)

        # K in absorption or non-absorption space: [B, Hkv, T_orig, D_abs]
        if absorption:
            k_block = torch.cat((ckv_block, k_pe_block), dim=-1)  # [B, Hkv, T, 576]
            ckv_for_v = ckv_block  # [B, Hkv, T, 512]
        else:
            # Each query head needs its own K: expand ckv to NQH=Hkv, apply per-head k_up
            # ckv_block:  [B, orig_Hkv, T, kv_lora_rank]
            orig_Hkv = getattr(module, "num_key_value_heads", 1)
            n_rep_kv = NQH // orig_Hkv
            ckv_nqh = (
                ckv_block.unsqueeze(2).expand(-1, orig_Hkv, n_rep_kv, -1, -1).reshape(B, NQH, T_orig, kv_lora_rank)
            )  # [B, NQH, T, 512]
            k_pe_nqh = (
                k_pe_block.unsqueeze(2)
                .expand(-1, orig_Hkv, n_rep_kv, -1, -1)
                .reshape(B, NQH, T_orig, module.config.qk_rope_head_dim)
            )
            # per_head_k_up_normal: [1, NQH, kv_lora_rank, qk_nope_head_dim]
            k_nope = torch.matmul(ckv_nqh, per_head_k_up_normal)  # [B, NQH, T, 128]
            k_block = torch.cat((k_nope, k_pe_nqh), dim=-1)  # [B, NQH, T, 192]
            ckv_for_v = ckv_nqh  # [B, NQH, T, 512]

        # Pad T to multiple of split
        T_blk = T_orig
        pad = 0
        if T_blk % split != 0:
            pad = split - (T_blk % split)
            k_block = F.pad(k_block, (0, 0, 0, pad))
            ckv_for_v = F.pad(ckv_for_v, (0, 0, 0, pad))
            T_blk += pad
        T_h = T_blk // split

        # 5D K/V: [B, Hkv, split, T_h, D]
        K_5d = k_block.view(B, Hkv, split, T_h, D_abs)
        V_5d = ckv_for_v.view(B, Hkv, split, T_h, kv_lora_rank)

        # Attention scores: [B, Hkv, split, QL*n_rep, T_h]
        attn = torch.matmul(Q_5d, K_5d.transpose(-1, -2)) * scaling

        # Padding mask
        if pad > 0:
            chunk_start = torch.arange(split, device=attn.device) * T_h
            valid_in_chunk = T_orig - chunk_start
            k_idx = torch.arange(T_h, device=attn.device)
            pad_mask = k_idx.unsqueeze(0) >= valid_in_chunk.unsqueeze(1)  # [split, T_h]
            attn = attn.masked_fill(pad_mask.view(1, 1, split, 1, T_h), -3.0e4)

        # Causal mask: offsets within block vs query position
        off = kv_offsets if T_h == T_h_nom else kv_offsets[:, :, :, :, :T_h]
        causal_mask = off > (position_ids - start_index)[:, None, None, :, None]
        attn = attn.masked_fill(causal_mask, -3.0e4)

        m_blk = attn.max(dim=-1).values  # [B, Hkv, split, QL*n_rep]
        exp_blk = torch.exp(attn - m_blk.unsqueeze(-1))

        if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
            m_blk = torch.where(skip_future, torch.full_like(m_blk, float(MIN_MASKED_ATTENTION_VALUE)), m_blk)
            exp_blk = torch.where(skip_future, torch.zeros_like(exp_blk), exp_blk)

        sum_blk = exp_blk.sum(dim=-1)  # [B, Hkv, split, QL*n_rep]
        out_blk = torch.matmul(exp_blk, V_5d)  # [B, Hkv, split, QL*n_rep, kv_lora_rank]

        if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
            sum_blk = torch.where(skip_future, torch.zeros_like(sum_blk), sum_blk)
            out_blk = torch.where(skip_future, torch.zeros_like(out_blk), out_blk)

        max_buf.append(m_blk)
        sum_buf.append(sum_blk)
        out_buf.append(out_blk)

    # ── Stage 1: merge across KV blocks ──────────────────────────────────────
    max_stk = torch.stack(max_buf)  # [nkvb, B, Hkv, split, QL*n_rep]
    sum_stk = torch.stack(sum_buf)
    out_stk = torch.stack(out_buf)  # [nkvb, B, Hkv, split, QL*n_rep, kv_lora_rank]
    m1 = max_stk.max(dim=0).values
    w1 = torch.exp(max_stk - m1.unsqueeze(0))
    s1 = (w1 * sum_stk).sum(dim=0)  # [B, Hkv, split, QL*n_rep]
    o1 = (w1.unsqueeze(-1) * out_stk).sum(dim=0)  # [B, Hkv, split, QL*n_rep, kv_lora_rank]

    # ── Stage 2: merge across splits ─────────────────────────────────────────
    m2 = m1.max(dim=2).values  # [B, Hkv, QL*n_rep]
    w2 = torch.exp(m1 - m2.unsqueeze(2))
    s2 = (w2 * s1).sum(dim=2)
    o2 = (w2.unsqueeze(-1) * o1).sum(dim=2)  # [B, Hkv, QL*n_rep, kv_lora_rank]
    output = o2 / s2.unsqueeze(-1)

    # ── Unfold + v_up (GQA style) ─────────────────────────────────────────────
    # [B, Hkv, QL*n_rep, kv_lora_rank] → [B, NQH, QL, kv_lora_rank]
    output = output.view(B, Hkv, n_rep, QL, kv_lora_rank).reshape(B, NQH, QL, kv_lora_rank)
    attn_output = torch.matmul(output, per_head_v_up)  # [B, NQH, QL, v_head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()  # [B, QL, NQH, v_head_dim]

    return attn_output, None


def blocked_kv_par_mla_attention_prefill_forward(
    module: nn.Module,
    query: torch.Tensor,  # [B, NQH, QL, D_abs]
    per_head_v_up: torch.Tensor,  # [1, NQH, kv_lora_rank, v_head_dim]
    per_head_k_up_normal: torch.Tensor,  # [1, NQH, kv_lora_rank, qk_nope_head_dim]
    absorption: bool,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    par_num_split: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    compressed_kvs,
    blocking_config,
    position_ids: Optional[torch.Tensor] = None,
    n_rep_chunk: int = 16,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Prefill version of blocked_kv_par_mla_attention_forward.

    Flattens each repeat-head chunk with QL so accelerator matmuls stay 5D.
    The causal positions are expanded in the same repeat-major order.
    """
    B, NQH, QL, D_abs = query.shape
    kv_lora_rank = module.config.kv_lora_rank
    split = par_num_split

    if absorption:
        Hkv = getattr(module, "num_key_value_heads", 1)
        n_rep = NQH // Hkv
    else:
        Hkv = NQH
        n_rep = 1

    q_fold = query.reshape(B, Hkv, n_rep, QL, D_abs)

    ctx_len = compressed_kvs.layers[layer_idx].ckv.shape[2]
    kv_block_size = -(-ctx_len // num_kv_blocks)
    T_h_nom = -(-kv_block_size // split)

    # kv_offsets 5D: [1, 1, split, 1, T_h_nom]
    kv_offsets = (
        torch.arange(split, device=query.device)[:, None] * T_h_nom
        + torch.arange(T_h_nom, device=query.device)[None, :]
    ).view(1, 1, split, 1, T_h_nom)

    current_position = position_ids.max(dim=-1).values
    skip_kv = getattr(blocking_config, "skip_kv", True)

    max_buf: list = []
    sum_buf: list = []
    out_buf: list = []

    for j in range(num_kv_blocks):
        start_index = j * kv_block_size
        kv_len_block = ctx_len - start_index if j == num_kv_blocks - 1 else kv_block_size
        end_index = start_index + kv_len_block
        T_orig = kv_len_block

        skip_future = None
        if skip_kv:
            skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
            if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                if skip_future.item():
                    break

        ckv_block = compressed_kvs.read_only_blocked_ckv(start_index, end_index, layer_idx, cache_kwargs)
        k_pe_block = compressed_kvs.read_only_blocked_k_pe(start_index, end_index, layer_idx, cache_kwargs)

        if absorption:
            k_block = torch.cat((ckv_block, k_pe_block), dim=-1)
            ckv_for_v = ckv_block
        else:
            orig_Hkv = getattr(module, "num_key_value_heads", 1)
            n_rep_kv = NQH // orig_Hkv
            ckv_nqh = (
                ckv_block.unsqueeze(2).expand(-1, orig_Hkv, n_rep_kv, -1, -1).reshape(B, NQH, T_orig, kv_lora_rank)
            )
            k_pe_nqh = (
                k_pe_block.unsqueeze(2)
                .expand(-1, orig_Hkv, n_rep_kv, -1, -1)
                .reshape(B, NQH, T_orig, module.config.qk_rope_head_dim)
            )
            k_nope = torch.matmul(ckv_nqh, per_head_k_up_normal)
            k_block = torch.cat((k_nope, k_pe_nqh), dim=-1)
            ckv_for_v = ckv_nqh

        T_blk = T_orig
        pad = 0
        if T_blk % split != 0:
            pad = split - (T_blk % split)
            k_block = F.pad(k_block, (0, 0, 0, pad))
            ckv_for_v = F.pad(ckv_for_v, (0, 0, 0, pad))
            T_blk += pad
        T_h = T_blk // split

        # 5D K/V: [B, Hkv, split, T_h, D]
        K_5d = k_block.view(B, Hkv, split, T_h, D_abs)
        V_5d = ckv_for_v.view(B, Hkv, split, T_h, kv_lora_rank)

        off = kv_offsets if T_h == T_h_nom else kv_offsets[:, :, :, :, :T_h]

        rep_max: list = []
        rep_sum: list = []
        rep_out: list = []

        for r_start in range(0, n_rep, n_rep_chunk):
            r_end = min(r_start + n_rep_chunk, n_rep)
            chunk_size = r_end - r_start
            query_size = chunk_size * QL
            Q_chunk = q_fold[:, :, r_start:r_end, :, :].reshape(B, Hkv, query_size, D_abs)
            Q_chunk = Q_chunk.unsqueeze(2).expand(B, Hkv, split, query_size, D_abs)
            attn_c = torch.matmul(Q_chunk, K_5d.transpose(-1, -2)) * scaling

            if pad > 0:
                chunk_start = torch.arange(split, device=attn_c.device) * T_h
                valid_in_chunk = T_orig - chunk_start
                k_idx = torch.arange(T_h, device=attn_c.device)
                pad_mask = k_idx.unsqueeze(0) >= valid_in_chunk.unsqueeze(1)
                attn_c = attn_c.masked_fill(pad_mask.view(1, 1, split, 1, T_h), -3.0e4)

            chunk_positions = position_ids[:, None, :].expand(B, chunk_size, QL).reshape(B, query_size)
            causal_mask_c = off > (chunk_positions - start_index)[:, None, None, :, None]
            attn_c = attn_c.masked_fill(causal_mask_c, -3.0e4)

            m_c = attn_c.max(dim=-1).values
            exp_c = torch.exp(attn_c - m_c.unsqueeze(-1))

            if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
                m_c = torch.where(skip_future, torch.full_like(m_c, float(MIN_MASKED_ATTENTION_VALUE)), m_c)
                exp_c = torch.where(skip_future, torch.zeros_like(exp_c), exp_c)

            sum_c = exp_c.sum(dim=-1)
            out_c = torch.matmul(exp_c, V_5d)

            if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
                sum_c = torch.where(skip_future, torch.zeros_like(sum_c), sum_c)
                out_c = torch.where(skip_future, torch.zeros_like(out_c), out_c)

            rep_max.append(m_c)
            rep_sum.append(sum_c)
            rep_out.append(out_c)

        # Concatenate repeat-head chunks in repeat-major, then query-major order.
        m_blk = torch.cat(rep_max, dim=3)
        sum_blk = torch.cat(rep_sum, dim=3)
        out_blk = torch.cat(rep_out, dim=3)

        max_buf.append(m_blk)
        sum_buf.append(sum_blk)
        out_buf.append(out_blk)

    # ── Stage 1: merge across KV blocks ──────────────────────────────────────
    max_stk = torch.stack(max_buf)  # [nkvb, B, Hkv, split, n_rep*QL]
    sum_stk = torch.stack(sum_buf)
    out_stk = torch.stack(out_buf)
    m1 = max_stk.max(dim=0).values
    w1 = torch.exp(max_stk - m1.unsqueeze(0))
    s1 = (w1 * sum_stk).sum(dim=0)
    o1 = (w1.unsqueeze(-1) * out_stk).sum(dim=0)

    # ── Stage 2: merge across splits ─────────────────────────────────────────
    m2 = m1.max(dim=2).values
    w2 = torch.exp(m1 - m2.unsqueeze(2))
    s2 = (w2 * s1).sum(dim=2)
    o2 = (w2.unsqueeze(-1) * o1).sum(dim=2)
    output = o2 / s2.unsqueeze(-1)

    # ── Unfold + v_up ─────────────────────────────────────────────────────────
    # [B, Hkv, n_rep*QL, kv_lora_rank] -> [B, NQH, QL, kv_lora_rank]
    output = output.reshape(B, NQH, QL, kv_lora_rank)
    attn_output = torch.matmul(output, per_head_v_up)  # [B, NQH, QL, v_head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()  # [B, QL, NQH, v_head_dim]

    return attn_output, None


def blocked_kv_par_mla_attention_prefill_online_forward(
    module: nn.Module,
    query: torch.Tensor,  # [B, NQH, QL, D_abs]
    per_head_v_up: torch.Tensor,  # [1, NQH, kv_lora_rank, v_head_dim]
    per_head_k_up_normal: torch.Tensor,
    absorption: bool,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    par_num_split: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    compressed_kvs,
    blocking_config,
    position_ids: Optional[torch.Tensor] = None,
    n_rep_chunk: int = 16,
    ql_chunk: int = 128,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Online (flash-attention style) prefill version with n_rep_chunk and ql_chunk loop.
    """
    B, NQH, QL, D_abs = query.shape
    kv_lora_rank = module.config.kv_lora_rank
    split = par_num_split

    if absorption:
        Hkv = getattr(module, "num_key_value_heads", 1)
        n_rep = NQH // Hkv
    else:
        Hkv = NQH
        n_rep = 1

    # q_fold without expand — expand only per-chunk inside loop
    q_fold = query.reshape(B, Hkv, n_rep, QL, D_abs)

    ctx_len = compressed_kvs.layers[layer_idx].ckv.shape[2]
    kv_block_size = -(-ctx_len // num_kv_blocks)
    T_h_nom = -(-kv_block_size // split)

    # kv_offsets 5D: [1, 1, split, 1, T_h_nom]
    kv_offsets = (
        torch.arange(split, device=query.device)[:, None] * T_h_nom
        + torch.arange(T_h_nom, device=query.device)[None, :]
    ).view(1, 1, split, 1, T_h_nom)

    current_position = position_ids.max(dim=-1).values
    skip_kv = getattr(blocking_config, "skip_kv", True)

    # Running accumulators: [B, Hkv, split, n_rep, QL]
    m_acc = torch.full(
        (B, Hkv, split, n_rep, QL), float(MIN_MASKED_ATTENTION_VALUE), device=query.device, dtype=query.dtype
    )
    s_acc = torch.zeros(B, Hkv, split, n_rep, QL, device=query.device, dtype=query.dtype)
    o_acc = torch.zeros(B, Hkv, split, n_rep, QL, kv_lora_rank, device=query.device, dtype=query.dtype)

    for j in range(num_kv_blocks):
        start_index = j * kv_block_size
        kv_len_block = ctx_len - start_index if j == num_kv_blocks - 1 else kv_block_size
        end_index = start_index + kv_len_block
        T_orig = kv_len_block

        skip_future = None
        if skip_kv:
            skip_future = (torch.tensor(start_index, device=query.device) > current_position).all()
            if not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing():
                if skip_future.item():
                    break

        ckv_block = compressed_kvs.read_only_blocked_ckv(start_index, end_index, layer_idx, cache_kwargs)
        k_pe_block = compressed_kvs.read_only_blocked_k_pe(start_index, end_index, layer_idx, cache_kwargs)

        if absorption:
            k_block = torch.cat((ckv_block, k_pe_block), dim=-1)
            ckv_for_v = ckv_block
        else:
            orig_Hkv = getattr(module, "num_key_value_heads", 1)
            n_rep_kv = NQH // orig_Hkv
            ckv_nqh = (
                ckv_block.unsqueeze(2).expand(-1, orig_Hkv, n_rep_kv, -1, -1).reshape(B, NQH, T_orig, kv_lora_rank)
            )
            k_pe_nqh = (
                k_pe_block.unsqueeze(2)
                .expand(-1, orig_Hkv, n_rep_kv, -1, -1)
                .reshape(B, NQH, T_orig, module.config.qk_rope_head_dim)
            )
            k_nope = torch.matmul(ckv_nqh, per_head_k_up_normal)
            k_block = torch.cat((k_nope, k_pe_nqh), dim=-1)
            ckv_for_v = ckv_nqh

        T_blk = T_orig
        pad = 0
        if T_blk % split != 0:
            pad = split - (T_blk % split)
            k_block = F.pad(k_block, (0, 0, 0, pad))
            ckv_for_v = F.pad(ckv_for_v, (0, 0, 0, pad))
            T_blk += pad
        T_h = T_blk // split

        K_5d = k_block.view(B, Hkv, split, T_h, D_abs)
        V_5d = ckv_for_v.view(B, Hkv, split, T_h, kv_lora_rank)

        off = kv_offsets if T_h == T_h_nom else kv_offsets[:, :, :, :, :T_h]

        ql_max: list = []
        ql_sum: list = []
        ql_out: list = []

        for t_start in range(0, QL, ql_chunk):
            t_end = min(t_start + ql_chunk, QL)
            tc = t_end - t_start

            rep_max: list = []
            rep_sum: list = []
            rep_out: list = []

            for r_start in range(0, n_rep, n_rep_chunk):
                r_end = min(r_start + n_rep_chunk, n_rep)
                rc = r_end - r_start

                # expand only (r_chunk × t_chunk) portion of Q
                Q_sub = q_fold[:, :, r_start:r_end, t_start:t_end, :].unsqueeze(2).expand(B, Hkv, split, rc, tc, D_abs)
                attn_c = torch.matmul(Q_sub, K_5d.unsqueeze(3).transpose(-1, -2)) * scaling

                if pad > 0:
                    chunk_start = torch.arange(split, device=attn_c.device) * T_h
                    valid_in_chunk = T_orig - chunk_start
                    k_idx = torch.arange(T_h, device=attn_c.device)
                    pad_mask = k_idx.unsqueeze(0) >= valid_in_chunk.unsqueeze(1)
                    attn_c = attn_c.masked_fill(pad_mask.view(1, 1, split, 1, 1, T_h), -3.0e4)

                # causal mask uses position_ids for this t_chunk
                pos_sub = position_ids[:, t_start:t_end]
                causal_mask_c = off.unsqueeze(3) > (pos_sub - start_index)[:, None, None, None, :, None]
                attn_c = attn_c.masked_fill(causal_mask_c, -3.0e4)

                m_c = attn_c.max(dim=-1).values
                exp_c = torch.exp(attn_c - m_c.unsqueeze(-1))

                if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
                    m_c = torch.where(skip_future, torch.full_like(m_c, -3.0e4), m_c)
                    exp_c = torch.where(skip_future, torch.zeros_like(exp_c), exp_c)

                sum_c = exp_c.sum(dim=-1)
                out_c = torch.matmul(exp_c, V_5d.unsqueeze(3))

                if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
                    sum_c = torch.where(skip_future, torch.zeros_like(sum_c), sum_c)
                    out_c = torch.where(skip_future, torch.zeros_like(out_c), out_c)

                rep_max.append(m_c)
                rep_sum.append(sum_c)
                rep_out.append(out_c)

            # cat along n_rep dim (dim=3) for this t_chunk
            ql_max.append(torch.cat(rep_max, dim=3))
            ql_sum.append(torch.cat(rep_sum, dim=3))
            ql_out.append(torch.cat(rep_out, dim=3))

        m_blk = torch.cat(ql_max, dim=4)
        sum_blk = torch.cat(ql_sum, dim=4)
        out_blk = torch.cat(ql_out, dim=4)

        # Online merge (o_acc unnormalized)
        new_m = torch.max(m_acc, m_blk)
        previous_weight = torch.exp(m_acc - new_m)
        block_weight = torch.exp(m_blk - new_m)
        new_s = s_acc * previous_weight + sum_blk * block_weight
        new_o = previous_weight.unsqueeze(-1) * o_acc + block_weight.unsqueeze(-1) * out_blk

        if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
            m_acc = torch.where(skip_future, m_acc, new_m)
            s_acc = torch.where(skip_future, s_acc, new_s)
            o_acc = torch.where(skip_future.unsqueeze(-1), o_acc, new_o)
        else:
            m_acc, s_acc, o_acc = new_m, new_s, new_o

    # ── Merge across splits (Stage 2 only) ───────────────────────────────────
    m2 = m_acc.max(dim=2).values
    w2 = torch.exp(m_acc - m2.unsqueeze(2))
    s2 = (w2 * s_acc).sum(dim=2)
    o2 = (w2.unsqueeze(-1) * o_acc).sum(dim=2)  # o_acc unnormalized → no s_acc factor
    output = o2 / s2.unsqueeze(-1)  # single division at the end

    # [B, Hkv, n_rep, QL, kv_lora_rank] → [B, NQH, QL, kv_lora_rank]
    output = output.reshape(B, NQH, QL, kv_lora_rank)
    attn_output = torch.matmul(output, per_head_v_up)  # [B, NQH, QL, v_head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()  # [B, QL, NQH, v_head_dim]

    return attn_output, None
