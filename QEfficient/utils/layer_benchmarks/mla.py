# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import nn

from QEfficient.blocking.attention_blocking import AttentionBlockingConfig, BlockingMode
from QEfficient.blocking.mla import (
    DSAIndexerKeyCache,
    blocked_kv_par_mla_attention_forward,
    blocked_kv_par_mla_attention_prefill_forward,
    blocked_kv_par_mla_attention_prefill_online_forward,
    build_dsa_topk_indices,
    build_dsa_topk_indices_blocked,
    dsa_par_mla_attention_forward,
)
from QEfficient.transformers.cache_utils import QEffDynamicCompressedKVRopeCache
from QEfficient.transformers.models.deepseek_v3.modeling_deepseek import (
    DeepseekV3RotaryEmbedding,
    DeepseekV3YarnRotaryEmbedding,
    QEffDeepseekV3Attention,
    orig_apply_rotary_pos_emb,
    yarn_get_mscale,
)
from QEfficient.utils.layer_benchmarks.contracts import BenchmarkCase, LayerBenchmark

MLA_IMPLEMENTATIONS = ("mla", "dsa_par", "dsa_par_blocked")
MLA_BLOCKING_MODES = ("none", "kv", "h", "par", "prefill_par", "prefill_par_online")
MLA_PROFILES = {
    "kimi_k25": {
        "hidden_size": 7168,
        "num_attention_heads": 64,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
        "qk_nope_head_dim": 128,
        "rope_theta": 50_000.0,
        "max_position_embeddings": 262_144,
        "rope_scaling_factor": 64.0,
        "dsa_topk": 2048,
        "dsa_index_head_dim": 128,
        "dsa_index_n_heads": 64,
    },
    "deepseek_v32": {
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 128,
        "qk_nope_head_dim": 128,
        "rope_theta": 10_000.0,
        "max_position_embeddings": 163_840,
        "rope_scaling_factor": 40.0,
        "dsa_topk": 2048,
        "dsa_index_head_dim": 128,
        "dsa_index_n_heads": 64,
    },
    "glm5": {
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "q_lora_rank": 2048,
        "qk_rope_head_dim": 64,
        "kv_lora_rank": 512,
        "v_head_dim": 256,
        "qk_nope_head_dim": 192,
        "rope_theta": 1_000_000.0,
        "max_position_embeddings": 202_752,
        "rope_scaling_factor": None,
        "dsa_topk": 2048,
        "dsa_index_head_dim": 128,
        "dsa_index_n_heads": 32,
    },
}


@dataclass(frozen=True)
class MLAConfig:
    implementation: str
    model_profile: str
    batch_size: int
    seq_len: int
    ctx_len: int
    start_position: int
    hidden_size: int
    num_attention_heads: int
    q_lora_rank: int
    qk_rope_head_dim: int
    kv_lora_rank: int
    v_head_dim: int
    qk_nope_head_dim: int
    rope_theta: float
    max_position_embeddings: int
    mla_absorption: bool
    mla_online: bool
    attention_blocking_mode: str
    num_kv_blocks: int | None
    head_block_size: int | None
    par_num_split: int | None
    repeat_kv_heads: int
    dsa_topk: int
    dsa_index_head_dim: int
    dsa_index_n_heads: int
    kv_cache_dtype: str
    dtype: str


class _RMSNorm(nn.Module):
    def __init__(self, size: int, epsilon: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.variance_epsilon = epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        normalized = hidden_states.float() * torch.rsqrt(hidden_states.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return self.weight * normalized.to(dtype)


class MLABenchmarkLayer(QEffDeepseekV3Attention):
    """Minimal MLA layer backed directly by QEff's production attention methods."""

    def __init__(self, config: MLAConfig):
        nn.Module.__init__(self)
        self.benchmark_config = config
        self.config = SimpleNamespace(
            hidden_size=config.hidden_size,
            kv_lora_rank=config.kv_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            torch_dtype=torch.float32,
        )
        self.layer_idx = 0
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.repeat_kv_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.softmax_scale = self.q_head_dim**-0.5
        rope_scaling_factor = MLA_PROFILES[config.model_profile]["rope_scaling_factor"]
        if rope_scaling_factor is not None:
            scale = yarn_get_mscale(rope_scaling_factor, 1.0)
            self.softmax_scale *= scale * scale

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = _RMSNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(config.q_lora_rank, config.num_attention_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            config.repeat_kv_heads * (config.kv_lora_rank + config.qk_rope_head_dim),
            bias=False,
        )
        if config.repeat_kv_heads > 1:
            head_width = config.kv_lora_rank + config.qk_rope_head_dim
            with torch.no_grad():
                first_head = self.kv_a_proj_with_mqa.weight[:head_width].clone()
                self.kv_a_proj_with_mqa.weight.copy_(first_head.repeat(config.repeat_kv_heads, 1))
        self.kv_a_layernorm = _RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(config.num_attention_heads * config.v_head_dim, config.hidden_size, bias=False)
        self.__qeff_init__()
        if config.implementation in {"dsa_par", "dsa_par_blocked"}:
            self.dsa_index_head_dim = config.dsa_index_head_dim
            self.dsa_index_n_heads = config.dsa_index_n_heads
            self.dsa_softmax_scale = config.dsa_index_head_dim**-0.5
            self.dsa_wq_b = nn.Linear(
                config.q_lora_rank, config.dsa_index_n_heads * config.dsa_index_head_dim, bias=False
            )
            self.dsa_wk = nn.Linear(config.hidden_size, config.dsa_index_head_dim, bias=False)
            self.dsa_k_norm = nn.LayerNorm(config.dsa_index_head_dim, eps=1e-6)
            self.dsa_weights_proj = nn.Linear(config.hidden_size, config.dsa_index_n_heads, bias=False)

        if rope_scaling_factor is None:
            rotary = DeepseekV3RotaryEmbedding(
                config.qk_rope_head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        else:
            rotary = DeepseekV3YarnRotaryEmbedding(
                dtype=torch.float32,
                dim=config.qk_rope_head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                scaling_factor=rope_scaling_factor,
                original_max_position_embeddings=4096,
                beta_fast=32.0,
                beta_slow=1.0,
                mscale=1.0,
                mscale_all_dim=1.0,
            )
        self.register_buffer("cos_cached", rotary.cos_cached, persistent=False)
        self.register_buffer("sin_cached", rotary.sin_cached, persistent=False)

        mode = {
            "none": BlockingMode.NONE,
            "kv": BlockingMode.KV,
            "h": BlockingMode.H,
            "par": BlockingMode.KV,
            "prefill_par": BlockingMode.KV,
            "prefill_par_online": BlockingMode.KV,
        }[config.attention_blocking_mode]
        self.attn_blocking_config = AttentionBlockingConfig(
            mode=mode,
            num_kv_blocks=config.num_kv_blocks,
            head_block_size=config.head_block_size,
            skip_kv=True,
            par_num_split=config.par_num_split,
        )
        self.mla_absorption = {"absorption": config.mla_absorption, "online": config.mla_online}

    def _run(
        self,
        method,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = QEffDynamicCompressedKVRopeCache.from_legacy_cache(((compressed_kv.clone(), k_pe.clone()),))
        cos = self.cos_cached[position_ids].unsqueeze(1).to(hidden_states.dtype)
        sin = self.sin_cached[position_ids].unsqueeze(1).to(hidden_states.dtype)
        key_positions = torch.arange(self.benchmark_config.ctx_len, device=hidden_states.device).view(1, 1, 1, -1)
        attention_mask = key_positions > position_ids[:, None, :, None]
        output, _, updated_cache = method(
            self,
            hidden_states,
            (cos, sin),
            attention_mask=attention_mask,
            position_ids=position_ids,
            compressed_kvs=cache,
            mla_absorption=self.mla_absorption,
            cos_cached=cos,
            sin_cached=sin,
        )
        retained = updated_cache.to_legacy_cache()[0]
        return output, retained[0], retained[1]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        indexer_key_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self.benchmark_config.implementation in {"dsa_par", "dsa_par_blocked"}:
            if indexer_key_cache is None:
                raise ValueError("DSA MLA requires indexer_key_cache.")
            return self._run_dsa(hidden_states, position_ids, compressed_kv, k_pe, indexer_key_cache)
        if self.benchmark_config.attention_blocking_mode in {"par", "prefill_par", "prefill_par_online"}:
            return self._run_parallel(hidden_states, position_ids, compressed_kv, k_pe)
        methods = {
            "none": QEffDeepseekV3Attention.fused_forward_orig,
            "kv": QEffDeepseekV3Attention.fused_forward_kv_blocking,
            "h": QEffDeepseekV3Attention.fused_forward_h_blocking,
        }
        return self._run(
            methods[self.benchmark_config.attention_blocking_mode], hidden_states, position_ids, compressed_kv, k_pe
        )

    def reference(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        indexer_key_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self._run(
            QEffDeepseekV3Attention.fused_forward_orig,
            hidden_states,
            position_ids,
            compressed_kv,
            k_pe,
        )
        if indexer_key_cache is None:
            return outputs
        dsa_cache = DSAIndexerKeyCache(indexer_key_cache.clone())
        q_a_proj_out = self.q_a_layernorm(self.q_a_proj(hidden_states))
        build_dsa_topk_indices(
            module=self,
            hidden_states=hidden_states,
            q_a_proj_out=q_a_proj_out,
            position_ids=position_ids,
            indexer_key_cache=dsa_cache,
            cache_kwargs={"position_ids": position_ids, "batch_index": None},
            topk=self.benchmark_config.dsa_topk,
        )
        return (*outputs, dsa_cache.key_cache)

    def _project_blocked(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, QEffDynamicCompressedKVRopeCache, dict[str, torch.Tensor | None]]:
        batch_size, query_length, _ = hidden_states.shape
        projected_kv = self.kv_a_proj_with_mqa(hidden_states).view(
            batch_size, query_length, -1, self.kv_lora_rank + self.qk_rope_head_dim
        )
        projected_kv = projected_kv.transpose(1, 2)
        compressed = self.kv_a_layernorm(projected_kv[..., : self.kv_lora_rank])
        rope = projected_kv[..., self.kv_lora_rank :]
        q_a_proj_out = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_pe = torch.matmul(q_a_proj_out, self.q_rope).view(
            batch_size, query_length, self.num_heads, self.qk_rope_head_dim
        )
        q_pe = q_pe.transpose(1, 2)
        cos = self.cos_cached[position_ids].unsqueeze(1).to(hidden_states.dtype)
        sin = self.sin_cached[position_ids].unsqueeze(1).to(hidden_states.dtype)
        q_pe, rope = orig_apply_rotary_pos_emb(q_pe, rope, cos, sin)
        cache = QEffDynamicCompressedKVRopeCache.from_legacy_cache(((compressed_kv.clone(), k_pe_cache.clone()),))
        cache_kwargs = {"position_ids": position_ids, "batch_index": None}
        cache.write_only_ckv(compressed, self.layer_idx, cache_kwargs)
        cache.write_only_k_pe(rope, self.layer_idx, cache_kwargs)
        if self.mla_absorption["absorption"]:
            if self.mla_absorption["online"]:
                fused = torch.matmul(self.per_head_q_up, self.per_head_k_up)
                q_nope = torch.matmul(q_a_proj_out, fused)
            else:
                q_nope = torch.matmul(q_a_proj_out, self.fusedqk)
        else:
            q_nope = torch.bmm(q_a_proj_out, self.q_up).view(
                batch_size, query_length, self.num_heads, self.qk_nope_head_dim
            )
            q_nope = q_nope.transpose(1, 2)
        return torch.cat((q_nope, q_pe), dim=-1), q_a_proj_out, cache, cache_kwargs

    def _run_parallel(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, _, cache, cache_kwargs = self._project_blocked(hidden_states, position_ids, compressed_kv, k_pe)
        kernels = {
            "par": blocked_kv_par_mla_attention_forward,
            "prefill_par": blocked_kv_par_mla_attention_prefill_forward,
            "prefill_par_online": blocked_kv_par_mla_attention_prefill_online_forward,
        }
        output, _ = kernels[self.benchmark_config.attention_blocking_mode](
            module=self,
            query=query,
            per_head_v_up=self.per_head_v_up,
            per_head_k_up_normal=self.per_head_k_up_normal,
            absorption=self.mla_absorption["absorption"],
            attention_mask=None,
            scaling=self.softmax_scale,
            num_kv_blocks=self.benchmark_config.num_kv_blocks,
            par_num_split=self.benchmark_config.par_num_split,
            cache_kwargs=cache_kwargs,
            layer_idx=self.layer_idx,
            compressed_kvs=cache,
            blocking_config=self.attn_blocking_config,
            position_ids=position_ids,
        )
        retained = cache.to_legacy_cache()[0]
        return self.o_proj(output.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)), *retained

    def _run_dsa(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        indexer_key_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query, q_a_proj_out, cache, cache_kwargs = self._project_blocked(
            hidden_states, position_ids, compressed_kv, k_pe
        )
        dsa_cache = DSAIndexerKeyCache(indexer_key_cache.clone())
        builder = (
            build_dsa_topk_indices_blocked
            if self.benchmark_config.implementation == "dsa_par_blocked"
            else build_dsa_topk_indices
        )
        builder_kwargs = {
            "module": self,
            "hidden_states": hidden_states,
            "q_a_proj_out": q_a_proj_out,
            "position_ids": position_ids,
            "indexer_key_cache": dsa_cache,
            "cache_kwargs": cache_kwargs,
            "topk": self.benchmark_config.dsa_topk,
        }
        if self.benchmark_config.implementation == "dsa_par_blocked":
            builder_kwargs["num_kv_blocks"] = self.benchmark_config.num_kv_blocks
        topk_indices, valid_topk = builder(**builder_kwargs)
        output, _ = dsa_par_mla_attention_forward(
            module=self,
            query=query,
            per_head_v_up=self.per_head_v_up,
            per_head_k_up_normal=self.per_head_k_up_normal,
            absorption=self.mla_absorption["absorption"],
            scaling=self.softmax_scale,
            par_num_split=self.benchmark_config.par_num_split,
            layer_idx=self.layer_idx,
            compressed_kvs=cache,
            topk_indices=topk_indices,
            valid_topk=valid_topk,
        )
        retained = cache.to_legacy_cache()[0]
        output = self.o_proj(output.reshape(hidden_states.shape[0], hidden_states.shape[1], -1))
        return output, retained[0], retained[1], dsa_cache.key_cache


class MLABenchmark(LayerBenchmark):
    name = "mla"
    description = "DeepSeek-style multi-head latent attention with compressed retained state."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--implementation", "--impls", dest="implementation", choices=MLA_IMPLEMENTATIONS, default="mla"
        )
        parser.add_argument("--model-profile", choices=tuple(MLA_PROFILES), default="kimi_k25")
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument("--seq-len", type=int, default=1)
        parser.add_argument("--ctx-len", type=int, default=32)
        parser.add_argument("--start-position", "--start-pos-id", dest="start_position", type=int, default=0)
        parser.add_argument("--hidden-size", type=int)
        parser.add_argument("--num-attention-heads", type=int)
        parser.add_argument("--q-lora-rank", type=int)
        parser.add_argument("--qk-rope-head-dim", type=int)
        parser.add_argument("--kv-lora-rank", type=int)
        parser.add_argument("--v-head-dim", type=int)
        parser.add_argument("--qk-nope-head-dim", type=int)
        parser.add_argument("--rope-theta", type=float)
        parser.add_argument("--max-position-embeddings", type=int)
        parser.add_argument("--mla-absorption", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--mla-online", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--attn-blocking-mode", choices=MLA_BLOCKING_MODES, default="none")
        parser.add_argument("--attn-num-kv-blocks", dest="num_kv_blocks", type=int)
        parser.add_argument("--attn-head-block-size", dest="head_block_size", type=int)
        parser.add_argument("--par-num-split", type=int)
        parser.add_argument("--repeat-kv-heads", type=int, default=1)
        parser.add_argument("--dsa-topk", type=int)
        parser.add_argument("--dsa-index-head-dim", type=int)
        parser.add_argument("--dsa-index-n-heads", type=int)
        parser.add_argument("--kv-cache-dtype", choices=("float16", "mxint8"), default="float16")
        parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")

    def resolved_config(self, args: argparse.Namespace) -> Mapping[str, Any]:
        profile = MLA_PROFILES[args.model_profile]
        for name, value in profile.items():
            if name != "rope_scaling_factor" and getattr(args, name) is None:
                setattr(args, name, value)
        if args.par_num_split is None:
            args.par_num_split = args.repeat_kv_heads
        dimensions = (
            args.batch_size,
            args.seq_len,
            args.ctx_len,
            args.hidden_size,
            args.num_attention_heads,
            args.q_lora_rank,
            args.qk_rope_head_dim,
            args.kv_lora_rank,
            args.v_head_dim,
            args.qk_nope_head_dim,
            args.repeat_kv_heads,
            args.par_num_split,
            args.dsa_topk,
            args.dsa_index_head_dim,
            args.dsa_index_n_heads,
        )
        if min(dimensions) <= 0:
            raise ValueError("MLA dimensions must be positive.")
        if args.ctx_len <= args.seq_len:
            raise ValueError("--ctx-len must be greater than --seq-len.")
        if args.start_position < 0 or args.start_position + args.seq_len > args.ctx_len:
            raise ValueError("The requested MLA positions must fit within --ctx-len.")
        if args.ctx_len > args.max_position_embeddings:
            raise ValueError("--ctx-len cannot exceed --max-position-embeddings.")
        if args.num_attention_heads % args.repeat_kv_heads:
            raise ValueError("--repeat-kv-heads must divide --num-attention-heads.")
        if args.qk_rope_head_dim % 2:
            raise ValueError("--qk-rope-head-dim must be even.")
        if args.attn_blocking_mode in {"kv", "par", "prefill_par", "prefill_par_online"} and (
            args.num_kv_blocks is None or args.num_kv_blocks <= 0
        ):
            raise ValueError(f"{args.attn_blocking_mode} blocking requires --attn-num-kv-blocks to be positive.")
        if args.attn_blocking_mode == "h" and (args.head_block_size is None or args.head_block_size <= 0):
            raise ValueError("Head blocking requires --attn-head-block-size to be positive.")
        if args.head_block_size is not None and args.num_attention_heads % args.head_block_size:
            raise ValueError("--attn-head-block-size must divide --num-attention-heads.")
        if args.implementation in {"dsa_par", "dsa_par_blocked"}:
            if args.attn_blocking_mode != "par":
                raise ValueError(f"--impls {args.implementation} requires --attn-blocking-mode par.")
            if args.seq_len != 1:
                raise ValueError(f"--impls {args.implementation} is decode-only; use --seq-len 1.")
            if not args.mla_absorption:
                raise ValueError(f"--impls {args.implementation} requires --mla-absorption.")
            if args.dsa_topk > args.ctx_len:
                raise ValueError("--dsa-topk cannot exceed --ctx-len.")
        if args.implementation == "dsa_par_blocked" and args.num_kv_blocks <= 1:
            raise ValueError("--impls dsa_par_blocked requires --attn-num-kv-blocks greater than one.")
        return asdict(
            MLAConfig(
                implementation=args.implementation,
                model_profile=args.model_profile,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                ctx_len=args.ctx_len,
                start_position=args.start_position,
                hidden_size=args.hidden_size,
                num_attention_heads=args.num_attention_heads,
                q_lora_rank=args.q_lora_rank,
                qk_rope_head_dim=args.qk_rope_head_dim,
                kv_lora_rank=args.kv_lora_rank,
                v_head_dim=args.v_head_dim,
                qk_nope_head_dim=args.qk_nope_head_dim,
                rope_theta=args.rope_theta,
                max_position_embeddings=args.max_position_embeddings,
                mla_absorption=args.mla_absorption,
                mla_online=args.mla_online,
                attention_blocking_mode=args.attn_blocking_mode,
                num_kv_blocks=args.num_kv_blocks,
                head_block_size=args.head_block_size,
                par_num_split=args.par_num_split,
                repeat_kv_heads=args.repeat_kv_heads,
                dsa_topk=args.dsa_topk,
                dsa_index_head_dim=args.dsa_index_head_dim,
                dsa_index_n_heads=args.dsa_index_n_heads,
                kv_cache_dtype=args.kv_cache_dtype,
                dtype=args.dtype,
            )
        )

    def build_case(self, config: Mapping[str, Any], seed: int) -> BenchmarkCase:
        resolved = MLAConfig(**config)
        generator = torch.Generator().manual_seed(seed)
        module = MLABenchmarkLayer(resolved).eval()
        hidden_states = torch.randn(resolved.batch_size, resolved.seq_len, resolved.hidden_size, generator=generator)
        position_ids = torch.arange(
            resolved.start_position, resolved.start_position + resolved.seq_len, dtype=torch.int64
        ).expand(resolved.batch_size, -1)
        compressed_kv = torch.zeros(
            resolved.batch_size,
            resolved.repeat_kv_heads,
            resolved.ctx_len,
            resolved.kv_lora_rank,
        )
        k_pe = torch.zeros(
            resolved.batch_size,
            resolved.repeat_kv_heads,
            resolved.ctx_len,
            resolved.qk_rope_head_dim,
        )
        if resolved.start_position:
            compressed_prefix = torch.randn(
                resolved.batch_size,
                1,
                resolved.start_position,
                resolved.kv_lora_rank,
                generator=generator,
            )
            rope_prefix = torch.randn(
                resolved.batch_size,
                1,
                resolved.start_position,
                resolved.qk_rope_head_dim,
                generator=generator,
            )
            compressed_kv[:, :, : resolved.start_position] = compressed_prefix.expand(
                -1, resolved.repeat_kv_heads, -1, -1
            )
            k_pe[:, :, : resolved.start_position] = rope_prefix.expand(-1, resolved.repeat_kv_heads, -1, -1)
        inputs = OrderedDict(
            hidden_states=hidden_states,
            position_ids=position_ids,
            **{"compressed_kv.0": compressed_kv, "k_pe.0": k_pe},
        )
        retained_names = {
            "compressed_kv.0",
            "k_pe.0",
            "compressed_kv.0_RetainedState",
            "k_pe.0_RetainedState",
        }
        output_names = ["attn_output", "compressed_kv.0_RetainedState", "k_pe.0_RetainedState"]
        dynamic_axes = {
            "hidden_states": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "attn_output": {0: "batch_size", 1: "seq_len"},
        }
        if resolved.implementation in {"dsa_par", "dsa_par_blocked"}:
            indexer_key_cache = torch.zeros(resolved.batch_size, resolved.ctx_len, resolved.dsa_index_head_dim)
            if resolved.start_position:
                indexer_key_cache[:, : resolved.start_position] = torch.randn(
                    resolved.batch_size,
                    resolved.start_position,
                    resolved.dsa_index_head_dim,
                    generator=generator,
                )
            inputs["indexer_key_cache"] = indexer_key_cache
            retained_names.update({"indexer_key_cache", "indexer_key_cache_RetainedState"})
            output_names.append("indexer_key_cache_RetainedState")
            dynamic_axes["indexer_key_cache"] = {0: "batch_size"}
            dynamic_axes["indexer_key_cache_RetainedState"] = {0: "batch_size"}
        return BenchmarkCase(
            module=module,
            reference=module.reference,
            inputs=inputs,
            output_names=tuple(output_names),
            dynamic_axes=dynamic_axes,
            specializations={
                "batch_size": resolved.batch_size,
                "seq_len": resolved.seq_len,
                "ctx_len": resolved.ctx_len,
            },
            custom_io={name: resolved.kv_cache_dtype for name in retained_names},
            retained_names=retained_names,
        )
