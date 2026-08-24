# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Production-aligned GQA kernels shared by QEff models and layer benchmarks."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache

from QEfficient.blocking.gqa_packed import gqa_packed_attention
from QEfficient.customop.utils import ctx_gather_blocked_kv, ctx_gather_blocked_kv_cb
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE

DECODE_GQA_IMPLEMENTATIONS = (
    "decode_attn_headpar",
    "decode_attn_headpar_chunk_kv",
    "decode_attn_headpar_chunk_kv_unroll",
    "decode_attn_headpar_batch_split",
    "decode_attn_headpar_batch_split_unroll",
)
PREFILL_GQA_IMPLEMENTATIONS = (
    "prefill_attn_parallel",
    "prefill_attn_parallel_chunk_kv",
    "prefill_attn_online_prefill",
)
GQA_IMPLEMENTATIONS = DECODE_GQA_IMPLEMENTATIONS + PREFILL_GQA_IMPLEMENTATIONS
CHUNK_KV_GQA_IMPLEMENTATIONS = (
    "decode_attn_headpar_chunk_kv",
    "decode_attn_headpar_chunk_kv_unroll",
    "prefill_attn_parallel_chunk_kv",
)
UNROLLED_GQA_IMPLEMENTATIONS = (
    "decode_attn_headpar_chunk_kv_unroll",
    "decode_attn_headpar_batch_split_unroll",
)


def _raw_cache_tensors(past_key_value: Cache, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the retained tensors without introducing another gather."""
    try:
        key_cache, value_cache = past_key_value[layer_idx]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"GQA implementation requires an initialized cache at layer {layer_idx}.") from exc
    if key_cache is None or value_cache is None:
        raise ValueError(f"GQA implementation requires an initialized cache at layer {layer_idx}.")
    return key_cache, value_cache


def _read_folded_block(
    cache: torch.Tensor,
    position_ids: torch.Tensor,
    start_index: int,
    end_index: int,
    *,
    zero_invalid: bool,
    batch_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fold batch and KV heads before the blocked gather used by head-parallel GQA."""
    batch_size, num_kv_heads, ctx_len, head_dim = cache.shape
    block_size = end_index - start_index
    gather_limit = position_ids.max(dim=1, keepdim=True).values.unsqueeze(1)
    indices = torch.arange(start_index, end_index, device=cache.device).view(1, 1, block_size)
    invalid = indices > gather_limit
    invalid_index = torch.iinfo(torch.int32).max if torch.onnx.is_in_onnx_export() else 0
    indices = torch.where(invalid, invalid_index, indices).to(torch.int32)
    if batch_index is None:
        indices = indices.expand(batch_size, num_kv_heads, block_size).reshape(1, batch_size * num_kv_heads, block_size)
        block = ctx_gather_blocked_kv(cache.reshape(1, batch_size * num_kv_heads, ctx_len, head_dim), indices).reshape(
            1, batch_size * num_kv_heads, block_size, head_dim
        )
        invalid = invalid.expand(batch_size, num_kv_heads, block_size).reshape(1, batch_size * num_kv_heads, block_size)
    else:
        block = ctx_gather_blocked_kv_cb(cache, batch_index, indices).reshape(
            1, batch_size * num_kv_heads, block_size, head_dim
        )
        invalid = invalid.expand(batch_size, num_kv_heads, block_size).reshape(1, batch_size * num_kv_heads, block_size)
    if zero_invalid:
        block = torch.where(invalid.unsqueeze(-1), torch.zeros_like(block), block)
    return block


def _merge_block_statistics(
    maxima: list[torch.Tensor],
    denominators: list[torch.Tensor],
    numerators: list[torch.Tensor],
) -> torch.Tensor:
    maxima_stacked = torch.stack(maxima)
    denominators_stacked = torch.stack(denominators)
    numerators_stacked = torch.stack(numerators)
    global_max = maxima_stacked.max(dim=0).values
    weights = torch.exp(maxima_stacked - global_max.unsqueeze(0))
    denominator = (weights * denominators_stacked).sum(dim=0)
    numerator = (weights.unsqueeze(-1) * numerators_stacked).sum(dim=0)
    return numerator / denominator.unsqueeze(-1)


def _online_block_statistics(
    maxima: list[torch.Tensor],
    denominators: list[torch.Tensor],
    numerators: list[torch.Tensor],
) -> torch.Tensor:
    """Merge sequentially to retain the standalone online-prefill graph."""
    maximum = maxima[0]
    denominator = denominators[0]
    numerator = numerators[0]
    for block_maximum, block_denominator, block_numerator in zip(maxima[1:], denominators[1:], numerators[1:]):
        updated_maximum = torch.maximum(maximum, block_maximum)
        previous_weight = torch.exp(maximum - updated_maximum)
        current_weight = torch.exp(block_maximum - updated_maximum)
        denominator = previous_weight * denominator + current_weight * block_denominator
        numerator = previous_weight.unsqueeze(-1) * numerator + current_weight.unsqueeze(-1) * block_numerator
        maximum = updated_maximum
    return numerator / denominator.unsqueeze(-1)


def gqa_head_parallel_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    scaling: float,
    num_kv_blocks: int,
    skip_kv: bool = True,
    query_block_size: Optional[int] = None,
    batch_index: Optional[torch.Tensor] = None,
    merge_online: bool = False,
    kv_block_unroll: int = 1,
) -> torch.Tensor:
    """GQA online-softmax kernel with batch/KV-head folding and parallel block reduction.

    The return layout matches the normal attention-internal layout: ``[B, Hq, Q, D]``.
    ``query_block_size`` bounds prefill activation memory; decode naturally uses one query token.
    """
    batch_size, num_query_heads, query_len, head_dim = query.shape
    cache_batch, num_kv_heads, ctx_len, cache_head_dim = key_cache.shape
    if cache_batch != batch_size or value_cache.shape != key_cache.shape:
        raise ValueError("GQA key/value caches must have matching [batch, KV heads, context, head dim] shapes.")
    if cache_head_dim != head_dim or num_query_heads % num_kv_heads:
        raise ValueError("GQA query heads must be divisible by KV heads and use the cache head dimension.")
    if position_ids.shape != (batch_size, query_len):
        raise ValueError(f"position_ids must have shape {(batch_size, query_len)}, got {tuple(position_ids.shape)}.")
    if batch_index is not None:
        batch_index = batch_index.to(torch.int32)

    num_kv_blocks = max(1, int(num_kv_blocks))
    kv_block_size = -(-ctx_len // num_kv_blocks)
    num_kv_groups = num_query_heads // num_kv_heads
    query_block_size = query_len if not query_block_size else max(1, int(query_block_size))
    output_chunks = []

    for query_start in range(0, query_len, query_block_size):
        query_end = min(query_start + query_block_size, query_len)
        query_chunk_len = query_end - query_start
        query_chunk = query[:, :, query_start:query_end, :]
        query_folded = query_chunk.reshape(batch_size, num_kv_heads, num_kv_groups, query_chunk_len, head_dim).reshape(
            1, batch_size * num_kv_heads, num_kv_groups * query_chunk_len, head_dim
        )
        query_positions = (
            position_ids[:, query_start:query_end]
            .reshape(batch_size, 1, 1, query_chunk_len)
            .expand(batch_size, num_kv_heads, num_kv_groups, query_chunk_len)
            .reshape(1, batch_size * num_kv_heads, num_kv_groups * query_chunk_len, 1)
        )

        maxima: list[torch.Tensor] = []
        denominators: list[torch.Tensor] = []
        numerators: list[torch.Tensor] = []
        current_position = position_ids[:, query_start:query_end].max()
        stop = False
        for group_start in range(0, num_kv_blocks, max(1, kv_block_unroll)):
            active = []
            for block_idx in range(group_start, min(group_start + max(1, kv_block_unroll), num_kv_blocks)):
                start_index = block_idx * kv_block_size
                end_index = min(start_index + kv_block_size, ctx_len)
                if start_index >= end_index:
                    continue
                skip_future = torch.tensor(start_index, device=query.device) > current_position
                if skip_kv and not (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()) and skip_future.item():
                    stop = True
                    break
                active.append((start_index, end_index, skip_future))
            key_blocks = [
                _read_folded_block(
                    key_cache,
                    position_ids,
                    start_index,
                    end_index,
                    zero_invalid=False,
                    batch_index=batch_index,
                )
                for start_index, end_index, _ in active
            ]
            group_exponentials = []
            for key_block, (start_index, end_index, skip_future) in zip(key_blocks, active):
                scores = torch.matmul(query_folded, key_block.transpose(-1, -2)) * scaling
                key_positions = torch.arange(start_index, end_index, device=query.device).view(1, 1, 1, -1)
                scores = torch.where(
                    key_positions > query_positions,
                    torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=scores.dtype, device=scores.device),
                    scores,
                )
                block_max = scores.max(dim=-1).values
                row_skipped = torch.tensor(start_index, device=query.device) > query_positions.squeeze(-1)
                safe_max = torch.where(row_skipped, torch.zeros_like(block_max), block_max)
                exponentials = torch.exp(scores - safe_max.unsqueeze(-1))
                exponentials = torch.where(row_skipped.unsqueeze(-1), torch.zeros_like(exponentials), exponentials)
                block_max = torch.where(row_skipped, torch.full_like(block_max, float("-inf")), block_max)
                block_denominator = exponentials.sum(dim=-1)
                if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
                    block_max = torch.where(skip_future, torch.full_like(block_max, float("-inf")), block_max)
                    block_denominator = torch.where(skip_future, torch.zeros_like(block_denominator), block_denominator)
                maxima.append(block_max)
                denominators.append(block_denominator)
                group_exponentials.append(exponentials)
            value_blocks = [
                _read_folded_block(
                    value_cache,
                    position_ids,
                    start_index,
                    end_index,
                    zero_invalid=True,
                    batch_index=batch_index,
                )
                for start_index, end_index, _ in active
            ]
            for exponentials, value_block, (_, _, skip_future) in zip(group_exponentials, value_blocks, active):
                block_numerator = torch.matmul(exponentials, value_block)
                if skip_kv and (torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()):
                    block_numerator = torch.where(skip_future, torch.zeros_like(block_numerator), block_numerator)
                numerators.append(block_numerator)
            if stop:
                break

        if not maxima:
            raise ValueError("No GQA KV blocks were available for the requested positions.")
        merge = _online_block_statistics if merge_online else _merge_block_statistics
        output = merge(maxima, denominators, numerators)
        output_chunks.append(
            output.reshape(batch_size, num_kv_heads, num_kv_groups, query_chunk_len, head_dim).reshape(
                batch_size, num_query_heads, query_chunk_len, head_dim
            )
        )
    return torch.cat(output_chunks, dim=2)


def _gqa_split_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    scaling: float,
    num_kv_blocks: int,
    split: int,
    skip_kv: bool,
    query_block_size: Optional[int],
    kv_block_unroll: int,
) -> torch.Tensor:
    """Core-split schedule used by the head-parallel and parallel-prefill variants."""
    batch_size, num_query_heads, query_len, head_dim = query.shape
    num_kv_heads = key_cache.shape[1]
    context_length = key_cache.shape[2]
    groups = num_query_heads // num_kv_heads
    block_size = -(-context_length // num_kv_blocks)
    if block_size % split:
        raise ValueError("Each GQA KV block must be divisible by the selected core split.")
    query_block_size = query_len if not query_block_size else query_block_size
    batch_outputs = []
    is_export = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
    for batch in range(batch_size):
        query_outputs = []
        for query_start in range(0, query_len, query_block_size):
            query_stop = min(query_start + query_block_size, query_len)
            chunk_length = query_stop - query_start
            query_chunk = query[batch : batch + 1, :, query_start:query_stop]
            query_folded = query_chunk.reshape(1, num_kv_heads, groups * chunk_length, head_dim)
            query_split = query_folded.unsqueeze(2).expand(1, num_kv_heads, split, groups * chunk_length, head_dim)
            query_positions = position_ids[batch : batch + 1, query_start:query_stop]
            maxima = []
            denominators = []
            numerators = []
            stop = False
            for group_start in range(0, num_kv_blocks, max(1, kv_block_unroll)):
                active = []
                for block in range(group_start, min(group_start + max(1, kv_block_unroll), num_kv_blocks)):
                    start = block * block_size
                    end = min(start + block_size, context_length)
                    skip_future = torch.tensor(start, device=query.device) > query_positions.max()
                    if skip_kv and not is_export and skip_future.item():
                        stop = True
                        break
                    active.append((start, end, skip_future))
                key_blocks = [
                    _read_folded_block(
                        key_cache[batch : batch + 1],
                        position_ids[batch : batch + 1],
                        start,
                        end,
                        zero_invalid=False,
                    )
                    for start, end, _ in active
                ]
                exponentials = []
                for key_block, (start, end, skip_future) in zip(key_blocks, active):
                    split_length = (end - start) // split
                    key_split = key_block.view(1, num_kv_heads, split, split_length, head_dim)
                    scores = torch.matmul(query_split, key_split.transpose(-1, -2)) * scaling
                    offsets = torch.arange(end - start, device=query.device).view(split, split_length)
                    mask = offsets.view(1, 1, split, 1, split_length) > (query_positions - start).view(
                        1, 1, 1, chunk_length, 1
                    )
                    mask = mask.repeat(1, num_kv_heads, 1, groups, 1)
                    scores = torch.where(
                        mask,
                        torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=scores.dtype, device=scores.device),
                        scores,
                    )
                    block_max = scores.max(dim=-1).values
                    split_starts = start + torch.arange(split, device=query.device) * split_length
                    row_skipped = split_starts.view(1, 1, split, 1) > query_positions.view(1, 1, 1, chunk_length)
                    row_skipped = row_skipped.repeat(1, num_kv_heads, 1, groups)
                    safe_max = torch.where(row_skipped, torch.zeros_like(block_max), block_max)
                    block_exp = torch.exp(scores - safe_max.unsqueeze(-1))
                    block_exp = torch.where(row_skipped.unsqueeze(-1), torch.zeros_like(block_exp), block_exp)
                    block_max = torch.where(row_skipped, torch.full_like(block_max, float("-inf")), block_max)
                    block_sum = block_exp.sum(dim=-1)
                    if skip_kv and is_export:
                        block_max = torch.where(skip_future, torch.full_like(block_max, float("-inf")), block_max)
                        block_exp = torch.where(skip_future, torch.zeros_like(block_exp), block_exp)
                        block_sum = torch.where(skip_future, torch.zeros_like(block_sum), block_sum)
                    maxima.append(block_max)
                    denominators.append(block_sum)
                    exponentials.append(block_exp)
                value_blocks = [
                    _read_folded_block(
                        value_cache[batch : batch + 1],
                        position_ids[batch : batch + 1],
                        start,
                        end,
                        zero_invalid=True,
                    ).view(1, num_kv_heads, split, (end - start) // split, head_dim)
                    for start, end, _ in active
                ]
                for block_exp, value_block, (_, _, skip_future) in zip(exponentials, value_blocks, active):
                    block_output = torch.matmul(block_exp, value_block)
                    if skip_kv and is_export:
                        block_output = torch.where(skip_future, torch.zeros_like(block_output), block_output)
                    numerators.append(block_output)
                if stop:
                    break
            maximum = torch.stack(maxima).max(dim=0).values
            weights = torch.exp(torch.stack(maxima) - maximum.unsqueeze(0))
            denominator = (weights * torch.stack(denominators)).sum(dim=0)
            numerator = (weights.unsqueeze(-1) * torch.stack(numerators)).sum(dim=0)
            split_maximum = maximum.max(dim=2).values
            split_weights = torch.exp(maximum - split_maximum.unsqueeze(2))
            denominator = (split_weights * denominator).sum(dim=2)
            numerator = (split_weights.unsqueeze(-1) * numerator).sum(dim=2)
            query_outputs.append(
                (numerator / denominator.unsqueeze(-1)).view(1, num_query_heads, chunk_length, head_dim)
            )
        batch_outputs.append(torch.cat(query_outputs, dim=2))
    return torch.cat(batch_outputs, dim=0)


def gqa_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    implementation: str,
    scaling: float,
    num_kv_blocks: int,
    skip_kv: bool = True,
    query_block_size: Optional[int] = None,
    batch_index: Optional[torch.Tensor] = None,
    num_cores_per_device: Optional[int] = None,
    chunk_kv_n: int = 1,
    chunk_kv_size: int = 8192,
    query_head_block_size: int = 1,
    kv_block_unroll: int = 1,
) -> torch.Tensor:
    """Dispatch the standalone GQA graph variants through one production API.

    Head-parallel and packed-KV variants subdivide each logical KV block into
    independent core/device lanes before the final online-softmax reduction.
    Unrolled variants issue a group of gathers before consuming those blocks.
    """
    if implementation not in GQA_IMPLEMENTATIONS:
        raise ValueError(f"Unknown GQA implementation {implementation!r}.")
    if implementation in CHUNK_KV_GQA_IMPLEMENTATIONS:
        split = max(1, int(num_cores_per_device or query.shape[1] * chunk_kv_n / key_cache.shape[1]))
        return gqa_packed_attention(
            query,
            key_cache,
            value_cache,
            position_ids,
            phase="prefill" if implementation.startswith("prefill") else "decode",
            scaling=scaling,
            num_kv_blocks=num_kv_blocks,
            split=split,
            chunk_kv_n=chunk_kv_n,
            chunk_kv_size=chunk_kv_size,
            skip_kv=skip_kv,
            query_block_size=query_block_size,
            query_head_block_size=query_head_block_size,
            kv_block_unroll=kv_block_unroll if implementation in UNROLLED_GQA_IMPLEMENTATIONS else 1,
        )
    if implementation not in {
        "decode_attn_headpar_batch_split",
        "decode_attn_headpar_batch_split_unroll",
        "prefill_attn_online_prefill",
    }:
        if implementation.startswith("prefill"):
            split = max(1, (num_cores_per_device or query.shape[1] // key_cache.shape[1]) // key_cache.shape[1])
        else:
            split = max(1, num_cores_per_device or query.shape[1] // key_cache.shape[1])
        return _gqa_split_attention(
            query,
            key_cache,
            value_cache,
            position_ids,
            scaling=scaling,
            num_kv_blocks=num_kv_blocks,
            split=split,
            skip_kv=skip_kv,
            query_block_size=query_block_size,
            kv_block_unroll=kv_block_unroll if implementation in UNROLLED_GQA_IMPLEMENTATIONS else 1,
        )
    return gqa_head_parallel_attention(
        query,
        key_cache,
        value_cache,
        position_ids,
        scaling=scaling,
        num_kv_blocks=num_kv_blocks,
        skip_kv=skip_kv,
        query_block_size=query_block_size,
        batch_index=batch_index,
        merge_online=implementation == "prefill_attn_online_prefill",
        kv_block_unroll=kv_block_unroll if implementation in UNROLLED_GQA_IMPLEMENTATIONS else 1,
    )


def gqa_v1_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: int,
    cache_kwargs: Dict[str, Any],
    layer_idx: int,
    past_key_value: Cache,
    *,
    skip_kv: bool = True,
    query_block_size: Optional[int] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """Adapter from QEff's blocked-attention interface to the shared GQA kernel."""
    del module, key, value, attention_mask, kwargs
    key_cache, value_cache = _raw_cache_tensors(past_key_value, layer_idx)
    output = gqa_head_parallel_attention(
        query,
        key_cache,
        value_cache,
        cache_kwargs["position_ids"],
        scaling=scaling,
        num_kv_blocks=num_kv_blocks,
        skip_kv=skip_kv,
        query_block_size=query_block_size,
        batch_index=cache_kwargs.get("batch_index"),
    )
    return output.transpose(1, 2).contiguous(), None
