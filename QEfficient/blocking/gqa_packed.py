# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Physically packed KV-cache schedules used by GQA layer benchmarks."""

from __future__ import annotations

from typing import Optional

import torch

from QEfficient.customop.utils import ctx_gather_blocked_kv
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


def _read_block(
    cache: torch.Tensor,
    position_ids: torch.Tensor,
    start: int,
    end: int,
    *,
    packing: int,
    ways: int,
    zero_invalid: bool,
) -> torch.Tensor:
    batch_size, rows, _, _ = cache.shape
    indices = torch.arange(start, end, device=cache.device).view(1, 1, -1)
    maximum_position = position_ids.max(dim=1, keepdim=True).values
    maximum_column = (maximum_position // (packing * ways)) * packing + packing - 1
    invalid = indices > maximum_column.unsqueeze(1)
    invalid_index = torch.iinfo(torch.int32).max if torch.onnx.is_in_onnx_export() else 0
    indices = torch.where(invalid, invalid_index, indices).to(torch.int32).expand(batch_size, rows, -1)
    return ctx_gather_blocked_kv(cache, indices)


def _packed_positions(start: int, length: int, rows: int, ways: int, packing: int, device) -> torch.Tensor:
    slots = start + torch.arange(length, device=device)
    row_in_head = torch.arange(rows, device=device) % ways
    return (
        (slots.unsqueeze(0) // packing) * packing * ways
        + row_in_head.unsqueeze(1) * packing
        + slots.unsqueeze(0) % packing
    )


def _decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    scaling: float,
    num_kv_blocks: int,
    split: int,
    chunk_kv_n: int,
    chunk_kv_size: int,
    skip_kv: bool,
    kv_block_unroll: int,
) -> torch.Tensor:
    batch_size, num_query_heads, query_len, head_dim = query.shape
    if query_len != 1:
        raise ValueError("Packed decode requires a single query token.")
    ways = chunk_kv_n * split
    num_kv_heads = key_cache.shape[1] // ways
    groups = num_query_heads // num_kv_heads
    packing = chunk_kv_size // split
    cache_depth = key_cache.shape[2]
    block_slots = -(-cache_depth // num_kv_blocks)
    is_export = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
    outputs = []

    for batch in range(batch_size):
        folded_query = query[batch : batch + 1].reshape(1, num_kv_heads, groups, head_dim)
        lane_query = (
            folded_query.repeat_interleave(chunk_kv_n, dim=1)
            .unsqueeze(2)
            .expand(1, num_kv_heads * chunk_kv_n, split, groups, head_dim)
        )
        maxima = []
        denominators = []
        numerators = []
        stop = False
        for group_start in range(0, num_kv_blocks, max(1, kv_block_unroll)):
            active = []
            for block in range(group_start, min(group_start + max(1, kv_block_unroll), num_kv_blocks)):
                start = block * block_slots
                end = min(start + block_slots, cache_depth)
                skip_future = (
                    torch.tensor((start // packing) * packing * ways, device=query.device) > position_ids[batch].max()
                )
                if skip_kv and not is_export and skip_future.item():
                    stop = True
                    break
                active.append((start, end, skip_future))
            key_blocks = [
                _read_block(
                    key_cache[batch : batch + 1],
                    position_ids[batch : batch + 1],
                    start,
                    end,
                    packing=packing,
                    ways=ways,
                    zero_invalid=False,
                )
                for start, end, _ in active
            ]
            exponentials = []
            for keys, (start, end, skip_future) in zip(key_blocks, active):
                slot_count = end - start
                keys = keys.view(1, num_kv_heads * chunk_kv_n, split, slot_count, head_dim)
                scores = torch.matmul(lane_query, keys.transpose(-1, -2)) * scaling
                positions = _packed_positions(start, slot_count, ways, ways, packing, query.device).view(
                    1, chunk_kv_n, split, 1, slot_count
                )
                mask = positions.unsqueeze(1).expand(1, num_kv_heads, -1, -1, groups, -1).reshape_as(scores)
                mask = mask > position_ids[batch, 0]
                scores = torch.where(
                    mask,
                    torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=scores.dtype, device=scores.device),
                    scores,
                )
                block_maximum = scores.max(dim=-1).values
                row_skipped = mask.all(dim=-1)
                safe_maximum = torch.where(row_skipped, torch.zeros_like(block_maximum), block_maximum)
                block_exp = torch.exp(scores - safe_maximum.unsqueeze(-1))
                block_exp = torch.where(row_skipped.unsqueeze(-1), torch.zeros_like(block_exp), block_exp)
                block_maximum = torch.where(row_skipped, torch.full_like(block_maximum, float("-inf")), block_maximum)
                block_sum = block_exp.sum(dim=-1)
                if skip_kv and is_export:
                    block_maximum = torch.where(
                        skip_future, torch.full_like(block_maximum, float("-inf")), block_maximum
                    )
                    block_exp = torch.where(skip_future, torch.zeros_like(block_exp), block_exp)
                    block_sum = torch.where(skip_future, torch.zeros_like(block_sum), block_sum)
                maxima.append(block_maximum)
                denominators.append(block_sum)
                exponentials.append(block_exp)
            value_blocks = [
                _read_block(
                    value_cache[batch : batch + 1],
                    position_ids[batch : batch + 1],
                    start,
                    end,
                    packing=packing,
                    ways=ways,
                    zero_invalid=True,
                ).view(1, num_kv_heads * chunk_kv_n, split, end - start, head_dim)
                for start, end, _ in active
            ]
            for block_exp, values, (_, _, skip_future) in zip(exponentials, value_blocks, active):
                block_output = torch.matmul(block_exp, values)
                if skip_kv and is_export:
                    block_output = torch.where(skip_future, torch.zeros_like(block_output), block_output)
                numerators.append(block_output)
            if stop:
                break

        block_maxima = torch.stack(maxima)
        maximum = block_maxima.max(dim=0).values
        weights = torch.exp(block_maxima - maximum.unsqueeze(0))
        denominator = (weights * torch.stack(denominators)).sum(dim=0)
        numerator = (weights.unsqueeze(-1) * torch.stack(numerators)).sum(dim=0)
        split_maximum = maximum.max(dim=2).values
        split_weights = torch.exp(maximum - split_maximum.unsqueeze(2))
        denominator = (split_weights * denominator).sum(dim=2)
        numerator = (split_weights.unsqueeze(-1) * numerator).sum(dim=2)
        split_maximum = split_maximum.view(1, num_kv_heads, chunk_kv_n, groups)
        denominator = denominator.view(1, num_kv_heads, chunk_kv_n, groups)
        numerator = numerator.view(1, num_kv_heads, chunk_kv_n, groups, head_dim)
        final_maximum = split_maximum.max(dim=2).values
        final_weights = torch.exp(split_maximum - final_maximum.unsqueeze(2))
        denominator = (final_weights * denominator).sum(dim=2)
        numerator = (final_weights.unsqueeze(-1) * numerator).sum(dim=2)
        outputs.append((numerator / denominator.unsqueeze(-1)).view(1, num_query_heads, 1, head_dim))
    return torch.cat(outputs, dim=0)


def _running_softmax(
    maximum: torch.Tensor,
    scores: torch.Tensor,
    denominator: torch.Tensor,
    numerator: torch.Tensor,
    values: torch.Tensor,
    skipped: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    updated_maximum = torch.maximum(maximum, scores.max(dim=-1).values)
    previous_weight = torch.exp(maximum - updated_maximum)
    exponentials = torch.exp(scores - updated_maximum.unsqueeze(-1))
    updated_denominator = denominator * previous_weight + exponentials.sum(dim=-1)
    updated_numerator = numerator * previous_weight.unsqueeze(-1) + torch.matmul(exponentials, values)
    if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
        return (
            torch.where(skipped, maximum, updated_maximum),
            torch.where(skipped, denominator, updated_denominator),
            torch.where(skipped.unsqueeze(-1), numerator, updated_numerator),
        )
    return updated_maximum, updated_denominator, updated_numerator


def _prefill(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    scaling: float,
    num_kv_blocks: int,
    split: int,
    chunk_kv_n: int,
    chunk_kv_size: int,
    skip_kv: bool,
    query_block_size: Optional[int],
    query_head_block_size: int,
) -> torch.Tensor:
    batch_size, num_query_heads, query_len, head_dim = query.shape
    ways = chunk_kv_n * split
    num_kv_heads = key_cache.shape[1] // ways
    groups = num_query_heads // num_kv_heads
    packing = chunk_kv_size // split
    cache_depth = key_cache.shape[2]
    block_slots = -(-cache_depth // num_kv_blocks)
    target_heads = split
    rows = key_cache.shape[1]
    fold = rows // target_heads
    lanes_per_head = target_heads // num_kv_heads
    if rows % target_heads or target_heads % num_kv_heads or ways % fold:
        raise ValueError("Packed prefill cache rows cannot be folded onto the selected core split.")
    if groups % query_head_block_size:
        raise ValueError("q_head_block_chunk must divide the number of query groups.")
    query_block_size = query_len if not query_block_size else query_block_size
    query = query.reshape(batch_size, num_kv_heads, groups, query_len, head_dim)
    is_export = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
    masked = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=query.dtype, device=query.device)
    output_chunks = []

    for query_start in range(0, query_len, query_block_size):
        query_end = min(query_start + query_block_size, query_len)
        chunk_length = query_end - query_start
        positions = position_ids[:, query_start:query_end]
        accumulators = []
        for group_start in range(0, groups, query_head_block_size):
            group_end = group_start + query_head_block_size
            group_count = group_end - group_start
            lane_query = (
                query[:, :, group_start:group_end, query_start:query_end]
                .reshape(batch_size, num_kv_heads, group_count * chunk_length, head_dim)
                .repeat_interleave(lanes_per_head, dim=1)
            )
            accumulators.append(
                {
                    "groups": group_count,
                    "query": lane_query,
                    "maximum": torch.full(
                        (batch_size, target_heads, group_count * chunk_length),
                        float("-inf"),
                        dtype=query.dtype,
                        device=query.device,
                    ),
                    "denominator": torch.zeros(
                        batch_size, target_heads, group_count * chunk_length, dtype=query.dtype, device=query.device
                    ),
                    "numerator": torch.zeros(
                        batch_size,
                        target_heads,
                        group_count * chunk_length,
                        head_dim,
                        dtype=query.dtype,
                        device=query.device,
                    ),
                }
            )
        for block in range(num_kv_blocks):
            start = block * block_slots
            end = min(start + block_slots, cache_depth)
            skip_future = (
                torch.tensor((start // packing) * packing * ways, device=query.device) > positions.max(dim=-1).values
            )
            if skip_kv and not is_export and skip_future.all().item():
                break
            slot_count = end - start
            keys = _read_block(
                key_cache,
                position_ids,
                start,
                end,
                packing=packing,
                ways=ways,
                zero_invalid=False,
            ).view(batch_size, target_heads, fold * slot_count, head_dim)
            values = _read_block(
                value_cache,
                position_ids,
                start,
                end,
                packing=packing,
                ways=ways,
                zero_invalid=True,
            ).view(batch_size, target_heads, fold * slot_count, head_dim)
            absolute_positions = _packed_positions(start, slot_count, rows, ways, packing, query.device).view(
                target_heads, fold * slot_count
            )
            causal = absolute_positions[None, :, None, :] > positions[:, None, :, None]
            lane_skipped = causal.all(dim=-1)
            for accumulator in accumulators:
                group_count = accumulator["groups"]
                scores = torch.matmul(accumulator["query"], keys.transpose(-1, -2)) * scaling
                scores = torch.where(causal.repeat(1, 1, group_count, 1), masked, scores)
                skipped = lane_skipped.repeat(1, 1, group_count)
                if skip_kv and is_export:
                    skipped = skipped | skip_future[:, None, None]
                accumulator["maximum"], accumulator["denominator"], accumulator["numerator"] = _running_softmax(
                    accumulator["maximum"],
                    scores,
                    accumulator["denominator"],
                    accumulator["numerator"],
                    values,
                    skipped,
                )
        group_outputs = []
        for accumulator in accumulators:
            group_count = accumulator["groups"]
            maximum = accumulator["maximum"].view(batch_size, num_kv_heads, lanes_per_head, group_count * chunk_length)
            denominator = accumulator["denominator"].view_as(maximum)
            numerator = accumulator["numerator"].view(
                batch_size, num_kv_heads, lanes_per_head, group_count * chunk_length, head_dim
            )
            merged_maximum = maximum.max(dim=2).values
            weights = torch.exp(maximum - merged_maximum.unsqueeze(2))
            merged_denominator = (weights * denominator).sum(dim=2)
            merged_numerator = (weights.unsqueeze(-1) * numerator).sum(dim=2)
            group_outputs.append(
                (merged_numerator / merged_denominator.unsqueeze(-1)).view(
                    batch_size, num_kv_heads, group_count, chunk_length, head_dim
                )
            )
        output_chunks.append(torch.cat(group_outputs, dim=2))
    return torch.cat(output_chunks, dim=3).reshape(batch_size, num_query_heads, query_len, head_dim)


def gqa_packed_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    phase: str,
    scaling: float,
    num_kv_blocks: int,
    split: int,
    chunk_kv_n: int,
    chunk_kv_size: int,
    skip_kv: bool,
    query_block_size: Optional[int],
    query_head_block_size: int,
    kv_block_unroll: int,
) -> torch.Tensor:
    common = {
        "scaling": scaling,
        "num_kv_blocks": num_kv_blocks,
        "split": split,
        "chunk_kv_n": chunk_kv_n,
        "chunk_kv_size": chunk_kv_size,
        "skip_kv": skip_kv,
    }
    if phase == "prefill":
        return _prefill(
            query,
            key_cache,
            value_cache,
            position_ids,
            query_block_size=query_block_size,
            query_head_block_size=query_head_block_size,
            **common,
        )
    return _decode(
        query,
        key_cache,
        value_cache,
        position_ids,
        kv_block_unroll=kv_block_unroll,
        **common,
    )
