# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Fixture conversions for GQA's physically packed KV-cache implementations."""

import torch


def pack_gqa_cache(cache: torch.Tensor, *, ways: int, packing: int) -> torch.Tensor:
    """Convert ``[B, Hkv, context, D]`` to h-major packed cache storage."""
    batch_size, num_heads, context_length, head_dim = cache.shape
    slots = -(-context_length // ways)
    positions = torch.arange(context_length, device=cache.device)
    row_in_head = (positions // packing) % ways
    columns = (positions // (packing * ways)) * packing + positions % packing
    output = torch.zeros(
        batch_size,
        num_heads * ways,
        slots,
        head_dim,
        dtype=cache.dtype,
        device=cache.device,
    )
    batches = torch.arange(batch_size, device=cache.device)[:, None].expand(batch_size, context_length)
    for head in range(num_heads):
        rows = head * ways + row_in_head
        output[batches, rows.expand(batch_size, -1), columns.expand(batch_size, -1)] = cache[:, head]
    return output


def unpack_gqa_cache(
    cache: torch.Tensor, *, context_length: int, num_heads: int, ways: int, packing: int
) -> torch.Tensor:
    """Recover logical ``[B, Hkv, context, D]`` values from packed storage."""
    batch_size = cache.shape[0]
    positions = torch.arange(context_length, device=cache.device)
    row_in_head = (positions // packing) % ways
    columns = (positions // (packing * ways)) * packing + positions % packing
    batches = torch.arange(batch_size, device=cache.device)[:, None, None].expand(batch_size, num_heads, context_length)
    rows = (torch.arange(num_heads, device=cache.device)[:, None] * ways + row_in_head[None])[None].expand(
        batch_size, -1, -1
    )
    return cache[batches, rows, columns[None, None].expand(batch_size, num_heads, -1)]
