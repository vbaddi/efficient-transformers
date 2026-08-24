# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Scatter helpers for physically interleaved retained KV caches."""

import onnxscript
import torch

from QEfficient.customop.onnxscript_utils import qeff_custom_op

ops = None


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxChunkScatter(
    data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT
) -> onnxscript.FLOAT:
    batch_size = ops.Gather(ops.Shape(data), [0])
    num_heads = ops.Gather(ops.Shape(updates), [1])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])
    total_rows = ops.Gather(ops.Shape(data), [1])
    ways = ops.Div(total_rows, num_heads)
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    shape = ops.Concat(batch_size, num_heads, seq_len, one, axis=0)
    batch = ops.Expand(ops.Unsqueeze(ops.Range(zero, batch_size, one), [1, 2, 3]), shape)
    head = ops.Expand(ops.Unsqueeze(ops.Range(zero, num_heads, one), [0, 2, 3]), shape)
    positions = ops.Cast(position_ids, to=7)
    row_in_head = ops.Expand(ops.Unsqueeze(ops.Mod(positions, ways), [1, 3]), shape)
    column = ops.Expand(ops.Unsqueeze(ops.Div(positions, ways), [1, 3]), shape)
    row = ops.Add(ops.Mul(ops.Cast(head, to=7), ways), row_in_head)
    return ops.ScatterND(data, ops.Concat(batch, row, column, axis=3), updates)


@qeff_custom_op("com.qualcomm.cloud", 1)
def CtxChunkScatterPacked(
    data: onnxscript.FLOAT,
    position_ids: onnxscript.INT32,
    updates: onnxscript.FLOAT,
    packing: int,
) -> onnxscript.FLOAT:
    batch_size = ops.Gather(ops.Shape(data), [0])
    num_heads = ops.Gather(ops.Shape(updates), [1])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])
    total_rows = ops.Gather(ops.Shape(data), [1])
    ways = ops.Div(total_rows, num_heads)
    packing_value = ops.Constant(value_int=packing)
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    shape = ops.Concat(batch_size, num_heads, seq_len, one, axis=0)
    batch = ops.Expand(ops.Unsqueeze(ops.Range(zero, batch_size, one), [1, 2, 3]), shape)
    head = ops.Expand(ops.Unsqueeze(ops.Range(zero, num_heads, one), [0, 2, 3]), shape)
    positions = ops.Cast(position_ids, to=7)
    row_in_head = ops.Expand(ops.Unsqueeze(ops.Mod(ops.Div(positions, packing_value), ways), [1, 3]), shape)
    column = ops.Expand(
        ops.Unsqueeze(
            ops.Add(
                ops.Mul(ops.Div(positions, ops.Mul(ways, packing_value)), packing_value),
                ops.Mod(positions, packing_value),
            ),
            [1, 3],
        ),
        shape,
    )
    row = ops.Add(ops.Mul(ops.Cast(head, to=7), ops.Cast(ways, to=7)), ops.Cast(row_in_head, to=7))
    return ops.ScatterND(data, ops.Concat(batch, row, column, axis=3), updates)


class CtxChunkScatterFunc(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        batch_size, num_heads, query_len, _ = updates.shape
        ways = data.shape[1] // num_heads
        positions = position_ids.long()
        batch = torch.arange(batch_size, device=data.device).view(-1, 1, 1).expand(-1, num_heads, query_len)
        head = torch.arange(num_heads, device=data.device).view(1, -1, 1).expand(batch_size, -1, query_len)
        row = head * ways + positions.unsqueeze(1) % ways
        column = (positions // ways).unsqueeze(1).expand(-1, num_heads, -1)
        output = data.clone()
        output[batch, row, column] = updates
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def symbolic(g, data, position_ids, updates):
        return g.onnxscript_op(CtxChunkScatter, data, position_ids, updates).setTypeAs(data)


class CtxChunkScatterPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor, packing: int):
        batch_size, num_heads, query_len, _ = updates.shape
        ways = data.shape[1] // num_heads
        positions = position_ids.long()
        batch = torch.arange(batch_size, device=data.device).view(-1, 1, 1).expand(-1, num_heads, query_len)
        head = torch.arange(num_heads, device=data.device).view(1, -1, 1).expand(batch_size, -1, query_len)
        row = head * ways + ((positions // packing) % ways).unsqueeze(1)
        column = ((positions // (packing * ways)) * packing + positions % packing).unsqueeze(1)
        output = data.clone()
        output[batch, row, column.expand(-1, num_heads, -1)] = updates
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def symbolic(g, data, position_ids, updates, packing):
        return g.onnxscript_op(CtxChunkScatterPacked, data, position_ids, updates, packing_i=packing).setTypeAs(data)


def ctx_chunk_scatter(
    data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor, packing: int
) -> torch.Tensor:
    """Scatter logical KV updates into an interleaved cache with ``packing`` tokens per lane."""
    if packing == 1:
        return CtxChunkScatterFunc.apply(data, position_ids, updates)
    return CtxChunkScatterPackedFunc.apply(data, position_ids, updates, packing)
