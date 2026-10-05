# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Load-time layout transforms carried in the weight spec (spec version 6)."""

import os
from collections.abc import Sequence
from typing import Any

import numpy as np

WEIGHT_SPEC_LAYOUT_VERSION = 6
SUPPORTED_LAYOUT_OPS = ("reshape", "transpose")
LAYOUT_TRANSFORMS_ENV = "QEFF_WF_LAYOUT_TRANSFORMS"
LayoutOp = dict[str, Any]


def layout_transforms_enabled() -> bool:
    """Return True when weight-free export may emit v6 layout transforms."""
    return os.environ.get(LAYOUT_TRANSFORMS_ENV, "0") == "1"


def _prod(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def validate_layout_ops(ops: Sequence[LayoutOp]) -> None:
    """Raise ValueError if ``ops`` uses an unknown op or malformed arguments."""
    for index, op in enumerate(ops):
        kind = op.get("op")
        if kind not in SUPPORTED_LAYOUT_OPS:
            raise ValueError(f"Layout op {index} has unsupported type {kind!r}; supported: {SUPPORTED_LAYOUT_OPS}.")
        if kind == "reshape":
            shape = op.get("shape")
            if not isinstance(shape, (list, tuple)) or not shape or any(int(d) <= 0 for d in shape):
                raise ValueError(f"Layout op {index} (reshape) needs a non-empty list of positive dims, got {shape!r}.")
        if kind == "transpose":
            perm = op.get("perm")
            if not isinstance(perm, (list, tuple)) or sorted(int(p) for p in perm) != list(range(len(perm))):
                raise ValueError(f"Layout op {index} (transpose) needs a permutation of range(rank), got {perm!r}.")


def infer_layout_shape(source_shape: Sequence[int], ops: Sequence[LayoutOp]) -> list[int]:
    """Return the shape produced by applying ``ops`` to ``source_shape``."""
    validate_layout_ops(ops)
    shape = [int(d) for d in source_shape]
    block = shape[1:]
    block_rank = len(block)
    for index, op in enumerate(ops):
        if op["op"] == "reshape":
            new_shape = [int(d) for d in op["shape"]]
            if _prod(new_shape) != _prod(shape):
                raise ValueError(f"Layout op {index} (reshape) changes element count: {shape} -> {new_shape}.")
            if len(new_shape) <= block_rank or (block_rank and new_shape[-block_rank:] != block):
                raise ValueError(
                    f"Layout op {index} (reshape) must keep the block shape {block} as trailing dims, got {new_shape}."
                )
            shape = new_shape
        else:
            perm = [int(p) for p in op["perm"]]
            if len(perm) != len(shape):
                raise ValueError(f"Layout op {index} (transpose) perm rank {len(perm)} != tensor rank {len(shape)}.")
            lead = len(shape) - block_rank
            if perm[lead:] != list(range(lead, len(shape))):
                raise ValueError(
                    f"Layout op {index} (transpose) may only permute leading axes; "
                    f"the last {block_rank} axes must stay in place, got perm {perm}."
                )
            shape = [shape[p] for p in perm]
    return shape


def apply_layout_ops(array: np.ndarray, ops: Sequence[LayoutOp]) -> np.ndarray:
    """Apply ``ops`` to ``array`` and return a contiguous result."""
    infer_layout_shape(array.shape, ops)
    for op in ops:
        if op["op"] == "reshape":
            array = np.reshape(array, [int(d) for d in op["shape"]], order="C")
        else:
            array = np.transpose(array, [int(p) for p in op["perm"]])
    return np.ascontiguousarray(array)


def expert_parallel_layout_ops(
    canonical_shape: Sequence[int], num_pipeline_stages: int, num_parallelized_experts: int
) -> list[LayoutOp]:
    """Return ops that turn canonical ``[E, ...]`` into ``[E/P, P, ...]``."""
    num_experts, *rest = [int(d) for d in canonical_shape]
    if num_experts != num_pipeline_stages * num_parallelized_experts:
        raise ValueError(
            f"Expert-parallel layout needs E == P * E/P, got E={num_experts}, "
            f"P={num_pipeline_stages}, E/P={num_parallelized_experts}."
        )
    rank = 2 + len(rest)
    return [
        {"op": "reshape", "shape": [num_pipeline_stages, num_parallelized_experts, *rest]},
        {"op": "transpose", "perm": [1, 0, *range(2, rank)]},
    ]
