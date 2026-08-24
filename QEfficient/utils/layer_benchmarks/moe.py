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
from typing import Any, Mapping

import torch

from QEfficient.blocking.moe import (
    DECODE_MOE_IMPLEMENTATIONS,
    MOE_IMPLEMENTATIONS,
    ROUTED_MOE_IMPLEMENTATIONS,
    ExpertBlockedMoE,
    GatherBmmMoE,
)
from QEfficient.utils.layer_benchmarks.contracts import BenchmarkCase, LayerBenchmark


@dataclass(frozen=True)
class MoEConfig:
    implementation: str
    batch_size: int
    seq_len: int
    hidden_size: int
    intermediate_size: int
    num_nsp: int
    local_experts: int
    total_experts: int
    num_experts_per_token: int
    avg_valid_rows: int
    pattern: str
    packed_chunk_size: int
    ffn_blocking_mode: str
    ffn_token_block_size: int | None
    ffn_weight_block_size: int | None
    experts_per_soc: int | None
    tree_reduce: bool
    dtype: str
    dynamo: bool


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _mask(tokens: int, valid_rows: int, pattern: str, generator: torch.Generator) -> torch.Tensor:
    result = torch.zeros(tokens, dtype=torch.bool)
    if valid_rows == 0:
        return result
    if pattern == "frontloaded":
        result[:valid_rows] = True
    elif pattern == "even":
        positions = torch.linspace(0, tokens - 1, steps=valid_rows).round().long().unique()
        result[positions] = True
        if result.sum() != valid_rows:
            result.zero_()
            result[torch.randperm(tokens, generator=generator)[:valid_rows]] = True
    else:
        result[torch.randperm(tokens, generator=generator)[:valid_rows]] = True
    return result


class MoEBenchmark(LayerBenchmark):
    name = "moe"
    description = "Expert-blocked NSP-parallel MoE prefill and decode kernels."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--implementation", "--impls", dest="implementation", choices=MOE_IMPLEMENTATIONS, required=True
        )
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument("--seq-len", "--T", dest="seq_len", type=int, default=16)
        parser.add_argument("--hidden-size", "--H", dest="hidden_size", type=int, default=64)
        parser.add_argument("--intermediate-size", "--I", dest="intermediate_size", type=int, default=128)
        parser.add_argument("--num-nsp", type=int, default=4)
        parser.add_argument("--local-experts", type=int, default=2)
        parser.add_argument("--total-experts", type=int)
        parser.add_argument(
            "--num-experts-per-token", "--num-experts-per-tok", dest="num_experts_per_token", type=int, default=2
        )
        parser.add_argument("--avg-valid-rows", type=int)
        parser.add_argument("--pattern", choices=("random", "frontloaded", "even"), default="random")
        parser.add_argument("--packed-chunk-size", type=int, default=8)
        parser.add_argument(
            "--ffn-blocking-mode", choices=("default", "token", "weight", "token_weight"), default="default"
        )
        parser.add_argument("--ffn-token-block-size", type=int)
        parser.add_argument("--ffn-weight-block-size", type=int)
        parser.add_argument("--experts-per-soc", type=int)
        parser.add_argument("--tree-reduce", action="store_true")
        parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
        parser.add_argument("--dynamo", action=argparse.BooleanOptionalAction, default=False)

    def resolved_config(self, args: argparse.Namespace) -> Mapping[str, Any]:
        total_experts = args.total_experts or args.num_nsp * args.local_experts
        tokens = args.batch_size * args.seq_len
        avg_valid_rows = args.avg_valid_rows
        if avg_valid_rows is None:
            avg_valid_rows = max(1, round(tokens * args.num_experts_per_token / total_experts))
        if args.dynamo:
            raise ValueError("The integrated MoE benchmark currently requires --no-dynamo (legacy ONNX export).")
        if (
            min(
                args.batch_size,
                args.seq_len,
                args.hidden_size,
                args.intermediate_size,
                args.num_nsp,
                args.local_experts,
            )
            <= 0
        ):
            raise ValueError("MoE dimensions must be positive.")
        if total_experts != args.num_nsp * args.local_experts:
            raise ValueError("--total-experts must equal --num-nsp * --local-experts.")
        if not 0 < args.num_experts_per_token <= total_experts:
            raise ValueError("--num-experts-per-token must be in [1, total_experts].")
        if not 0 <= avg_valid_rows <= tokens:
            raise ValueError("--avg-valid-rows must be in [0, batch_size * seq_len].")
        if args.packed_chunk_size <= 0:
            raise ValueError("--packed-chunk-size must be positive.")
        if args.experts_per_soc is not None and args.num_nsp % args.experts_per_soc:
            raise ValueError("--experts-per-soc must divide --num-nsp.")
        if args.tree_reduce and args.experts_per_soc is None:
            raise ValueError("--tree-reduce requires --experts-per-soc.")
        if args.implementation in DECODE_MOE_IMPLEMENTATIONS and args.seq_len != 1:
            raise ValueError("Decode MoE implementations require --seq-len/--T 1.")
        return asdict(
            MoEConfig(
                implementation=args.implementation,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                num_nsp=args.num_nsp,
                local_experts=args.local_experts,
                total_experts=total_experts,
                num_experts_per_token=args.num_experts_per_token,
                avg_valid_rows=avg_valid_rows,
                pattern=args.pattern,
                packed_chunk_size=args.packed_chunk_size,
                ffn_blocking_mode=args.ffn_blocking_mode,
                ffn_token_block_size=args.ffn_token_block_size,
                ffn_weight_block_size=args.ffn_weight_block_size,
                experts_per_soc=args.experts_per_soc,
                tree_reduce=args.tree_reduce,
                dtype=args.dtype,
                dynamo=False,
            )
        )

    def build_case(self, config: Mapping[str, Any], seed: int) -> BenchmarkCase:
        resolved = MoEConfig(**config)
        generator = torch.Generator().manual_seed(seed)
        dtype = _dtype(resolved.dtype)
        tokens = resolved.batch_size * resolved.seq_len
        if resolved.implementation in DECODE_MOE_IMPLEMENTATIONS:
            module = GatherBmmMoE(
                resolved.total_experts,
                resolved.hidden_size,
                resolved.intermediate_size,
                resolved.num_experts_per_token,
                dtype,
                loop=resolved.implementation.endswith("_loop"),
            ).eval()
            x = torch.randn(tokens, resolved.hidden_size, generator=generator).to(dtype)
            indices = torch.stack(
                [
                    torch.randperm(resolved.total_experts, generator=generator)[: resolved.num_experts_per_token]
                    for _ in range(tokens)
                ]
            ).to(torch.int32)
            weights = torch.softmax(
                torch.randn(tokens, resolved.num_experts_per_token, generator=generator), dim=-1
            ).to(dtype)
            inputs = OrderedDict(x=x, router_indices=indices, router_weights=weights)
            dynamic_axes = {
                "x": {0: "batch_size"},
                "router_indices": {0: "batch_size"},
                "router_weights": {0: "batch_size"},
                "output": {0: "batch_size"},
            }
        else:
            module = ExpertBlockedMoE(
                resolved.implementation,
                resolved.num_nsp,
                resolved.local_experts,
                resolved.hidden_size,
                resolved.intermediate_size,
                dtype,
                resolved.packed_chunk_size,
                resolved.ffn_blocking_mode,
                resolved.ffn_token_block_size,
                resolved.ffn_weight_block_size,
                resolved.num_experts_per_token,
                resolved.experts_per_soc,
                resolved.tree_reduce,
            ).eval()
            if (
                resolved.implementation == "cumsum_scatter_gather_update_with_router"
                or resolved.implementation in ROUTED_MOE_IMPLEMENTATIONS
            ):
                x = torch.randn(resolved.batch_size, resolved.seq_len, resolved.hidden_size, generator=generator).to(
                    dtype
                )
                inputs = OrderedDict(x=x)
                dynamic_axes = {"x": {0: "batch_size", 1: "seq_len"}, "output": {0: "tokens"}}
            else:
                x = torch.randn(tokens, resolved.hidden_size, generator=generator).to(dtype)
                masks = (
                    torch.stack(
                        [
                            _mask(tokens, resolved.avg_valid_rows, resolved.pattern, generator)
                            for _ in range(resolved.total_experts)
                        ]
                    )
                    .view(resolved.local_experts, resolved.num_nsp, tokens)
                    .transpose(0, 1)
                    .contiguous()
                )
                inputs = OrderedDict(x=x, local_t2e=masks)
                dynamic_axes = {"x": {0: "tokens"}, "local_t2e": {2: "tokens"}, "output": {0: "tokens"}}
        return BenchmarkCase(
            module=module,
            reference=module.reference,
            inputs=inputs,
            output_names=("output",),
            dynamic_axes=dynamic_axes,
            specializations={
                "batch_size": resolved.batch_size,
                "seq_len": resolved.seq_len,
                "tokens": tokens,
            },
        )
