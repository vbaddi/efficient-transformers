# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""MoE kernels shared by QEff model integrations and layer benchmarks."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from QEfficient.customop.utils import (
    ctx_gather_3d,
    ctx_gather_3d_generalized,
    ctx_scatter_3d_generalized,
    ctx_scatter_3d_int,
)

PREFILL_MOE_IMPLEMENTATIONS = (
    "reference",
    "gather",
    "packed_chunk",
    "packed_chunk_post_gather",
    "cumsum_scatter_gather_update",
    "cumsum_scatter_gather_update_with_router",
)
ROUTED_MOE_IMPLEMENTATIONS = ("decode_qwen3vl_cumsum_scatter_gather_update_with_router",)
DECODE_MOE_IMPLEMENTATIONS = ("decode_gather_bmm", "decode_gather_bmm_loop")
MOE_IMPLEMENTATIONS = PREFILL_MOE_IMPLEMENTATIONS + ROUTED_MOE_IMPLEMENTATIONS + DECODE_MOE_IMPLEMENTATIONS


def _expert_mlp(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    mode: str,
    token_block_size: int | None,
    weight_block_size: int | None,
) -> torch.Tensor:
    token_block_size = x.shape[1] if token_block_size is None else min(token_block_size, x.shape[1])
    weight_block_size = (
        down_weight.shape[1] if weight_block_size is None else min(weight_block_size, down_weight.shape[1])
    )

    def weight_blocked(token_block: torch.Tensor) -> torch.Tensor:
        pieces = []
        for start in range(0, down_weight.shape[1], weight_block_size):
            stop = min(start + weight_block_size, down_weight.shape[1])
            gate = token_block @ gate_weight[:, :, start:stop]
            up = token_block @ up_weight[:, :, start:stop]
            pieces.append((up * F.silu(gate)) @ down_weight[:, start:stop, :])
        return torch.stack(pieces).sum(dim=0)

    def full(token_block: torch.Tensor) -> torch.Tensor:
        if mode in {"weight", "token_weight"}:
            return weight_blocked(token_block)
        gate = token_block @ gate_weight
        up = token_block @ up_weight
        return (up * F.silu(gate)) @ down_weight

    if mode not in {"token", "token_weight"}:
        return full(x)
    return torch.cat(
        [full(x[:, start : start + token_block_size]) for start in range(0, x.shape[1], token_block_size)], dim=1
    )


def _tree_sum(values: torch.Tensor) -> torch.Tensor:
    width = values.shape[0]
    while width > 1:
        pairs = width // 2
        reduced = values[0 : 2 * pairs : 2] + values[1 : 2 * pairs : 2]
        if width % 2:
            reduced = torch.cat((reduced, values[width - 1 : width]), dim=0)
        values = reduced
        width = pairs + width % 2
    return values[0]


def _indices_from_mask(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = (torch.cumsum(mask.to(torch.int64), dim=1) - 1).to(torch.int32)
    invalid = ~mask
    invalid_index = torch.tensor(torch.iinfo(torch.int32).max, dtype=torch.int32, device=mask.device)
    return packed, invalid, invalid_index


def _scatter_gather_expert(
    x: torch.Tensor,
    mask: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    mode: str,
    token_block_size: int | None,
    weight_block_size: int | None,
) -> torch.Tensor:
    lanes, tokens = mask.shape
    packed, invalid, invalid_index = _indices_from_mask(mask)
    safe = torch.where(invalid, invalid_index, packed)
    template = torch.zeros(lanes, tokens, x.shape[-1], dtype=x.dtype, device=x.device)
    packed_x = ctx_scatter_3d_generalized(template, safe, x.unsqueeze(0).expand(lanes, -1, -1))
    output = _expert_mlp(packed_x, gate_weight, up_weight, down_weight, mode, token_block_size, weight_block_size)
    valid_rows = mask.to(torch.int32).sum(dim=1, keepdim=True)
    rows = torch.arange(tokens, dtype=torch.int32, device=x.device).unsqueeze(0)
    output = torch.where((rows < valid_rows).unsqueeze(-1), output, torch.zeros_like(output))
    output = ctx_gather_3d(output, safe)
    return torch.where(invalid.unsqueeze(-1), torch.zeros_like(output), output)


def _gather_expert(
    x: torch.Tensor,
    mask: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    mode: str,
    token_block_size: int | None,
    weight_block_size: int | None,
) -> torch.Tensor:
    lanes, tokens = mask.shape
    packed, invalid, invalid_index = _indices_from_mask(mask)
    source = torch.arange(tokens, device=x.device).unsqueeze(0).expand(lanes, -1)
    destination = torch.where(mask, packed.long(), torch.full_like(packed, tokens - 1, dtype=torch.int64))
    source = torch.where(mask, source, torch.zeros_like(source))
    gather_index = torch.zeros(lanes, tokens, dtype=torch.int64, device=x.device)
    gather_index.scatter_(1, destination, source)
    packed_x = torch.gather(
        x.unsqueeze(0).expand(lanes, -1, -1), 1, gather_index.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    )
    output = _expert_mlp(packed_x, gate_weight, up_weight, down_weight, mode, token_block_size, weight_block_size)
    valid_rows = mask.to(torch.int32).sum(dim=1, keepdim=True)
    rows = torch.arange(tokens, dtype=torch.int32, device=x.device).unsqueeze(0)
    output = torch.where((rows < valid_rows).unsqueeze(-1), output, torch.zeros_like(output))
    output = ctx_gather_3d(output, torch.where(invalid, invalid_index, packed))
    return torch.where(invalid.unsqueeze(-1), torch.zeros_like(output), output)


def _packed_expert(
    x: torch.Tensor,
    mask: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    chunk_size: int,
    mode: str,
    token_block_size: int | None,
    weight_block_size: int | None,
    *,
    post_gather: bool,
) -> torch.Tensor:
    lanes, tokens = mask.shape
    chunk_size = min(chunk_size, tokens)
    packed, invalid, invalid_index = _indices_from_mask(mask)
    valid_rows = mask.to(torch.int32).sum(dim=1, keepdim=True)
    rows = torch.arange(chunk_size, dtype=torch.int32, device=x.device).unsqueeze(0)
    expanded_x = x.unsqueeze(0).expand(lanes, -1, -1)
    chunks = []
    accumulated = torch.zeros(lanes, tokens, x.shape[-1], dtype=x.dtype, device=x.device)
    for start in range(0, tokens, chunk_size):
        in_chunk = mask & (packed >= start) & (packed < start + chunk_size)
        chunk_index = torch.where(in_chunk, packed - start, invalid_index)
        chunk = ctx_scatter_3d_generalized(
            torch.zeros(lanes, chunk_size, x.shape[-1], dtype=x.dtype, device=x.device),
            chunk_index,
            expanded_x,
        )
        output = _expert_mlp(chunk, gate_weight, up_weight, down_weight, mode, token_block_size, weight_block_size)
        output = torch.where(
            (rows < torch.clamp(valid_rows - start, min=0, max=chunk_size)).unsqueeze(-1),
            output,
            torch.zeros_like(output),
        )
        if post_gather:
            chunks.append(output)
        else:
            gathered = ctx_gather_3d_generalized(output, chunk_index)
            accumulated = accumulated + torch.where(in_chunk.unsqueeze(-1), gathered, torch.zeros_like(gathered))
    if not post_gather:
        return accumulated
    packed_output = torch.cat(chunks, dim=1)[:, :tokens]
    output = ctx_gather_3d(packed_output, torch.where(invalid, invalid_index, packed))
    return torch.where(invalid.unsqueeze(-1), torch.zeros_like(output), output)


def _matched_indices(mask: torch.Tensor) -> torch.Tensor:
    lanes, tokens = mask.shape
    invalid_index = torch.tensor(torch.iinfo(torch.int32).max, dtype=torch.int32, device=mask.device)
    destination = torch.where(mask, torch.cumsum(mask.to(torch.int32), dim=1) - 1, invalid_index)
    token_index = torch.arange(tokens, dtype=torch.int32, device=mask.device).unsqueeze(0).expand(lanes, -1)
    matched = torch.full_like(token_index, torch.iinfo(torch.int32).max)
    return ctx_scatter_3d_int(matched.unsqueeze(-1), destination, token_index.unsqueeze(-1)).squeeze(-1)


def _cumsum_update_expert(
    x: torch.Tensor,
    mask: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    expert_output: torch.Tensor,
    chunk_size: int,
    mode: str,
    token_block_size: int | None,
    weight_block_size: int | None,
    router_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    lanes, tokens = mask.shape
    chunk_size = min(chunk_size, tokens)
    matched = _matched_indices(mask)
    valid_rows = mask.to(torch.int32).sum(dim=1, keepdim=True)
    rows = torch.arange(chunk_size, dtype=torch.int32, device=x.device).unsqueeze(0)
    expanded_x = x.unsqueeze(0).expand(lanes, -1, -1)
    for start in range(0, tokens, chunk_size):
        chunk_index = matched[:, start : start + chunk_size]
        packed_x = ctx_gather_3d_generalized(expanded_x, chunk_index)
        output = _expert_mlp(packed_x, gate_weight, up_weight, down_weight, mode, token_block_size, weight_block_size)
        if router_weight is not None:
            output = output * ctx_gather_3d_generalized(router_weight, chunk_index)
        prior = ctx_gather_3d_generalized(expert_output, chunk_index)
        updated = prior + output
        updated = torch.where(
            (rows < torch.clamp(valid_rows - start, min=0, max=chunk_size)).unsqueeze(-1),
            updated,
            torch.zeros_like(updated),
        )
        expert_output = ctx_scatter_3d_generalized(expert_output, chunk_index, updated)
    return expert_output


class _SigmoidTopKRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        dtype: torch.dtype,
        scaling_factor: float = 2.827,
    ):
        super().__init__()
        self.top_k = top_k
        self.scaling_factor = scaling_factor
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(num_experts, dtype=dtype))
        with torch.no_grad():
            initialized = torch.empty_like(self.weight, dtype=torch.float32)
            nn.init.kaiming_uniform_(initialized, a=math.sqrt(5))
            self.weight.copy_(initialized.to(dtype))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.flatten(0, 1), self.weight).sigmoid()
        _, indices = torch.topk(scores + self.bias, self.top_k, dim=-1, sorted=False)
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * self.scaling_factor
        return indices, weights


class _SoftmaxTopKRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int, dtype: torch.dtype):
        super().__init__()
        self.top_k = top_k
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size, dtype=dtype))
        with torch.no_grad():
            initialized = torch.empty_like(self.weight, dtype=torch.float32)
            nn.init.kaiming_uniform_(initialized, a=math.sqrt(5))
            self.weight.copy_(initialized.to(dtype))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.flatten(0, 1), self.weight).float().softmax(dim=-1)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        return indices, (weights / weights.sum(dim=-1, keepdim=True)).to(x.dtype)


class ExpertBlockedMoE(nn.Module):
    """Expert-parallel prefill kernel with selectable packing schedule."""

    def __init__(
        self,
        implementation: str,
        num_nsp: int,
        local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype,
        packed_chunk_size: int,
        ffn_blocking_mode: str = "default",
        ffn_token_block_size: int | None = None,
        ffn_weight_block_size: int | None = None,
        num_experts_per_token: int = 2,
        experts_per_soc: int | None = None,
        tree_reduce: bool = False,
    ):
        super().__init__()
        self.implementation = implementation
        self.num_nsp = num_nsp
        self.local_experts = local_experts
        self.packed_chunk_size = packed_chunk_size
        self.ffn_blocking_mode = ffn_blocking_mode
        self.ffn_token_block_size = ffn_token_block_size
        self.ffn_weight_block_size = ffn_weight_block_size
        self.experts_per_soc = experts_per_soc
        self.tree_reduce = tree_reduce
        self.gate_weight = nn.Parameter(
            torch.randn(num_nsp, local_experts, hidden_size, intermediate_size, dtype=dtype) * 0.02
        )
        self.up_weight = nn.Parameter(
            torch.randn(num_nsp, local_experts, hidden_size, intermediate_size, dtype=dtype) * 0.02
        )
        self.down_weight = nn.Parameter(
            torch.randn(num_nsp, local_experts, intermediate_size, hidden_size, dtype=dtype) * 0.02
        )
        if implementation == "cumsum_scatter_gather_update_with_router":
            self.router = _SigmoidTopKRouter(hidden_size, num_nsp * local_experts, num_experts_per_token, dtype)
        elif implementation in ROUTED_MOE_IMPLEMENTATIONS:
            self.router = _SoftmaxTopKRouter(hidden_size, num_nsp * local_experts, num_experts_per_token, dtype)

    def _routing(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        indices, weights = self.router(x)
        expert_ids = torch.arange(self.local_experts, device=x.device, dtype=indices.dtype).unsqueeze(
            0
        ) * self.num_nsp + torch.arange(self.num_nsp, device=x.device, dtype=indices.dtype).unsqueeze(1)
        matches = indices.unsqueeze(0).unsqueeze(0) == expert_ids.unsqueeze(-1).unsqueeze(-1)
        mask = matches.to(indices.dtype).sum(dim=-1) > 0
        routing_weight = (matches.to(weights.dtype) * weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1).unsqueeze(-1)
        return mask, routing_weight

    def _reduce(self, values: torch.Tensor) -> torch.Tensor:
        if self.experts_per_soc is not None:
            groups = self.num_nsp // self.experts_per_soc
            values = values.view(groups, self.experts_per_soc, *values.shape[1:]).sum(dim=1)
        return _tree_sum(values) if self.tree_reduce else values.sum(dim=0)

    def forward(self, x: torch.Tensor, local_t2e: torch.Tensor | None = None) -> torch.Tensor:
        routed = hasattr(self, "router")
        if routed:
            local_t2e, router_weights = self._routing(x)
            x = x.flatten(0, 1)
        else:
            router_weights = None
        assert local_t2e is not None
        if self.implementation == "reference":
            return self.reference(x, local_t2e)
        lane_output = torch.zeros(self.num_nsp, x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)
        for local_expert in range(self.local_experts):
            common = (
                x,
                local_t2e[:, local_expert],
                self.gate_weight[:, local_expert],
                self.up_weight[:, local_expert],
                self.down_weight[:, local_expert],
            )
            if self.implementation == "gather":
                delta = _gather_expert(
                    *common, self.ffn_blocking_mode, self.ffn_token_block_size, self.ffn_weight_block_size
                )
                lane_output = lane_output + delta
            elif self.implementation == "packed_chunk":
                delta = _packed_expert(
                    *common,
                    self.packed_chunk_size,
                    self.ffn_blocking_mode,
                    self.ffn_token_block_size,
                    self.ffn_weight_block_size,
                    post_gather=False,
                )
                lane_output = lane_output + delta
            elif self.implementation == "packed_chunk_post_gather":
                delta = _packed_expert(
                    *common,
                    self.packed_chunk_size,
                    self.ffn_blocking_mode,
                    self.ffn_token_block_size,
                    self.ffn_weight_block_size,
                    post_gather=True,
                )
                lane_output = lane_output + delta
            else:
                weight = router_weights[:, local_expert] if router_weights is not None else None
                lane_output = _cumsum_update_expert(
                    *common,
                    lane_output,
                    self.packed_chunk_size,
                    self.ffn_blocking_mode,
                    self.ffn_token_block_size,
                    self.ffn_weight_block_size,
                    weight,
                )
        return self._reduce(lane_output)

    def reference(self, x: torch.Tensor, local_t2e: torch.Tensor | None = None) -> torch.Tensor:
        if hasattr(self, "router"):
            local_t2e, router_weights = self._routing(x)
            x = x.flatten(0, 1)
        else:
            router_weights = None
        assert local_t2e is not None
        expanded = x.unsqueeze(0).expand(self.num_nsp, -1, -1)
        lanes = torch.zeros_like(expanded)
        for local_expert in range(self.local_experts):
            output = _expert_mlp(
                expanded,
                self.gate_weight[:, local_expert],
                self.up_weight[:, local_expert],
                self.down_weight[:, local_expert],
                self.ffn_blocking_mode,
                self.ffn_token_block_size,
                self.ffn_weight_block_size,
            )
            if router_weights is None:
                output = output * local_t2e[:, local_expert].unsqueeze(-1)
            else:
                output = output * router_weights[:, local_expert]
            lanes = lanes + output
        return self._reduce(lanes)


class GatherBmmMoE(nn.Module):
    """Decode kernel that gathers expert weights and evaluates selected experts with BMM."""

    def __init__(
        self, num_experts: int, hidden_size: int, intermediate_size: int, top_k: int, dtype: torch.dtype, loop: bool
    ):
        super().__init__()
        self.top_k = top_k
        self.loop = loop
        self.gate_weight = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size, dtype=dtype) * 0.02)
        self.up_weight = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size, dtype=dtype) * 0.02)
        self.down_weight = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size, dtype=dtype) * 0.02)
        self.gate_bias = nn.Parameter(torch.zeros(num_experts, intermediate_size, dtype=dtype))
        self.up_bias = nn.Parameter(torch.zeros(num_experts, intermediate_size, dtype=dtype))
        self.down_bias = nn.Parameter(torch.zeros(num_experts, hidden_size, dtype=dtype))
        self.register_buffer("gate_clamp_min", torch.tensor(torch.finfo(torch.float16).min, dtype=torch.float32))
        self.register_buffer("gate_clamp_max", torch.tensor(127.0, dtype=torch.float32))
        self.register_buffer("up_clamp_min", torch.tensor(-127.0, dtype=torch.float32))
        self.register_buffer("up_clamp_max", torch.tensor(127.0, dtype=torch.float32))
        self.register_buffer("alpha", torch.tensor(1.0, dtype=torch.float32))

    def _run(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        tokens, hidden = x.shape
        flat = indices.reshape(-1).long()
        expert_input = x.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, 1, hidden)
        gate = torch.bmm(expert_input, self.gate_weight[flat]) + self.gate_bias[flat].unsqueeze(1)
        up = torch.bmm(expert_input, self.up_weight[flat]) + self.up_bias[flat].unsqueeze(1)
        gate = torch.minimum(
            torch.maximum(gate, self.gate_clamp_min.to(gate.dtype)), self.gate_clamp_max.to(gate.dtype)
        )
        up = torch.minimum(torch.maximum(up, self.up_clamp_min.to(up.dtype)), self.up_clamp_max.to(up.dtype))
        output = torch.bmm((up + 1) * gate * torch.sigmoid(gate * self.alpha), self.down_weight[flat])
        output = output + self.down_bias[flat].unsqueeze(1)
        return (output.view(tokens, self.top_k, hidden) * weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, x: torch.Tensor, router_indices: torch.Tensor, router_weights: torch.Tensor) -> torch.Tensor:
        if not self.loop:
            return self._run(x, router_indices, router_weights)
        return torch.cat(
            [
                self._run(x[index : index + 1], router_indices[index : index + 1], router_weights[index : index + 1])
                for index in range(x.shape[0])
            ],
            dim=0,
        )

    def reference(self, x: torch.Tensor, router_indices: torch.Tensor, router_weights: torch.Tensor) -> torch.Tensor:
        return self._run(x, router_indices, router_weights)
