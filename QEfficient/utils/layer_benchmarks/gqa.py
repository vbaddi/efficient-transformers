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
from torch import nn

from QEfficient.blocking.gqa_attention import (
    CHUNK_KV_GQA_IMPLEMENTATIONS,
    DECODE_GQA_IMPLEMENTATIONS,
    PREFILL_GQA_IMPLEMENTATIONS,
    UNROLLED_GQA_IMPLEMENTATIONS,
    gqa_attention,
)
from QEfficient.customop.ctx_chunk_scatter import ctx_chunk_scatter
from QEfficient.customop.utils import ctx_scatter, ctx_scatter_cb
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE
from QEfficient.utils.layer_benchmarks.contracts import BenchmarkCase, LayerBenchmark
from QEfficient.utils.layer_benchmarks.gqa_cache import pack_gqa_cache, unpack_gqa_cache

DECODE_IMPLEMENTATION = "decode_attn_headpar_batch_split"
PREFILL_IMPLEMENTATION = "prefill_attn_parallel"


@dataclass(frozen=True)
class GQAConfig:
    phase: str
    attention_blocking_mode: str
    implementation: str
    batch_size: int
    seq_len: int
    ctx_len: int
    max_position_embeddings: int
    start_position: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_kv_blocks: int
    query_block_size: int | None
    rope_theta: float
    rms_norm_epsilon: float
    qk_norm: bool
    skip_kv: bool
    kv_cache_dtype: str
    dtype: str
    continuous_batching: bool
    num_layers: int
    num_cores_per_device: int | None
    repeat_kv_heads: int
    q_head_block_chunk: int
    chunk_kv_size: int
    chunk_kv_n: int
    kv_block_unroll: int


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Qwen3GQABenchmarkLayer(nn.Module):
    """Minimal Qwen3 attention layer whose optimized kernel is shared with QEff."""

    def __init__(self, config: GQAConfig):
        super().__init__()
        self.config = config
        self.scaling = config.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)
        self.q_norm = nn.Parameter(torch.ones(config.head_dim))
        self.k_norm = nn.Parameter(torch.ones(config.head_dim))

        inverse_frequency = 1.0 / (
            config.rope_theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        positions = torch.arange(config.ctx_len, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos_cached", embedding.cos(), persistent=False)
        self.register_buffer("sin_cached", embedding.sin(), persistent=False)

    def _project(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch_size, query_len, self.config.num_attention_heads, self.config.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch_size, query_len, self.config.num_key_value_heads, self.config.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch_size, query_len, self.config.num_key_value_heads, self.config.head_dim
        )
        query_dtype = query.dtype
        key_dtype = key.dtype
        if self.config.qk_norm:
            query = query * torch.rsqrt(query.float().pow(2).mean(dim=-1, keepdim=True) + self.config.rms_norm_epsilon)
            key = key * torch.rsqrt(key.float().pow(2).mean(dim=-1, keepdim=True) + self.config.rms_norm_epsilon)
            query = self.q_norm * query.to(query_dtype)
            key = self.k_norm * key.to(key_dtype)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        cos = self.cos_cached[position_ids].unsqueeze(1).to(query.dtype)
        sin = self.sin_cached[position_ids].unsqueeze(1).to(query.dtype)
        return (query * cos) + (_rotate_half(query) * sin), (key * cos) + (_rotate_half(key) * sin), value

    def _updated_state(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        batch_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = self._project(hidden_states, position_ids)
        if self.config.implementation in CHUNK_KV_GQA_IMPLEMENTATIONS:
            split = self.config.num_cores_per_device or (
                int(self.config.num_attention_heads / self.config.num_key_value_heads)
            )
            packing = int(self.config.chunk_kv_size / split)
            return (
                query,
                ctx_chunk_scatter(past_key, position_ids.to(torch.int32), key, packing),
                ctx_chunk_scatter(past_value, position_ids.to(torch.int32), value, packing),
            )
        if batch_index is not None:
            batch_index = batch_index.to(torch.int32)
            scatter_position_ids = position_ids.to(torch.int32)
            return (
                query,
                ctx_scatter_cb(past_key, batch_index, scatter_position_ids, key),
                ctx_scatter_cb(past_value, batch_index, scatter_position_ids, value),
            )
        return query, ctx_scatter(past_key, position_ids, key), ctx_scatter(past_value, position_ids, value)

    def _output_projection(self, attention_output: torch.Tensor) -> torch.Tensor:
        batch_size, _, query_len, _ = attention_output.shape
        return self.o_proj(
            attention_output.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, query_len, self.config.num_attention_heads * self.config.head_dim)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        batch_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key_cache, value_cache = self._updated_state(
            hidden_states, position_ids, past_key, past_value, batch_index
        )
        attention_output = gqa_attention(
            query,
            key_cache,
            value_cache,
            position_ids,
            implementation=self.config.implementation,
            scaling=self.scaling,
            num_kv_blocks=self.config.num_kv_blocks,
            skip_kv=self.config.skip_kv,
            query_block_size=self.config.query_block_size if self.config.phase == "prefill" else None,
            batch_index=batch_index,
            num_cores_per_device=self.config.num_cores_per_device,
            chunk_kv_n=self.config.chunk_kv_n,
            chunk_kv_size=self.config.chunk_kv_size,
            query_head_block_size=self.config.q_head_block_chunk,
            kv_block_unroll=self.config.kv_block_unroll,
        )
        return self._output_projection(attention_output), key_cache, value_cache

    def reference(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        batch_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key_cache, value_cache = self._updated_state(
            hidden_states, position_ids, past_key, past_value, batch_index
        )
        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        if self.config.implementation in CHUNK_KV_GQA_IMPLEMENTATIONS:
            split = self.config.num_cores_per_device or groups
            ways = self.config.chunk_kv_n * split
            packing = int(self.config.chunk_kv_size / split)
            logical_key_cache = unpack_gqa_cache(
                key_cache,
                context_length=self.config.ctx_len,
                num_heads=self.config.num_key_value_heads,
                ways=ways,
                packing=packing,
            )
            logical_value_cache = unpack_gqa_cache(
                value_cache,
                context_length=self.config.ctx_len,
                num_heads=self.config.num_key_value_heads,
                ways=ways,
                packing=packing,
            )
        else:
            logical_key_cache = key_cache if batch_index is None else key_cache.index_select(0, batch_index.reshape(-1))
            logical_value_cache = (
                value_cache if batch_index is None else value_cache.index_select(0, batch_index.reshape(-1))
            )
        key = (
            logical_key_cache[:, :, None, :, :]
            .expand(-1, -1, groups, -1, -1)
            .reshape(self.config.batch_size, self.config.num_attention_heads, self.config.ctx_len, self.config.head_dim)
        )
        value = logical_value_cache[:, :, None, :, :].expand(-1, -1, groups, -1, -1).reshape_as(key)
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
        key_positions = torch.arange(self.config.ctx_len, device=query.device).view(1, 1, 1, -1)
        mask = key_positions > position_ids[:, None, :, None]
        scores = torch.where(
            mask,
            torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=scores.dtype, device=scores.device),
            scores,
        )
        probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        return self._output_projection(torch.matmul(probabilities, value)), key_cache, value_cache


class GQABenchmark(LayerBenchmark):
    name = "gqa"
    description = "Qwen3-style grouped-query attention with retained KV state."

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--phase", choices=("decode", "prefill"))
        parser.add_argument("--attn-blocking-mode", choices=("none", "kv"), default="kv")
        parser.add_argument("--implementation", "--impls", dest="implementation")
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument("--seq-len", type=int, default=None)
        parser.add_argument("--ctx-len", type=int, default=32768)
        parser.add_argument("--max-position-embeddings", type=int, default=262144)
        parser.add_argument("--start-position", "--start-pos-id", dest="start_position", type=int, default=0)
        parser.add_argument("--hidden-size", type=int, default=2048)
        parser.add_argument("--num-attention-heads", type=int, default=32)
        parser.add_argument("--num-key-value-heads", type=int, default=4)
        parser.add_argument("--head-dim", type=int, default=128)
        parser.add_argument("--num-kv-blocks", "--attn-num-kv-blocks", dest="num_kv_blocks", type=int, default=8)
        parser.add_argument("--query-block-size", "--q-block-size", dest="query_block_size", type=int, default=256)
        parser.add_argument("--rope-theta", type=float, default=10_000_000.0)
        parser.add_argument("--rms-norm-epsilon", "--rms-norm-eps", dest="rms_norm_epsilon", type=float, default=1e-6)
        parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--skip-kv", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--kv-cache-dtype", choices=("float16", "mxfp6", "mxint8"), default="float16")
        parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
        parser.add_argument("--continuous-batching", action="store_true")
        parser.add_argument("--num-layers", type=int, default=1)
        parser.add_argument("--num-cores-per-device", type=int)
        parser.add_argument("--repeat-kv-heads", type=int, default=1)
        parser.add_argument("--q-head-block-chunk", type=int, default=1)
        parser.add_argument("--chunk-kv-size", type=int, default=8192)
        parser.add_argument("--chunk-kv-n", type=int, default=2)
        parser.add_argument("--kv-block-unroll", type=int, default=2)

    def resolved_config(self, args: argparse.Namespace) -> Mapping[str, Any]:
        phase = args.phase or ("prefill" if args.seq_len is not None and args.seq_len > 1 else "decode")
        default_implementation = DECODE_IMPLEMENTATION if phase == "decode" else PREFILL_IMPLEMENTATION
        implementation = args.implementation or default_implementation
        allowed = DECODE_GQA_IMPLEMENTATIONS if phase == "decode" else PREFILL_GQA_IMPLEMENTATIONS
        if implementation not in allowed:
            raise ValueError(f"Phase {phase!r} requires implementation from {allowed}, got {implementation!r}.")
        seq_len = args.seq_len if args.seq_len is not None else (1 if phase == "decode" else 128)
        if phase == "decode" and seq_len != 1:
            raise ValueError("Decode GQA requires --seq-len 1.")
        if phase == "prefill" and seq_len <= 1:
            raise ValueError("Prefill GQA requires --seq-len greater than 1.")
        if args.attn_blocking_mode != "kv":
            raise ValueError("The GQA v1 benchmark requires --attn-blocking-mode kv.")
        if args.hidden_size <= 0 or args.head_dim <= 0 or args.num_kv_blocks <= 0:
            raise ValueError("GQA dimensions and --num-kv-blocks must be positive.")
        if args.num_attention_heads % args.num_key_value_heads:
            raise ValueError("--num-attention-heads must be divisible by --num-key-value-heads.")
        if args.start_position < 0 or args.start_position + seq_len > args.ctx_len:
            raise ValueError("The requested query positions must fit within --ctx-len.")
        if args.ctx_len > args.max_position_embeddings:
            raise ValueError("--ctx-len cannot exceed --max-position-embeddings.")
        if args.num_layers != 1:
            raise ValueError("The GQA v1 benchmark currently supports --num-layers 1 only.")
        if args.repeat_kv_heads != 1:
            raise ValueError("The GQA v1 benchmark currently supports --repeat-kv-heads 1 only.")
        if args.q_head_block_chunk <= 0:
            raise ValueError("--q-head-block-chunk must be positive.")
        chunk_kv_size = getattr(args, "chunk_kv_size", 8192)
        chunk_kv_n = getattr(args, "chunk_kv_n", 2)
        kv_block_unroll = getattr(args, "kv_block_unroll", 2)
        if chunk_kv_size <= 0 or chunk_kv_n <= 0 or kv_block_unroll <= 0:
            raise ValueError("GQA chunk and unroll values must be positive.")
        if implementation in CHUNK_KV_GQA_IMPLEMENTATIONS and args.continuous_batching:
            raise ValueError("Packed-KV implementations do not support --continuous-batching.")
        if args.continuous_batching and implementation not in {
            "decode_attn_headpar_batch_split",
            "decode_attn_headpar_batch_split_unroll",
        }:
            raise ValueError("--continuous-batching requires a decode batch-split implementation.")
        if implementation in UNROLLED_GQA_IMPLEMENTATIONS and kv_block_unroll < 2:
            raise ValueError("Unrolled GQA implementations require --kv-block-unroll >= 2.")
        if implementation in CHUNK_KV_GQA_IMPLEMENTATIONS:
            cores = args.num_cores_per_device or args.num_attention_heads // args.num_key_value_heads
            if chunk_kv_size % cores:
                raise ValueError("--chunk-kv-size must be divisible by --num-cores-per-device.")
            if implementation.startswith("prefill") and cores % args.num_key_value_heads:
                raise ValueError("Packed prefill requires cores per device divisible by KV heads.")
            groups = args.num_attention_heads // args.num_key_value_heads
            if implementation.startswith("prefill") and groups % args.q_head_block_chunk:
                raise ValueError("--q-head-block-chunk must divide the number of query groups.")
        config = GQAConfig(
            phase=phase,
            attention_blocking_mode=args.attn_blocking_mode,
            implementation=implementation,
            batch_size=args.batch_size,
            seq_len=seq_len,
            ctx_len=args.ctx_len,
            max_position_embeddings=args.max_position_embeddings,
            start_position=args.start_position,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            head_dim=args.head_dim,
            num_kv_blocks=args.num_kv_blocks,
            query_block_size=args.query_block_size if phase == "prefill" else None,
            rope_theta=args.rope_theta,
            rms_norm_epsilon=args.rms_norm_epsilon,
            qk_norm=args.qk_norm,
            skip_kv=args.skip_kv,
            kv_cache_dtype=args.kv_cache_dtype,
            dtype=args.dtype,
            continuous_batching=args.continuous_batching,
            num_layers=args.num_layers,
            num_cores_per_device=args.num_cores_per_device,
            repeat_kv_heads=args.repeat_kv_heads,
            q_head_block_chunk=args.q_head_block_chunk,
            chunk_kv_size=chunk_kv_size,
            chunk_kv_n=chunk_kv_n,
            kv_block_unroll=kv_block_unroll,
        )
        return asdict(config)

    def build_case(self, config: Mapping[str, Any], seed: int) -> BenchmarkCase:
        resolved = GQAConfig(**config)
        generator = torch.Generator().manual_seed(seed)
        module = Qwen3GQABenchmarkLayer(resolved).eval()
        hidden_states = torch.randn(resolved.batch_size, resolved.seq_len, resolved.hidden_size, generator=generator)
        position_ids = torch.arange(
            resolved.start_position, resolved.start_position + resolved.seq_len, dtype=torch.int64
        ).expand(resolved.batch_size, -1)
        cache_shape = (
            resolved.batch_size,
            resolved.num_key_value_heads,
            resolved.ctx_len,
            resolved.head_dim,
        )
        past_key = torch.zeros(cache_shape)
        past_value = torch.zeros(cache_shape)
        if resolved.start_position:
            past_key[:, :, : resolved.start_position] = torch.randn(
                resolved.batch_size,
                resolved.num_key_value_heads,
                resolved.start_position,
                resolved.head_dim,
                generator=generator,
            )
            past_value[:, :, : resolved.start_position] = torch.randn(
                resolved.batch_size,
                resolved.num_key_value_heads,
                resolved.start_position,
                resolved.head_dim,
                generator=generator,
            )
        if resolved.implementation in CHUNK_KV_GQA_IMPLEMENTATIONS:
            split = resolved.num_cores_per_device or (resolved.num_attention_heads // resolved.num_key_value_heads)
            ways = resolved.chunk_kv_n * split
            packing = resolved.chunk_kv_size // split
            past_key = pack_gqa_cache(past_key, ways=ways, packing=packing)
            past_value = pack_gqa_cache(past_value, ways=ways, packing=packing)
        inputs = OrderedDict(
            hidden_states=hidden_states,
            position_ids=position_ids,
            **{"past_key.0": past_key, "past_value.0": past_value},
        )
        if resolved.continuous_batching:
            batch_index = torch.randperm(resolved.batch_size, generator=generator, dtype=torch.int64).view(-1, 1)
            slots = batch_index.reshape(-1)
            inputs["past_key.0"] = torch.empty_like(past_key).index_copy(0, slots, past_key)
            inputs["past_value.0"] = torch.empty_like(past_value).index_copy(0, slots, past_value)
            inputs["batch_index"] = batch_index
        retained_names = {
            "past_key.0",
            "past_value.0",
            "past_key.0_RetainedState",
            "past_value.0_RetainedState",
        }
        custom_io = {name: resolved.kv_cache_dtype for name in retained_names}
        return BenchmarkCase(
            module=module,
            reference=module.reference,
            inputs=inputs,
            output_names=("attn_out", "past_key.0_RetainedState", "past_value.0_RetainedState"),
            dynamic_axes={
                "hidden_states": {0: "batch_size", 1: "seq_len"},
                "position_ids": {0: "batch_size", 1: "seq_len"},
                "attn_out": {0: "batch_size", 1: "seq_len"},
                **({"batch_index": {0: "batch_size"}} if resolved.continuous_batching else {}),
            },
            specializations={
                "batch_size": resolved.batch_size,
                "seq_len": resolved.seq_len,
                "ctx_len": resolved.ctx_len,
            },
            custom_io=custom_io,
            retained_names=retained_names,
        )
