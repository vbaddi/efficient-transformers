# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5TextModel,
    apply_rotary_pos_emb,
    l2norm,
    repeat_kv,
    rotate_half,
)

from QEfficient.transformers.cache_utils import QEffDynamicLayer
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


class QEffQwen3_5DynamicCache(Cache):
    """
    Hybrid cache for Qwen3.5 models.

    Full-attention layers retain KV cache, while linear-attention layers retain
    convolution and recurrent states.
    """

    def __init__(self, config):
        super().__init__(layers=[])
        self.config = config
        self.layer_types = list(config.layer_types)
        self.transformer_layers = [i for i, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"]
        self.last_linear_layer = next(
            (i for i in range(len(self.layer_types) - 1, -1, -1) if self.layer_types[i] == "linear_attention"),
            None,
        )
        self.kv_layers = [QEffDynamicLayer() if layer_type == "full_attention" else None for layer_type in self.layer_types]
        self.conv_states = [None for _ in self.layer_types]
        self.recurrent_states = [None for _ in self.layer_types]

    @classmethod
    def from_legacy_cache(
        cls,
        config,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor, ...], ...]] = None,
    ) -> "QEffQwen3_5DynamicCache":
        cache = cls(config)
        if past_key_values is None:
            return cache

        for layer_idx, layer_state in enumerate(past_key_values):
            if cache.layer_types[layer_idx] == "full_attention":
                key_states, value_states = layer_state
                layer = QEffDynamicLayer()
                layer.keys = key_states
                layer.values = value_states
                cache.kv_layers[layer_idx] = layer
            else:
                conv_state, recurrent_state = layer_state
                cache.conv_states[layer_idx] = conv_state
                cache.recurrent_states[layer_idx] = recurrent_state
        return cache

    def __len__(self):
        return len(self.layer_types)

    @property
    def key_cache(self):
        return [None if layer is None else layer.keys for layer in self.kv_layers]

    @property
    def value_cache(self):
        return [None if layer is None else layer.values for layer in self.kv_layers]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        layer = self.kv_layers[layer_idx]
        if layer is None:
            raise ValueError(f"Layer {layer_idx} is not a full_attention layer")
        return layer.update(key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx: Optional[int] = 0, cache_position: Optional[torch.LongTensor] = None) -> int:
        del cache_position
        if not self.transformer_layers:
            return 0
        if layer_idx not in self.transformer_layers:
            layer_idx = self.transformer_layers[0]
        layer = self.kv_layers[layer_idx]
        return 0 if layer is None or layer.keys is None else layer.keys.shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> Tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        return query_length + past_seen_tokens, kv_offset

    @property
    def has_previous_state(self) -> bool:
        if self.last_linear_layer is None:
            return False
        return self.conv_states[self.last_linear_layer] is not None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx, layer_type in enumerate(self.layer_types):
            if layer_type == "full_attention":
                layer = self.kv_layers[layer_idx]
                if layer is not None and layer.keys is not None:
                    device = layer.keys.device
                    beam_idx_device = beam_idx.to(device)
                    layer.keys = layer.keys.index_select(0, beam_idx_device)
                    layer.values = layer.values.index_select(0, beam_idx_device)
            elif self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx].device
                beam_idx_device = beam_idx.to(device)
                self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx_device)
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx_device)

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, ...], ...]:
        legacy_cache = ()
        for layer_idx, layer_type in enumerate(self.layer_types):
            if layer_type == "full_attention":
                layer = self.kv_layers[layer_idx]
                if layer is None or layer.keys is None:
                    legacy_cache += ((torch.empty(0), torch.empty(0)),)
                else:
                    legacy_cache += ((layer.keys, layer.values),)
            else:
                conv_state = self.conv_states[layer_idx]
                recurrent_state = self.recurrent_states[layer_idx]
                legacy_cache += (
                    (
                        torch.empty(0) if conv_state is None else conv_state,
                        torch.empty(0) if recurrent_state is None else recurrent_state,
                    ),
                )
        return legacy_cache


class QEffQwen3_5TextRotaryEmbedding(Qwen3_5TextRotaryEmbedding):
    """
    QEff wrapper for Qwen3.5 text RoPE.

    Similar to Qwen3, this precomputes a reusable base cache and then indexes it
    with the current 3D RoPE position ids before applying the Qwen3.5 MRoPE
    interleaving pattern.
    """

    def __init__(self, config, device=None):
        super().__init__(config=config, device=device)
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])
        self.register_buffer("mrope_source_index", self._build_mrope_source_index(), persistent=False)

    def _build_mrope_source_index(self) -> torch.Tensor:
        rotary_half_dim = self.inv_freq.shape[0]
        source_index = torch.zeros(rotary_half_dim, dtype=torch.long)
        for idx in range(rotary_half_dim):
            if idx % 3 == 1 and idx < self.mrope_section[1] * 3:
                source_index[idx] = 1
            elif idx % 3 == 2 and idx < self.mrope_section[2] * 3:
                source_index[idx] = 2
        return source_index

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached_half", freqs.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached_half", freqs.sin().to(dtype), persistent=False)

    def _apply_interleaved_mrope_cache(self, gathered_cache: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        safe_position_ids = torch.where(position_ids < 0, torch.zeros_like(position_ids), position_ids)
        cache = gathered_cache.unsqueeze(0).unsqueeze(1).expand(
            position_ids.shape[0], position_ids.shape[1], -1, gathered_cache.shape[-1]
        )
        gather_index = safe_position_ids.unsqueeze(-1).expand(-1, -1, -1, gathered_cache.shape[-1])
        stacked_cache = torch.gather(cache, 2, gather_index).permute(1, 2, 3, 0)
        source_index = self.mrope_source_index.to(device=stacked_cache.device)
        source_index = source_index.view(1, 1, -1, 1).expand(*stacked_cache.shape[:-1], 1)
        return torch.gather(stacked_cache, -1, source_index).squeeze(-1)

    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        cos_half = self._apply_interleaved_mrope_cache(self.cos_cached_half, position_ids)
        sin_half = self._apply_interleaved_mrope_cache(self.sin_cached_half, position_ids)

        cos = torch.cat((cos_half, cos_half), dim=-1) * self.attention_scaling
        sin = torch.cat((sin_half, sin_half), dim=-1) * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def qeff_apply_rotary_pos_emb(q, k, cos, sin, position_ids, mrope_section, unsqueeze_dim=1):
    del mrope_section
    cos = cos[position_ids]
    sin = sin[position_ids]
    cos = torch.cat([cos[0, ..., 0:32], cos[1, ..., 32:80], cos[2, ..., 80:128]], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([sin[0, ..., 0:32], sin[1, ..., 32:80], sin[2, ..., 80:128]], dim=-1).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    del kwargs
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def qeff_torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    updated_conv_state = hidden_states_new[:, :, -state_len:].to(hidden_states.dtype)
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:]).to(hidden_states.dtype)
    return out, updated_conv_state


class QEffQwen3_5Attention(Qwen3_5Attention):
    """
    Full-attention path with QEff cache updates for retained-state export.
    """

    qeff_export_mode = "unified"

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[QEffQwen3_5DynamicCache] = None,
        position_ids: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"batch_index": batch_index}
            if position_ids is not None:
                if torch.onnx.is_in_onnx_export() and self.qeff_export_mode != "decode":
                    cache_layer = past_key_values.kv_layers[self.layer_idx]
                    fallback_position = (
                        cache_layer.keys.shape[2] - 1
                        if cache_layer is not None and cache_layer.keys is not None
                        else position_ids.shape[-1] - 1
                    )
                    invalid_token_mask = (position_ids < 0).unsqueeze(1).unsqueeze(-1)
                    key_states = torch.where(invalid_token_mask, torch.zeros_like(key_states), key_states)
                    value_states = torch.where(invalid_token_mask, torch.zeros_like(value_states), value_states)
                    cache_kwargs["position_ids"] = torch.where(
                        position_ids < 0,
                        torch.tensor(fallback_position, dtype=position_ids.dtype, device=position_ids.device),
                        position_ids,
                    )
                elif torch.onnx.is_in_onnx_export():
                    cache_kwargs["position_ids"] = position_ids
                else:
                    scatter_position_ids = position_ids
                    if torch.any(position_ids < 0):
                        cache_layer = past_key_values.kv_layers[self.layer_idx]
                        if cache_layer is not None and cache_layer.keys is not None:
                            fallback_position = cache_layer.keys.shape[2] - 1
                        else:
                            fallback_position = int(position_ids.max().item())
                        scatter_position_ids = torch.where(
                            position_ids < 0,
                            torch.tensor(fallback_position, dtype=position_ids.dtype, device=position_ids.device),
                            position_ids,
                        )
                        invalid_token_mask = (position_ids < 0).unsqueeze(1).unsqueeze(-1)
                        key_states = torch.where(invalid_token_mask, torch.zeros_like(key_states), key_states)
                        value_states = torch.where(invalid_token_mask, torch.zeros_like(value_states), value_states)
                    cache_kwargs["position_ids"] = scatter_position_ids
            elif cache_position is not None:
                cache_kwargs["position_ids"] = cache_position.unsqueeze(0).expand(hidden_states.shape[0], -1)
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            elif attention_mask is not None:
                if torch.onnx.is_in_onnx_export() and position_ids is not None:
                    cache_kwargs["CCL"] = position_ids.max().to(torch.int64) + 1
                else:
                    cache_kwargs["CCL"] = attention_mask.shape[-1]
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        local_position_ids = position_ids
        if local_position_ids is None and cache_position is not None:
            local_position_ids = cache_position.unsqueeze(0).expand(hidden_states.shape[0], -1)

        if local_position_ids is not None:
            attention_mask = _create_causal_mask(
                position_ids=local_position_ids,
                target_length=key_states.shape[-2],
                sliding_window=None,
            )
        elif attention_mask is not None and attention_mask.shape[-1] != key_states.shape[-2]:
            if attention_mask.shape[-1] < key_states.shape[-2]:
                pad = torch.ones(
                    (*attention_mask.shape[:-1], key_states.shape[-2] - attention_mask.shape[-1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, pad], dim=-1)
            else:
                attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffQwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):
    """
    Linear-attention path with explicit conv/recurrent retained-state updates.
    """
    qeff_export_mode = "unified"

    def _build_prefill_conv_state(
        self,
        mixed_qkv_inputs: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, hidden_size, seq_len = mixed_qkv_inputs.shape
        valid_lengths = attention_mask.to(torch.int64).sum(dim=-1)
        window_positions = torch.arange(self.conv_kernel_size, device=mixed_qkv_inputs.device).view(1, -1)
        window_positions = window_positions + (valid_lengths.view(-1, 1) - self.conv_kernel_size)
        valid_window = window_positions >= 0
        gather_positions = window_positions.clamp(min=0, max=seq_len - 1)
        gather_positions = gather_positions.unsqueeze(1).expand(batch_size, hidden_size, self.conv_kernel_size)
        conv_state = mixed_qkv_inputs.gather(2, gather_positions)
        return torch.where(valid_window.unsqueeze(1), conv_state, torch.zeros_like(conv_state))

    def _recurrent_scan_gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unifies prefill and decode using recurrent state updates only.
        The loop length is fixed at trace time, while runtime sequence lengths are
        handled safely by clamped indexing + per-step validity masking.
        """
        initial_dtype = query.dtype
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
        ]
        query = query * (1 / (query.shape[-1] ** 0.5))

        batch_size, num_heads, traced_seq_len, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        seq_len_tensor = torch._shape_as_tensor(key)[2]

        if initial_state is None:
            recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=key.device, dtype=torch.float32)
        else:
            recurrent_state = initial_state.to(torch.float32)

        if token_mask is not None:
            token_mask = token_mask.to(dtype=torch.bool)

        core_attn_steps = []
        for i in range(traced_seq_len):
            i_tensor = torch.tensor(i, dtype=torch.int64, device=key.device)
            safe_idx = torch.clamp(i_tensor, min=0, max=seq_len_tensor - 1)
            valid_idx = safe_idx.eq(i_tensor)

            q_t = query.index_select(2, safe_idx.view(1)).squeeze(2)
            k_t = key.index_select(2, safe_idx.view(1)).squeeze(2)
            v_t = value.index_select(2, safe_idx.view(1)).squeeze(2)
            g_t = g.index_select(2, safe_idx.view(1)).squeeze(2).exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta.index_select(2, safe_idx.view(1)).squeeze(2).unsqueeze(-1)

            if token_mask is not None:
                mask_t = token_mask.index_select(1, safe_idx.view(1)).squeeze(1)
                valid = valid_idx & mask_t
            else:
                valid = valid_idx.expand(batch_size)

            decayed_state = recurrent_state * g_t
            kv_mem = (decayed_state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            candidate_state = decayed_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            candidate_out = (candidate_state * q_t.unsqueeze(-1)).sum(dim=-2)

            state_valid = valid.view(batch_size, 1, 1, 1).to(candidate_state.dtype)
            out_valid = valid.view(batch_size, 1, 1).to(candidate_out.dtype)
            recurrent_state = recurrent_state + state_valid * (candidate_state - recurrent_state)
            out_t = candidate_out * out_valid
            core_attn_steps.append(out_t.unsqueeze(2))

        core_attn_out = torch.cat(core_attn_steps, dim=2)
        core_attn_out = core_attn_out[:, :, : key.shape[2]]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, recurrent_state

    def _run_causal_conv_with_state(
        self,
        mixed_qkv: torch.Tensor,
        pre_conv_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        is_decode: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use the same conv flow for prefill and decode, only switching on runtime seq_len.
        Prefill keeps HF-compatible retained conv state, while decode uses the retained sliding window update.
        """
        decode_mixed_qkv, decode_conv_state = qeff_torch_causal_conv1d_update(
            mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias
        )
        prefill_mixed_qkv = F.silu(self.conv1d(pre_conv_qkv)[:, :, :seq_len])
        if attention_mask is not None:
            prefill_conv_state = self._build_prefill_conv_state(pre_conv_qkv, attention_mask)
        else:
            prefill_conv_state = pre_conv_qkv[:, :, -self.conv_kernel_size :]

        mixed_qkv = torch.where(is_decode, decode_mixed_qkv, prefill_mixed_qkv)
        new_conv_state = torch.where(is_decode, decode_conv_state, prefill_conv_state)
        return mixed_qkv, new_conv_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[QEffQwen3_5DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        del cache_position
        batch_size, seq_len, _ = hidden_states.shape
        seq_len_tensor = torch._shape_as_tensor(hidden_states)[1]
        is_decode = seq_len_tensor.eq(1)

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        pre_conv_qkv = mixed_qkv
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta = self.in_proj_b(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)

        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]
            if conv_state is None:
                conv_state = mixed_qkv.new_zeros((batch_size, self.conv_dim, self.conv_kernel_size))
            if recurrent_state is None:
                recurrent_state = hidden_states.new_zeros(
                    (batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim)
                )

            mixed_qkv, new_conv_state = self._run_causal_conv_with_state(
                mixed_qkv=mixed_qkv,
                pre_conv_qkv=pre_conv_qkv,
                conv_state=conv_state,
                attention_mask=attention_mask,
                seq_len=seq_len,
                is_decode=is_decode,
            )
            cache_params.conv_states[self.layer_idx] = new_conv_state
        else:
            recurrent_state = None
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if attention_mask is not None:
            valid_token_mask = attention_mask.to(dtype=torch.bool).unsqueeze(-1).unsqueeze(-1)
            scalar_valid_mask = attention_mask.to(dtype=torch.bool).unsqueeze(-1)
            query = torch.where(valid_token_mask, query, torch.zeros_like(query))
            key = torch.where(valid_token_mask, key, torch.zeros_like(key))
            value = torch.where(valid_token_mask, value, torch.zeros_like(value))
            z = torch.where(valid_token_mask, z, torch.zeros_like(z))
            beta = torch.where(scalar_valid_mask, beta, torch.zeros_like(beta))
            g = torch.where(scalar_valid_mask, g, torch.zeros_like(g))

        if cache_params is not None:
            core_attn_out, last_recurrent_state = self._recurrent_scan_gated_delta_rule(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                token_mask=attention_mask,
            )
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state
        else:
            core_attn_out, _ = self._recurrent_scan_gated_delta_rule(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                initial_state=None,
                token_mask=attention_mask,
            )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


class QEffQwen3_5DecoderLayer(Qwen3_5DecoderLayer):
    def __qeff_init__(self):
        if self.layer_type == "linear_attention":
            self.linear_attn.__class__ = QEffQwen3_5GatedDeltaNet
        elif self.layer_type == "full_attention":
            self.self_attn.__class__ = QEffQwen3_5Attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[QEffQwen3_5DynamicCache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        del use_cache
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QEffQwen3_5TextModel(Qwen3_5TextModel):
    def __qeff_init__(self):
        self.rotary_emb = QEffQwen3_5TextRotaryEmbedding(config=self.config)
        export_mode = getattr(self.config, "qeff_export_mode", "unified")
        for layer in self.layers:
            if getattr(layer, "layer_type", None) == "linear_attention" and hasattr(layer, "linear_attn"):
                layer.linear_attn.qeff_export_mode = export_mode
            elif getattr(layer, "layer_type", None) == "full_attention" and hasattr(layer, "self_attn"):
                layer.self_attn.qeff_export_mode = export_mode

    def _update_linear_attn_mask(self, attention_mask, cache_position):
        if torch.onnx.is_in_onnx_export():
            del cache_position
            return attention_mask
        return super()._update_linear_attn_mask(attention_mask, cache_position)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[QEffQwen3_5DynamicCache, Tuple[Tuple[torch.FloatTensor, ...], ...]]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_legacy_cache = False
        if past_key_values is not None and not isinstance(past_key_values, QEffQwen3_5DynamicCache):
            return_legacy_cache = True
            past_key_values = QEffQwen3_5DynamicCache.from_legacy_cache(self.config, past_key_values)
        elif use_cache and past_key_values is None:
            past_key_values = QEffQwen3_5DynamicCache(self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rotary_position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids
            rotary_position_ids = position_ids

        if attention_mask is None and text_position_ids is not None:
            attention_mask = (text_position_ids >= 0).to(dtype=inputs_embeds.dtype)

        if text_position_ids is not None:
            cache_position = torch.where(text_position_ids[0] >= 0, text_position_ids[0], 0)

        if torch.onnx.is_in_onnx_export():
            full_attention_layer = next(
                (idx for idx, layer_type in enumerate(self.config.layer_types) if layer_type == "full_attention"),
                None,
            )
            if full_attention_layer is not None and past_key_values is not None:
                cache_layer = past_key_values.kv_layers[full_attention_layer]
                target_length = cache_layer.keys.shape[2] if cache_layer is not None and cache_layer.keys is not None else attention_mask.shape[-1]
            else:
                target_length = (
                    attention_mask.shape[-1]
                    if isinstance(attention_mask, torch.Tensor)
                    else past_seen_tokens + inputs_embeds.shape[1]
                )
        else:
            target_length = (
                int(text_position_ids.max().item()) + 1
                if text_position_ids is not None
                else (
                    attention_mask.shape[-1]
                    if isinstance(attention_mask, torch.Tensor)
                    else past_seen_tokens + inputs_embeds.shape[1]
                )
            )
        causal_mask = _create_causal_mask(position_ids=text_position_ids, target_length=target_length, sliding_window=None)
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, rotary_position_ids)

        all_hidden_states = () if output_hidden_states else None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )


class QEffQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffQwen3_5DecoderLayer}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
        return past_key_values

    def _iter_retained_state_names(self) -> List[str]:
        names = []
        for layer_idx, layer_type in enumerate(self.config.layer_types):
            if layer_type == "full_attention":
                names.extend([f"past_key.{layer_idx}", f"past_value.{layer_idx}"])
            else:
                names.extend([f"conv_state.{layer_idx}", f"recurrent_state.{layer_idx}"])
        return names

    def get_retained_state_names(self) -> List[str]:
        return self._iter_retained_state_names()

    def get_onnx_retained_state_specs(
        self,
        batch_size: int,
        seq_len: int,
        kv_cache_shape: List[int],
        continuous_batching: bool = False,
        retain_full_kv: bool = False,
    ) -> dict:
        del seq_len, retain_full_kv
        batch_axis_name = "full_batch_size" if continuous_batching else "batch_size"
        specs = {
            "past_key_values": [],
            "input_names": [],
            "output_names": [],
            "dynamic_axes": {},
        }

        for layer_idx, layer_type in enumerate(self.config.layer_types):
            if layer_type == "full_attention":
                layer_names = [f"past_key.{layer_idx}", f"past_value.{layer_idx}"]
                layer_tensors = [
                    torch.zeros(tuple(kv_cache_shape), dtype=torch.float32),
                    torch.zeros(tuple(kv_cache_shape), dtype=torch.float32),
                ]
                layer_axes = [
                    {0: batch_axis_name, 2: "ctx_len"},
                    {0: batch_axis_name, 2: "ctx_len"},
                ]
            else:
                layer = self.model.layers[layer_idx].linear_attn
                conv_shape = (batch_size, layer.conv_dim, layer.conv_kernel_size)
                recurrent_shape = (batch_size, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim)
                layer_names = [f"conv_state.{layer_idx}", f"recurrent_state.{layer_idx}"]
                layer_tensors = [
                    torch.zeros(conv_shape, dtype=torch.float32),
                    torch.zeros(recurrent_shape, dtype=torch.float32),
                ]
                layer_axes = [{0: batch_axis_name}, {0: batch_axis_name}]

            specs["past_key_values"].append(layer_tensors)
            for name, axes in zip(layer_names, layer_axes):
                specs["input_names"].append(name)
                specs["output_names"].append(f"{name}_RetainedState")
                specs["dynamic_axes"][name] = axes

        return specs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[QEffQwen3_5DynamicCache, Tuple[Tuple[torch.FloatTensor, ...], ...]]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        del logits_to_keep
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        if position_ids is None:
            hidden_states = outputs.last_hidden_state[:, -1:, :]
        else:
            text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
            logit_index = text_position_ids.to(torch.int32).argmax(1, keepdim=True)
            hidden_states = outputs.last_hidden_state[torch.arange(text_position_ids.shape[0]).view(-1, 1), logit_index]

        logits = self.lm_head(hidden_states).float()
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
