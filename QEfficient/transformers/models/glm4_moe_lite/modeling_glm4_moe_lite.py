# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from typing import List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteModel,
    Glm4MoeLiteMoE,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
)

from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


def _qeff_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask,
            torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32),
            attn_weights,
        )
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class QEffGlm4MoeLiteMoE(Glm4MoeLiteMoE):
    def __qeff_init__(self):
        # Glm4MoeLite stores all experts as dense tensors already. Keep direct views
        # to run decode-only batched expert matmuls without per-expert loops.
        self.all_gate_up_proj = nn.Parameter(self.experts.gate_up_proj.detach().clone())
        self.all_down_proj = nn.Parameter(self.experts.down_proj.detach().clone())

    def _decode_moe(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        seq_len, hidden_size = hidden_states.shape
        expert_in = hidden_states.unsqueeze(1).expand(-1, self.top_k, -1).contiguous().view(-1, 1, hidden_size)

        gate_up_proj = self.all_gate_up_proj[topk_indices.flatten()]  # [T*K, 2I, H]
        down_proj = self.all_down_proj[topk_indices.flatten()]  # [T*K, H, I]

        gate_up = torch.bmm(expert_in, gate_up_proj.transpose(1, 2))
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = self.experts.act_fn(gate) * up
        expert_out = torch.bmm(hidden, down_proj.transpose(1, 2))  # [T*K, 1, H]

        experts_out = expert_out.view(seq_len, self.top_k, hidden_size)
        experts_out = experts_out * topk_weights.unsqueeze(-1)
        experts_out = torch.einsum("bkh->bh", experts_out)
        return experts_out.to(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self._decode_moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class QEffGlm4MoeLiteAttention(Glm4MoeLiteAttention):
    def __qeff_init__(self):
        if self.q_lora_rank is None:
            return

        q_b_proj = self.q_b_proj.weight.view(self.num_heads, self.qk_head_dim, self.q_lora_rank)
        q_nope = q_b_proj[:, : self.qk_nope_head_dim, :]
        q_rope = q_b_proj[:, self.qk_nope_head_dim :, :]

        kv_b_proj = self.kv_b_proj.weight.view(
            self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        k_up = kv_b_proj[:, : self.qk_nope_head_dim, :]
        v_up = kv_b_proj[:, self.qk_nope_head_dim :, :]

        self.q_nope_from_q_a = nn.Parameter(q_nope.transpose(1, 2).detach().clone())
        # q_a_out -> q_rope (per-head)
        self.q_rope_from_q_a = nn.Parameter(q_rope.transpose(1, 2).detach().clone())
        # q_a_out -> compressed-kv key space (per-head)
        self.fused_qk = nn.Parameter(torch.matmul(q_nope.transpose(1, 2), k_up).detach().clone())
        # compressed-kv -> value (per-head)
        self.v_up_per_head = nn.Parameter(v_up.detach().clone())

    def _decode_absorption(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Cache],
        cache_position: Optional[torch.LongTensor],
        batch_index: Optional[torch.LongTensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]

        q_a_out = self.q_a_layernorm(self.q_a_proj(hidden_states))
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kva = self.kv_a_layernorm(compressed_kv)

        # Absorption path: compute query no-pe directly from q_a output and pre-split q weights.
        q_nope = torch.einsum("bsr,hrd->bhsd", q_a_out, self.q_nope_from_q_a)
        q_rot = torch.einsum("bsr,hrd->bhsd", q_a_out, self.q_rope_from_q_a)

        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_pass = self.kv_b_proj(kva).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        if self.config.rope_interleave:
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
        query_states = torch.cat((q_nope, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        value_dim = value_states.shape[-1]
        if past_key_values is not None:
            value_states_cache = value_states
            key_states_cache = key_states
            layer = past_key_values.layers[self.layer_idx]
            cache_value_dim = layer.values.shape[-1] if getattr(layer, "values", None) is not None else value_dim
            cache_key_dim = layer.keys.shape[-1] if getattr(layer, "keys", None) is not None else key_states.shape[-1]
            if cache_value_dim > value_dim:
                value_states_cache = F.pad(value_states, [0, cache_value_dim - value_dim])
            if cache_key_dim > key_states.shape[-1]:
                key_states_cache = F.pad(key_states, [0, cache_key_dim - key_states.shape[-1]])
            cache_kwargs = {
                "position_ids": position_ids,
                "batch_index": batch_index,
                "cache_position": cache_position,
            }
            if attention_mask is not None:
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            key_states, value_states = past_key_values.update(
                key_states_cache,
                value_states_cache,
                self.layer_idx,
                cache_kwargs,
            )
            key_states = key_states[..., : query_states.shape[-1]]
            value_states = value_states[..., :value_dim]

        attn_output, attn_weights = _qeff_attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Cache | None = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        # MLA absorption path only for decode token-step.
        if (
            getattr(self, "q_nope_from_q_a", None) is not None
            and hidden_states.shape[1] == 1
            and kwargs.get("enable_mla", False)
            and kwargs.get("mla_absorption", {}).get("enable", True)
        ):
            attn_output, attn_weights = self._decode_absorption(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                batch_index=batch_index,
            )
            return attn_output, attn_weights

        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        if self.config.rope_interleave:
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        value_dim = value_states.shape[-1]
        if past_key_values is not None:
            value_states_cache = value_states
            key_states_cache = key_states
            layer = past_key_values.layers[self.layer_idx]
            cache_value_dim = layer.values.shape[-1] if getattr(layer, "values", None) is not None else value_dim
            cache_key_dim = layer.keys.shape[-1] if getattr(layer, "keys", None) is not None else key_states.shape[-1]
            if cache_value_dim > value_dim:
                value_states_cache = F.pad(value_states, [0, cache_value_dim - value_dim])
            if cache_key_dim > key_states.shape[-1]:
                key_states_cache = F.pad(key_states, [0, cache_key_dim - key_states.shape[-1]])
            cache_kwargs = {
                "position_ids": position_ids,
                "batch_index": batch_index,
                "cache_position": cache_position,
            }
            if attention_mask is not None:
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            key_states, value_states = past_key_values.update(
                key_states_cache,
                value_states_cache,
                self.layer_idx,
                cache_kwargs,
            )
            key_states = key_states[..., : query_states.shape[-1]]
            value_states = value_states[..., :value_dim]

        attn_output, attn_weights = _qeff_attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffGlm4MoeLiteDecoderLayer(Glm4MoeLiteDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            batch_index=batch_index,
            use_cache=use_cache,
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


class QEffGlm4MoeLiteModel(Glm4MoeLiteModel):
    def __qeff_init__(self):
        mla_env = os.environ.get("QEFF_ENABLE_GLM4_MLA_ABSORPTION", "0").lower()
        enable_mla = mla_env in {"1", "true", "yes", "on"}
        self.enable_mla = enable_mla
        self.mla_absorption_config = {"enable": enable_mla}

    def _resolve_target_length(
        self,
        past_key_values: Optional[Cache],
        legacy_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        position_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
    ) -> int:
        target_len = 0

        if legacy_cache is not None and len(legacy_cache) > 0:
            target_len = int(legacy_cache[0][0].shape[-2])
        elif past_key_values is not None:
            try:
                target_len = int(past_key_values[0][0].shape[-2])
            except Exception:
                target_len = int(getattr(past_key_values, "get_seq_length", lambda *a, **k: 0)())
                if target_len <= 0:
                    layers = getattr(past_key_values, "layers", [])
                    if layers and getattr(layers[0], "keys", None) is not None:
                        target_len = int(layers[0].keys.shape[-2])

        if target_len <= 0:
            target_len = int(position_ids.max().item()) + 1
        if target_len <= 0:
            target_len = int(inputs_embeds.shape[1])
        return target_len

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        legacy_cache = past_key_values if isinstance(past_key_values, (list, tuple)) else None
        if use_cache and not isinstance(past_key_values, Cache) and past_key_values is not None:
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)
        elif use_cache and past_key_values is None:
            past_key_values = QEffDynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        target_len = self._resolve_target_length(past_key_values, legacy_cache, position_ids, inputs_embeds)
        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=target_len)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                batch_index=batch_index,
                use_cache=use_cache,
                cache_position=cache_position,
                enable_mla=getattr(self, "enable_mla", False),
                mla_absorption=getattr(self, "mla_absorption_config", {"enable": False}),
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values.to_legacy_cache() if (use_cache and past_key_values is not None) else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
        )


class QEffGlm4MoeLiteForCausalLM(Glm4MoeLiteForCausalLM):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffGlm4MoeLiteDecoderLayer}

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state if return_dict else outputs[0]
        logit_idx = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_idx]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
