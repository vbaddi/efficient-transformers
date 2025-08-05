# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers.cache_utils import EncoderDecoderCache, HybridCache, HybridChunkedCache
from transformers.configuration_utils import PretrainedConfig

from QEfficient.customop import (
    CtxGatherFunc,
    CtxGatherFuncCB,
    CtxScatterFunc,
    CtxScatterFuncCB,
)


class OriginalHybridCache:
    """
    Hybrid Cache class to be used with `torch.compile` for Gemma2 models that alternate between a local sliding window attention
    and global attention in every other layer. Under the hood, Hybrid Cache leverages ["SlidingWindowCache"] for sliding window attention
    and ["StaticCache"] for global attention. For more information, see the documentation of each subcomponeent cache class.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        max_cache_len (`int`, *optional*):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. If you're using more than 1 computation device, you
            should pass the `layer_device_map` argument instead.
        dtype (torch.dtype, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map (`Optional[Dict[int, Union[str, torch.device, int]]]]`, *optional*):
            Mapping between the layers and its device. This is required when you are manually initializing the cache
            and the model is split between different gpus. You can know which layers mapped to which device by
            checking the associated device_map: `model.hf_device_map`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, HybridCache

        >>> model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

        >>> inputs = tokenizer(text="My name is Gemma", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = HybridCache(config=model.config, max_batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        HybridCache()
        ```
    """

    # TODO (joao): dive deeper into gemma2 and paligemma -- there are reports of speed loss with compilation. Revert
    # ALL changes from the PR that commented the line below when reactivating it.
    # is_compileable = True

    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.float32,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__()
        if not hasattr(config, "sliding_window") or config.sliding_window is None:
            raise ValueError(
                "Setting `cache_implementation` to 'sliding_window' requires the model config supporting "
                "sliding window attention, please check if there is a `sliding_window` field in the model "
                "config and it's not set to None."
            )
        self.max_cache_len = max_cache_len
        self.max_batch_size = max_batch_size
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self._dtype = dtype
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )

        layer_switch = config.sliding_window_pattern if hasattr(config, "sliding_window_pattern") else 2  # 2 is for BC
        self.is_sliding = torch.tensor(
            [bool((i + 1) % layer_switch) for i in range(config.num_hidden_layers)], dtype=torch.bool
        )
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        global_cache_shape = (self.max_batch_size, self.num_key_value_heads, max_cache_len, self.head_dim)
        sliding_cache_shape = (
            self.max_batch_size,
            self.num_key_value_heads,
            min(config.sliding_window, max_cache_len),
            self.head_dim,
        )
        device = torch.device(device) if device is not None and isinstance(device, str) else None
        for i in range(config.num_hidden_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[i]
            else:
                layer_device = device
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            cache_shape = global_cache_shape if not self.is_sliding[i] else sliding_cache_shape
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self._dtype, device=layer_device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self._dtype, device=layer_device)
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def _sliding_update(self, cache_position, layer_idx, key_states, value_states, k_out, v_out, max_cache_len):
        if cache_position.shape[0] > max_cache_len:
            k_out = key_states[:, :, -max_cache_len:, :]
            v_out = value_states[:, :, -max_cache_len:, :]
            # Assumption: caches are all zeros at this point, `+=` is equivalent to `=` but compile-friendly
            self.key_cache[layer_idx] += k_out
            self.value_cache[layer_idx] += v_out
            # we should return the whole states instead of k_out, v_out to take the whole prompt
            # into consideration when building kv cache instead of just throwing away tokens outside of the window
            return key_states, value_states

        slicing = torch.ones(max_cache_len, dtype=torch.long, device=value_states.device).cumsum(0)
        cache_position = cache_position.clamp(0, max_cache_len - 1)
        to_shift = cache_position >= max_cache_len - 1
        indices = (slicing + to_shift[-1].int() - 1) % max_cache_len
        k_out = k_out[:, :, indices]
        v_out = v_out[:, :, indices]

        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states
        # `_.zero()` followed by `+=` is equivalent `=`, but compile-friendly (without graph breaks due to assignment)
        self.key_cache[layer_idx].zero_()
        self.value_cache[layer_idx].zero_()

        self.key_cache[layer_idx] += k_out
        self.value_cache[layer_idx] += v_out
        return k_out, v_out

    def _static_update(self, cache_position, layer_idx, key_states, value_states, k_out, v_out, max_cache_len):
        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states

        self.key_cache[layer_idx] = k_out
        self.value_cache[layer_idx] = v_out
        return k_out, v_out

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cache_kwargs is None:
            cache_kwargs = {}
        cache_position = cache_kwargs.get("cache_position")
        sliding_window = cache_kwargs.get("sliding_window")

        # These two `if` blocks are only reached in multigpu and if `layer_device_map` is not passed. They are used
        # when the cache is initialized in the forward pass (e.g. Gemma2)
        if self.key_cache[layer_idx].device != key_states.device:
            self.key_cache[layer_idx] = self.key_cache[layer_idx].to(key_states.device)
        if self.value_cache[layer_idx].device != value_states.device:
            self.value_cache[layer_idx] = self.value_cache[layer_idx].to(value_states.device)

        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        key_states = key_states.to(k_out.dtype)
        value_states = value_states.to(v_out.dtype)

        if sliding_window:
            update_fn = self._sliding_update
        else:
            update_fn = self._static_update

        return update_fn(
            cache_position,
            layer_idx,
            key_states,
            value_states,
            k_out,
            v_out,
            k_out.shape[2],
        )

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def get_seq_length(self, layer_idx: Optional[int] = 0):
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        if layer_idx != 0:
            raise ValueError(
                "`get_seq_length` on `HybridCache` may get inconsistent results depending on the layer index. "
                "Using the `layer_idx` argument is not supported."
            )
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()


# class QEffDynamicCache(DynamicCache):
#     """
#     A cache that grows dynamically as more tokens are generated. This is the default for generative models.

#     It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
#     `[batch_size, num_heads, seq_len, head_dim]`.

#     - Optimized implementation for the Cloud AI 100 to reuse KV Cache.
#     - get the position_ids input using kwargs.
#     - Use custom Onnxscript ops to write optimized version to generate Onnx model.

#     """

#     def write_only(self, key_states, value_states, layer_idx, cache_kwargs):
#         """
#         Write in the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

#         Parameters:
#             key_states (`torch.Tensor`):
#                 The new key states to cache.
#             value_states (`torch.Tensor`):
#                 The new value states to cache.
#             layer_idx (`int`):
#                 The index of the layer to cache the states for.
#             cache_kwargs (`Dict[str, Any]`, `optional`):
#                 Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.
#         """
#         # Update the cache
#         if len(self.key_cache) <= layer_idx:
#             self.key_cache.append(key_states)
#             self.value_cache.append(value_states)
#         else:
#             position_ids = cache_kwargs.get("position_ids")
#             batch_index = cache_kwargs.get("batch_index", None)

#             # Scatter
#             if batch_index is not None:
#                 invalid_scatter_index = torch.iinfo(torch.int32).max
#                 scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)

#                 self.key_cache[layer_idx] = CtxScatterFuncCB.apply(
#                     self.key_cache[layer_idx], batch_index, scatter_position_ids, key_states
#                 )
#                 self.value_cache[layer_idx] = CtxScatterFuncCB.apply(
#                     self.value_cache[layer_idx], batch_index, scatter_position_ids, value_states
#                 )
#             else:
#                 self.key_cache[layer_idx] = CtxScatterFunc.apply(self.key_cache[layer_idx], position_ids, key_states)
#                 self.value_cache[layer_idx] = CtxScatterFunc.apply(
#                     self.value_cache[layer_idx], position_ids, value_states
#                 )

#     def read_only(self, layer_idx, cache_kwargs):
#         """
#         Reads the `key_states` and `value_states` for the layer `layer_idx`.

#         Parameters:
#             layer_idx (`int`):
#                 The index of the layer to cache the states for.
#             cache_kwargs (`Dict[str, Any]`, `optional`):
#                 Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

#         Return:
#             A tuple containing the updated key and value states.
#         """
#         k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]
#         position_ids = cache_kwargs.get("position_ids")
#         batch_index = cache_kwargs.get("batch_index", None)
#         ctx_len = k_out.shape[2]
#         ctx_indices = torch.arange(ctx_len)[None, None, ...]
#         gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
#         invalid_mask = ctx_indices > gather_limit

#         if torch.onnx.is_in_onnx_export():
#             invalid_idx_value = torch.iinfo(torch.int32).max
#         else:
#             invalid_idx_value = 0

#         ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

#         if batch_index is not None:
#             k_out = CtxGatherFuncCB.apply(k_out, batch_index, ctx_indices)
#             v_out = CtxGatherFuncCB.apply(v_out, batch_index, ctx_indices)
#         else:
#             k_out = CtxGatherFunc.apply(k_out, ctx_indices)
#             v_out = CtxGatherFunc.apply(v_out, ctx_indices)

#         v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)
#         return k_out, v_out

#     def update(
#         self,
#         key_states: torch.Tensor,
#         value_states: torch.Tensor,
#         layer_idx: int,
#         cache_kwargs: Optional[Dict[str, Any]] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

#         Parameters:
#             key_states (`torch.Tensor`):
#                 The new key states to cache.
#             value_states (`torch.Tensor`):
#                 The new value states to cache.
#             layer_idx (`int`):
#                 The index of the layer to cache the states for.
#             cache_kwargs (`Dict[str, Any]`, `optional`):
#                 Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

#         Return:
#             A tuple containing the updated key and value states.
#         """
#         # Update the cache
#         if len(self.key_cache) <= layer_idx:
#             self.key_cache.append(key_states)
#             self.value_cache.append(value_states)
#             k_out, v_out = key_states, value_states
#         else:
#             position_ids = cache_kwargs.get("position_ids")
#             batch_index = cache_kwargs.get("batch_index", None)  # Check and fetch batch index value form the kwargs

#             # Scatter
#             if batch_index is not None:
#                 invalid_scatter_index = torch.iinfo(torch.int32).max
#                 scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)

#                 self.key_cache[layer_idx] = CtxScatterFuncCB.apply(
#                     self.key_cache[layer_idx], batch_index, scatter_position_ids, key_states
#                 )

#                 self.value_cache[layer_idx] = CtxScatterFuncCB.apply(
#                     self.value_cache[layer_idx], batch_index, scatter_position_ids, value_states
#                 )
#             else:
#                 self.key_cache[layer_idx] = CtxScatterFunc.apply(self.key_cache[layer_idx], position_ids, key_states)
#                 self.value_cache[layer_idx] = CtxScatterFunc.apply(
#                     self.value_cache[layer_idx], position_ids, value_states
#                 )

#             k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

#             # Gather
#             ctx_len = k_out.shape[2]
#             ctx_indices = torch.arange(ctx_len)[None, None, ...]
#             gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
#             invalid_mask = ctx_indices > gather_limit

#             if torch.onnx.is_in_onnx_export():
#                 invalid_idx_value = torch.iinfo(torch.int32).max
#             else:
#                 invalid_idx_value = 0

#             ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)
#             if batch_index is not None:
#                 k_out = CtxGatherFuncCB.apply(k_out, batch_index, ctx_indices)
#                 v_out = CtxGatherFuncCB.apply(v_out, batch_index, ctx_indices)
#             else:
#                 k_out = CtxGatherFunc.apply(k_out, ctx_indices)
#                 v_out = CtxGatherFunc.apply(v_out, ctx_indices)
#             v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)

#         return k_out, v_out

#     def update3D(
#         self,
#         key_states: torch.Tensor,
#         value_states: torch.Tensor,
#         layer_idx: int,
#         cache_kwargs: Optional[Dict[str, Any]] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

#         Parameters:
#             key_states (`torch.Tensor`):
#                 The new key states to cache.
#             value_states (`torch.Tensor`):
#                 The new value states to cache.
#             layer_idx (`int`):
#                 The index of the layer to cache the states for.
#             cache_kwargs (`Dict[str, Any]`, `optional`):
#                 Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

#         Return:
#             A tuple containing the updated key and value states.
#         """
#         # Update the cache
#         if len(self.key_cache) <= layer_idx:
#             self.key_cache.append(key_states)
#             self.value_cache.append(value_states)
#             k_out, v_out = key_states, value_states
#         else:
#             position_ids = cache_kwargs.get("position_ids")
#             batch_index = cache_kwargs.get("batch_index", None)

#             if batch_index is not None:
#                 invalid_scatter_index = torch.iinfo(torch.int32).max
#                 scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)

#                 self.key_cache[layer_idx] = CtxScatterFuncCB3D.apply(
#                     self.key_cache[layer_idx], batch_index, scatter_position_ids, key_states
#                 )

#                 self.value_cache[layer_idx] = CtxScatterFuncCB3D.apply(
#                     self.value_cache[layer_idx], batch_index, scatter_position_ids, value_states
#                 )

#             else:
#                 self.key_cache[layer_idx] = CtxScatterFunc3D.apply(self.key_cache[layer_idx], position_ids, key_states)
#                 self.value_cache[layer_idx] = CtxScatterFunc3D.apply(
#                     self.value_cache[layer_idx], position_ids, value_states
#                 )
#             k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

#             # Gather
#             ctx_len = k_out.shape[1]
#             ctx_indices = torch.arange(ctx_len)[None, ...]
#             gather_limit = position_ids.max(1, keepdim=True).values
#             invalid_mask = ctx_indices > gather_limit
#             if torch.onnx.is_in_onnx_export():
#                 invalid_idx_value = torch.iinfo(torch.int32).max
#             else:
#                 invalid_idx_value = 0
#             ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)
#             if batch_index is not None:
#                 k_out = CtxGatherFuncCB3D.apply(k_out, batch_index, ctx_indices)
#                 v_out = CtxGatherFuncCB3D.apply(v_out, batch_index, ctx_indices)
#             else:
#                 k_out = CtxGatherFunc3D.apply(k_out, ctx_indices)
#                 v_out = CtxGatherFunc3D.apply(v_out, ctx_indices)

#             v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)

#         return k_out, v_out


class QEffDynamicCache:
    def __init__(self) -> None:
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: tuple[tuple[torch.FloatTensor, torch.FloatTensor], ...]
    ) -> "QEffDynamicCache":
        """
        Converts a cache in the legacy cache format into an equivalent `Cache`. Used for
        backward compatibility.
        """
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                # Directly populate the cache lists
                cache.key_cache.append(key_states)
                cache.value_cache.append(value_states)
        return cache

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or not self.key_cache[layer_idx].numel()  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_cache_shape()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        """
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            k_out, v_out = key_states, value_states
        else:
            position_ids = cache_kwargs.get("position_ids")
            batch_index = cache_kwargs.get("batch_index", None)
            # Scatter
            if batch_index is not None:
                invalid_scatter_index = torch.iinfo(torch.int32).max
                scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)
                self.key_cache[layer_idx] = CtxScatterFuncCB.apply(
                    self.key_cache[layer_idx], batch_index, scatter_position_ids, key_states
                ).clone()
                self.value_cache[layer_idx] = CtxScatterFuncCB.apply(
                    self.value_cache[layer_idx], batch_index, scatter_position_ids, value_states
                ).clone()
            else:
                self.key_cache[layer_idx] = CtxScatterFunc.apply(
                    self.key_cache[layer_idx], position_ids, key_states
                ).clone()
                self.value_cache[layer_idx] = CtxScatterFunc.apply(
                    self.value_cache[layer_idx], position_ids, value_states
                ).clone()
            k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]
            # Gather
            ctx_len = k_out.shape[2]
            ctx_indices = torch.arange(ctx_len)[None, None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit
            if torch.onnx.is_in_onnx_export():
                invalid_idx_value = torch.iinfo(torch.int32).max
            else:
                invalid_idx_value = 0
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)
            if batch_index is not None:
                k_out = CtxGatherFuncCB.apply(k_out, batch_index, ctx_indices)
                v_out = CtxGatherFuncCB.apply(v_out, batch_index, ctx_indices)
            else:
                k_out = CtxGatherFunc.apply(k_out, ctx_indices)
                v_out = CtxGatherFunc.apply(v_out, ctx_indices)
            v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)
        return k_out, v_out


class QEffEncoderDecoderCache(EncoderDecoderCache):
    """
    Updated the `EncoderDecoderCache` to use the `QEffDynamicCache` for both self-attention and cross-attention caches.
    """

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "EncoderDecoderCache":
        """Converts a cache in the legacy cache format into an equivalent `EncoderDecoderCache`."""
        cache = cls(
            self_attention_cache=QEffDynamicCache(),
            cross_attention_cache=QEffDynamicCache(),
        )
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx][:2]
                cache.self_attention_cache.update(key_states, value_states, layer_idx)
                if len(past_key_values[layer_idx]) > 2:
                    key_states, value_states = past_key_values[layer_idx][2:]
                    cache.cross_attention_cache.update(key_states, value_states, layer_idx)
                    cache.is_updated[layer_idx] = True
        return cache


class QEffHybridCache(OriginalHybridCache):
    def __init__(self, config, batch_size, max_cache_len):
        super().__init__(config, batch_size, max_cache_len=max_cache_len)
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    @classmethod
    def from_legacy_cache(
        cls, config, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "HybridCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        cache = cls(config, batch_size=past_key_values[0][0].shape[0], max_cache_len=past_key_values[0][0].shape[2])
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_cache_shape()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            k_out, v_out = key_states, value_states
        else:
            position_ids = cache_kwargs.get("position_ids")
            config = cache_kwargs.get("config")
            is_sliding_layer = True if config.layer_types[layer_idx] == "sliding_attention" else False
            layer_ctx_len = self.key_cache[layer_idx].shape[2]
            kv_position_ids = torch.where(
                (~is_sliding_layer | (position_ids == -1)), position_ids, position_ids % (layer_ctx_len - 1)
            )

            kv_position_ids = torch.where(
                is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1) * 2),
                (position_ids + 1) % layer_ctx_len,
                kv_position_ids,
            )

            valid_mask = (kv_position_ids != -1).unsqueeze(1).unsqueeze(-1)
            key_states = torch.where(valid_mask == 1, key_states, torch.zeros_like(key_states))
            value_states = torch.where(valid_mask == 1, value_states, torch.zeros_like(value_states))
            self.key_cache[layer_idx] = CtxScatterFunc.apply(self.key_cache[layer_idx], kv_position_ids, key_states)
            self.value_cache[layer_idx] = CtxScatterFunc.apply(
                self.value_cache[layer_idx], kv_position_ids, value_states
            )
            k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

            # Original Gather
            ctx_len = self.key_cache[layer_idx].shape[2]
            ctx_indices = torch.arange(ctx_len)[None, None, ...]
            gather_limit = kv_position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit
            if torch.onnx.is_in_onnx_export():
                invalid_idx_value = torch.iinfo(torch.int32).max
            else:
                invalid_idx_value = 0
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

            all_indices = torch.arange(layer_ctx_len) + kv_position_ids.max() + 1
            rolling_indices = torch.where(all_indices > layer_ctx_len - 1, all_indices % layer_ctx_len, all_indices)
            final_indices = torch.where(
                (is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1))), rolling_indices, ctx_indices
            )
            k_out = CtxGatherFunc.apply(k_out, final_indices)
            v_out = CtxGatherFunc.apply(v_out, final_indices)
            ctx_v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)
            v_out = torch.where((is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1))), v_out, ctx_v_out)
        return k_out, v_out


class QEffHybridChunkedCache(HybridChunkedCache):
    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `HybridChunkedCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, config, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "HybridChunkedCache":
        """Converts a cache in the legacy cache format into an equivalent `HybridChunkedCache`. Used for
        backward compatibility."""
        cache = cls(config, max_batch_size=past_key_values[0][0].shape[0], max_cache_len=past_key_values[0][0].shape[2])
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            k_out, v_out = key_states, value_states

        else:
            position_ids = cache_kwargs.get("position_ids")
            is_sliding_layer = torch.tensor(bool(self.is_sliding[layer_idx]))

            # Update the position_ids to handle the sliding window
            layer_ctx_len = self.key_cache[layer_idx].shape[2]
            kv_position_ids = torch.where(
                (~is_sliding_layer | (position_ids == -1)), position_ids, position_ids % (layer_ctx_len - 1)
            )

            kv_position_ids = torch.where(
                is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1) * 2),
                (position_ids + 1) % layer_ctx_len,
                kv_position_ids,
            )

            valid_mask = (kv_position_ids != -1).unsqueeze(1).unsqueeze(-1)
            key_states = torch.where(valid_mask == 1, key_states, torch.zeros_like(key_states))
            value_states = torch.where(valid_mask == 1, value_states, torch.zeros_like(value_states))
            self.key_cache[layer_idx] = CtxScatterFunc.apply(self.key_cache[layer_idx], kv_position_ids, key_states)
            self.value_cache[layer_idx] = CtxScatterFunc.apply(
                self.value_cache[layer_idx], kv_position_ids, value_states
            )
            k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

            # Original Gather
            ctx_len = min(layer_ctx_len, k_out.shape[2])
            ctx_indices = torch.arange(ctx_len)[None, None, ...]
            gather_limit = kv_position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit
            if torch.onnx.is_in_onnx_export():
                invalid_idx_value = torch.iinfo(torch.int32).max
            else:
                invalid_idx_value = 0
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

            # Rolling indices for sliding window
            all_indices = torch.arange(layer_ctx_len) + kv_position_ids.max() + 1
            rolling_indices = torch.where(all_indices > layer_ctx_len - 1, all_indices % layer_ctx_len, all_indices)
            final_indices = torch.where(
                (is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1))), rolling_indices, ctx_indices
            )
            k_out = CtxGatherFunc.apply(k_out, final_indices)
            v_out = CtxGatherFunc.apply(v_out, final_indices)
            ctx_v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)
            v_out = torch.where((is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1))), v_out, ctx_v_out)
        return k_out, v_out
