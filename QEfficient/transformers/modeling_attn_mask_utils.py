# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Optional

import torch


# --- static local‑window triangle cache ----------------------------------
def custom_tril(x):
    n, m = x.shape[-2], x.shape[-1]
    row_indices = torch.arange(n).view(-1, 1)
    col_indices = torch.arange(m).view(1, -1)
    mask = (col_indices <= row_indices - 1).bool()
    return x * mask


def custom_triu(x):
    n, m = x.shape[-2], x.shape[-1]
    row_indices = torch.arange(n, device=x.device).view(-1, 1)
    col_indices = torch.arange(m, device=x.device).view(1, -1)
    mask = (col_indices >= row_indices + 1).float()
    return x * mask


_local_tri_cache = {}


def _get_tri_mask(W: int):
    mask = _local_tri_cache.get(W)
    if mask is None:
        mask = custom_tril(torch.ones(1, 1, W, W))
        _local_tri_cache[W] = mask
    return mask


def _create_causal_mask_sliding_window(*, q_len: int, k_len: int, dtype, sliding_window=None):
    min_val = torch.finfo(dtype).min
    if sliding_window is None:
        full = torch.ones((1, 1, q_len, k_len), dtype=dtype)
        mask = custom_triu(full * min_val)
        return mask
    W = sliding_window
    k_eff = min(k_len, W)
    tri = _get_tri_mask(W)[:, :, :q_len, -k_eff:]
    return tri.to(dtype).mul(min_val)


def _create_causal_mask(
    position_ids,
    target_length,
    sliding_window: Optional[int] = None,
):
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with slided window
    """
    if sliding_window is not None:
        query_indices = position_ids.unsqueeze(-1)
        kv_indices = torch.arange(target_length).view(1, -1)
        # --- Rolling buffer ---
        pos_max = position_ids.max(1, keepdim=True).values
        kv_start = (pos_max // target_length) * target_length
        kv_indices_high = kv_indices + kv_start
        kv_indices_low = torch.where(kv_indices_high < target_length, kv_indices, kv_indices_high - target_length)
        kv_indices = torch.where(kv_indices_high > pos_max, kv_indices_low, kv_indices_high)
        kv_indices = kv_indices.unsqueeze(1)
        # ------
        causal_mask = kv_indices > query_indices
        attention_mask = causal_mask

        window_indices = query_indices - sliding_window + 1
        window_mask = kv_indices < window_indices
        attention_mask = attention_mask | window_mask
        attention_mask = attention_mask.unsqueeze(1)
    else:
        query_indices = position_ids.unsqueeze(-1)
        kv_indices = torch.arange(target_length).view(1, 1, -1)
        attention_mask = kv_indices > query_indices
        attention_mask = attention_mask.unsqueeze(1)

    return attention_mask
