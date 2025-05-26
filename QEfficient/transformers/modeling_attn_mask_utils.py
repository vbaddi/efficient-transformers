# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Optional

import torch


def _create_causal_mask_sliding(
    position_ids: torch.Tensor,  # (bs, qlen)
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """
    Returns a boolean mask of shape **(bs, 1, qlen, width)**

    • dense mode  (`sliding_window=None`) reproduces a full causal mask.
    • sliding-window mode keeps only the previous `W` tokens visible.
    """
    # ------------------------------- dense causal ---------------------------
    if sliding_window is None:
        q = position_ids.unsqueeze(-1)  # (bs,qlen,1)
        k = torch.arange(position_ids.shape[-1], device=q.device)  # (width,)
        k = k.view(1, 1, -1)
        return (k > q).unsqueeze(1)  # masked=1

    # ------------------------- sliding-window causal ------------------------
    W = sliding_window
    q = position_ids  # (bs,qlen)
    window_lo = (q.unsqueeze(-1) - (W - 1)).clamp(min=0)  # (bs,qlen,1).
    kv_abs = window_lo + torch.arange(W)  # (bs,qlen,W)
    too_new = kv_abs > q.unsqueeze(-1)
    too_old = kv_abs < window_lo
    return (too_new | too_old).unsqueeze(1)


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
        # window_indices = query_indices - sliding_window
        window_mask = kv_indices < window_indices
        attention_mask = attention_mask | window_mask
        attention_mask = attention_mask.unsqueeze(1)
    else:
        query_indices = position_ids.unsqueeze(-1)
        kv_indices = torch.arange(target_length).view(1, 1, -1)
        attention_mask = kv_indices > query_indices
        attention_mask = attention_mask.unsqueeze(1)

    return attention_mask
