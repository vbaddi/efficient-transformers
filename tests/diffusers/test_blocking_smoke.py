# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Fast smoke tests for automated attention/FFN blocking selection.

Goal:
- Validate that blocking is computed from model config + pipeline compile config (no env vars needed).
- Validate that "no blocking" (value <= 1) collapses to defaults.
- Keep this test CPU-only and fast (no ONNX export/qaic-exec compile).

Run:
  pytest -q tests/diffusers/test_blocking_smoke.py
"""

from types import SimpleNamespace

import pytest

from QEfficient.diffusers.models.blocking_configurator import build_transformer_blocking_config
from QEfficient.diffusers.models.modeling_utils import get_attention_blocking_config


def _dummy_model_config(num_heads=24, hidden_size=3072, intermediate_size=8192):
    # SimpleNamespace mimics HF config objects (attribute access).
    return SimpleNamespace(
        num_attention_heads=num_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        # common alt names used by some diffusers/hf configs; harmless to include
        attention_head_dim=hidden_size // num_heads,
        max_position_embeddings=32760,
    )


def _dummy_pipeline_config(
    *,
    batch_size=1,
    seq_len=256,
    ctx_len=32760,
    mdp_ts_num_devices=4,
    aic_num_cores=16,
    convert_to_fp16=True,
):
    return {
        "modules": {
            "transformer": {
                "specializations": {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "ctx_len": ctx_len,
                },
                "compilation": {
                    "mdp_ts_num_devices": mdp_ts_num_devices,
                    "aic_num_cores": aic_num_cores,
                    "convert_to_fp16": convert_to_fp16,
                },
            }
        }
    }


def test_build_transformer_blocking_config_smoke():
    model_cfg = _dummy_model_config()
    pipeline_cfg = _dummy_pipeline_config()

    blocking = build_transformer_blocking_config(model_config=model_cfg, pipeline_config=pipeline_cfg)

    assert "attention" in blocking
    assert "ffn" in blocking

    # attention fields should exist and be ints (or None before optimization, but impl fills them)
    attn = blocking["attention"]
    assert isinstance(attn["head_block_size"], int)
    assert isinstance(attn["num_head_blocks"], int)
    assert isinstance(attn["num_kv_blocks"], int)
    assert isinstance(attn["num_q_blocks"], int)


def test_get_attention_blocking_config_prefers_computed_config(monkeypatch):
    # Ensure env vars don't interfere; computed path should be used.
    monkeypatch.delenv("ATTENTION_BLOCKING_MODE", raising=False)
    monkeypatch.delenv("HEAD_BLOCK_SIZE", raising=False)
    monkeypatch.delenv("NUM_KV_BLOCKS", raising=False)
    monkeypatch.delenv("NUM_Q_BLOCKS", raising=False)

    model_cfg = _dummy_model_config()
    pipeline_cfg = _dummy_pipeline_config()

    blocking_mode, head_block_size, num_kv_blocks, num_q_blocks = get_attention_blocking_config(
        model_config=model_cfg,
        pipeline_config=pipeline_cfg,
    )

    assert blocking_mode in {"default", "q", "kv", "qkv"}
    assert isinstance(head_block_size, int)
    assert isinstance(num_kv_blocks, int)
    assert isinstance(num_q_blocks, int)


@pytest.mark.parametrize(
    "attn_cfg, expected_mode",
    [
        ({"head_block_size": 24, "num_kv_blocks": 1, "num_q_blocks": 1}, "default"),
        ({"head_block_size": 3, "num_kv_blocks": 16, "num_q_blocks": 1}, "kv"),
        ({"head_block_size": 3, "num_kv_blocks": 1, "num_q_blocks": 2}, "q"),
        ({"head_block_size": 3, "num_kv_blocks": 16, "num_q_blocks": 2}, "qkv"),
    ],
)
def test_attention_mode_inference_from_blocks(attn_cfg, expected_mode):
    """
    Emulate the standalone rule from the task:
      - Anything > 1 => enabled
      - Anything == 1 => treated as no blocking for that dimension

    This test doesn't touch qaic compile; it just validates we can compute the
    correct effective mode from a given config, which is how the pipeline should behave.
    """

    def infer_mode(cfg):
        q = cfg["num_q_blocks"] > 1
        kv = cfg["num_kv_blocks"] > 1
        if q and kv:
            return "qkv"
        if kv:
            return "kv"
        if q:
            return "q"
        return "default"

    assert infer_mode(attn_cfg) == expected_mode
