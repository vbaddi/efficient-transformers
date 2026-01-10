# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from QEfficient.diffusers.models.blocking_configurator import build_transformer_blocking_config


class DummyTransformerConfig:
    num_attention_heads = 4
    attention_head_dim = 16
    hidden_size = 64
    intermediate_size = 128


def test_build_transformer_blocking_config_smoke():
    pipeline_config = {
        "modules": {
            "transformer": {
                "specializations": {"batch_size": 1, "seq_len": 16},
                "compilation": {"mdp_ts_num_devices": 1, "aic_num_cores": 1, "convert_to_fp16": True},
            }
        }
    }

    blocking_config = build_transformer_blocking_config(
        model_config=DummyTransformerConfig(),
        pipeline_config=pipeline_config,
        blocking_mode="hqkv",
    )

    assert blocking_config["blocking_mode"] == "qkv"
    attention_cfg = blocking_config["attention"]
    assert attention_cfg["head_block_size"] > 0
    assert attention_cfg["num_q_blocks"] is not None
    assert attention_cfg["num_kv_blocks"] is not None
    assert blocking_config["ffn"]["num_token_blocks"] > 0
