# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from transformers import PreTrainedTokenizerFast

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

transformers_qwen3_5 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
Qwen3_5ForCausalLM = transformers_qwen3_5.Qwen3_5ForCausalLM
Qwen3_5ForConditionalGeneration = transformers_qwen3_5.Qwen3_5ForConditionalGeneration

MODEL_SNAPSHOT = Path(
    "/home/ubuntu/huggingface_hub/models--tiny-random--qwen3.5/snapshots/07ad2f323e908adc8d023136610ded7656a40ff0/"
)
PROMPT = "My name is "
PROMPT_LEN = 8
CTX_LEN = 16


def _require_snapshot() -> Path:
    if not MODEL_SNAPSHOT.exists():
        pytest.skip(f"Missing local model snapshot: {MODEL_SNAPSHOT}")
    return MODEL_SNAPSHOT


def _build_text_only_models():
    snapshot = _require_snapshot()
    full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        snapshot, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    hf_model = Qwen3_5ForCausalLM(full_model.config.text_config).eval()

    remapped_state_dict = {}
    for key, value in full_model.state_dict().items():
        if key.startswith("model.language_model."):
            remapped_state_dict[key.replace("model.language_model.", "model.")] = value
        elif key == "lm_head.weight":
            remapped_state_dict[key] = value

    missing, unexpected = hf_model.load_state_dict(remapped_state_dict, strict=False)
    assert not missing and not unexpected

    qeff_source_model = Qwen3_5ForCausalLM(full_model.config.text_config).eval()
    qeff_source_model.load_state_dict(copy.deepcopy(hf_model.state_dict()))
    qeff_model = QEFFAutoModelForCausalLM(qeff_source_model)
    return hf_model, qeff_model


def _build_tokenizer():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(_require_snapshot(), local_files_only=True)
    tokenizer.padding_side = "right"
    return tokenizer


def _build_api_runner(config):
    return ApiRunner(
        batch_size=1,
        tokenizer=_build_tokenizer(),
        config=config,
        prompt=[PROMPT],
        prompt_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
    )


def test_qwen3_5_real_model_hf_matches_qeff_and_exports_retained_states(tmp_path):
    hf_model, qeff_model = _build_text_only_models()
    api_runner = _build_api_runner(qeff_model.model.config)

    hf_tokens = np.asarray(api_runner.run_hf_model_on_pytorch(hf_model)).reshape(1, -1)
    qeff_tokens = np.asarray(api_runner.run_kv_model_on_pytorch(qeff_model.model)).reshape(1, -1)

    assert np.array_equal(hf_tokens, qeff_tokens)

    onnx_path = qeff_model.export(tmp_path)
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    input_names = [graph_input.name for graph_input in onnx_model.graph.input]
    output_names = [graph_output.name for graph_output in onnx_model.graph.output]

    assert any(name.startswith("conv_state.") for name in input_names)
    assert any(name.startswith("recurrent_state.") for name in input_names)
    assert any(name.startswith("past_key.") for name in input_names)
    assert any(name.startswith("past_value.") for name in input_names)
    assert any(name.startswith("conv_state.") and name.endswith("_RetainedState") for name in output_names)
    assert any(name.startswith("recurrent_state.") and name.endswith("_RetainedState") for name in output_names)
    assert any(name.startswith("past_key.") and name.endswith("_RetainedState") for name in output_names)
    assert any(name.startswith("past_value.") and name.endswith("_RetainedState") for name in output_names)


def test_qwen3_5_real_model_qeff_matches_ort(tmp_path):
    _, qeff_model = _build_text_only_models()
    api_runner = _build_api_runner(qeff_model.model.config)

    qeff_tokens = np.asarray(api_runner.run_kv_model_on_pytorch(qeff_model.model)).reshape(1, -1)
    onnx_path = qeff_model.export(tmp_path)
    ort_tokens = np.asarray(api_runner.run_kv_model_on_ort(onnx_path)).reshape(1, -1)

    assert np.array_equal(qeff_tokens, ort_tokens)
