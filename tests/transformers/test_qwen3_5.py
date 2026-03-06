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
import pytest
import torch

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

transformers_qwen3_5 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
Qwen3_5ForCausalLM = transformers_qwen3_5.Qwen3_5ForCausalLM
Qwen3_5TextConfig = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5").Qwen3_5TextConfig


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    vocab_size = 64
    padding_side = "right"

    @staticmethod
    def _encode_prompt(prompt):
        return [int(tok) for tok in prompt.split()]

    def __call__(self, prompt, return_tensors="pt", padding=True):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        encoded = [self._encode_prompt(item) for item in prompts]
        max_len = max(len(item) for item in encoded) if padding else None

        input_ids = []
        attention_mask = []
        for item in encoded:
            pad_len = (max_len - len(item)) if max_len is not None else 0
            input_ids.append(item + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(item) + [0] * pad_len)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        if return_tensors == "np":
            return {
                "input_ids": np.asarray(input_ids, dtype=np.int64),
                "attention_mask": np.asarray(attention_mask, dtype=np.int64),
            }
        raise ValueError(f"Unsupported return_tensors={return_tensors}")

    def encode(self, prompt, return_tensors="pt"):
        encoded = self._encode_prompt(prompt)
        if return_tensors == "pt":
            return torch.tensor([encoded], dtype=torch.long)
        raise ValueError(f"Unsupported return_tensors={return_tensors}")

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        values = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        return " ".join(str(int(tok)) for tok in values if int(tok) != self.pad_token_id)

    def batch_decode(self, batch_token_ids, skip_special_tokens=True):
        del skip_special_tokens
        values = batch_token_ids.tolist() if hasattr(batch_token_ids, "tolist") else batch_token_ids
        return [self.decode(row) for row in values]


def _make_config(num_hidden_layers=2, layer_types=None):
    if layer_types is None:
        layer_types = ["linear_attention", "full_attention"]
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=layer_types,
        linear_conv_kernel_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        max_position_embeddings=128,
        use_cache=True,
    )


def test_qwen3_5_api_runner_pytorch_matches_ort(tmp_path):
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(_make_config()).eval()
    qeff_model = QEFFAutoModelForCausalLM(model)

    tokenizer = DummyTokenizer()
    api_runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=qeff_model.model.config,
        prompt=["1 2 3 4"],
        prompt_len=4,
        ctx_len=6,
    )

    pytorch_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
    onnx_path = qeff_model.export(tmp_path)
    ort_tokens = api_runner.run_kv_model_on_ort(onnx_path)

    assert np.array_equal(pytorch_tokens, ort_tokens)


def test_qwen3_5_api_runner_hf_matches_qeff_and_ort(tmp_path):
    torch.manual_seed(0)
    hf_model = Qwen3_5ForCausalLM(_make_config()).eval()
    qeff_source_model = Qwen3_5ForCausalLM(_make_config()).eval()
    qeff_source_model.load_state_dict(copy.deepcopy(hf_model.state_dict()))
    qeff_model = QEFFAutoModelForCausalLM(qeff_source_model)

    tokenizer = DummyTokenizer()
    api_runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=qeff_model.model.config,
        prompt=["1 2 3 4"],
        prompt_len=4,
        ctx_len=6,
    )

    hf_tokens = api_runner.run_hf_model_on_pytorch(hf_model)
    qeff_pytorch_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
    onnx_path = qeff_model.export(tmp_path)
    ort_tokens = api_runner.run_kv_model_on_ort(onnx_path)

    hf_tokens = np.asarray(hf_tokens).reshape(1, -1)
    qeff_pytorch_tokens = np.asarray(qeff_pytorch_tokens).reshape(1, -1)
    ort_tokens = np.asarray(ort_tokens).reshape(1, -1)

    assert np.array_equal(hf_tokens, qeff_pytorch_tokens)
    assert np.array_equal(qeff_pytorch_tokens, ort_tokens)


def test_qwen3_5_api_runner_hf_matches_qeff_and_ort_with_prefill_padding(tmp_path):
    torch.manual_seed(0)
    hf_model = Qwen3_5ForCausalLM(_make_config()).eval()
    qeff_source_model = Qwen3_5ForCausalLM(_make_config()).eval()
    qeff_source_model.load_state_dict(copy.deepcopy(hf_model.state_dict()))
    qeff_model = QEFFAutoModelForCausalLM(qeff_source_model)

    tokenizer = DummyTokenizer()
    api_runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=qeff_model.model.config,
        prompt=["1 2 3"],
        prompt_len=6,
        ctx_len=8,
    )

    hf_tokens = api_runner.run_hf_model_on_pytorch(hf_model)
    qeff_pytorch_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
    onnx_path = qeff_model.export(tmp_path)
    ort_tokens = api_runner.run_kv_model_on_ort(onnx_path)

    hf_tokens = np.asarray(hf_tokens).reshape(1, -1)
    qeff_pytorch_tokens = np.asarray(qeff_pytorch_tokens).reshape(1, -1)
    ort_tokens = np.asarray(ort_tokens).reshape(1, -1)

    assert np.array_equal(hf_tokens, qeff_pytorch_tokens)
    assert np.array_equal(qeff_pytorch_tokens, ort_tokens)


def test_qwen3_5_api_runner_hf_matches_qeff_and_ort_4layer_prompt8_ctx32(tmp_path):
    torch.manual_seed(0)
    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    hf_model = Qwen3_5ForCausalLM(_make_config(num_hidden_layers=4, layer_types=layer_types)).eval()
    qeff_source_model = Qwen3_5ForCausalLM(_make_config(num_hidden_layers=4, layer_types=layer_types)).eval()
    qeff_source_model.load_state_dict(copy.deepcopy(hf_model.state_dict()))
    qeff_model = QEFFAutoModelForCausalLM(qeff_source_model)

    tokenizer = DummyTokenizer()
    api_runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=qeff_model.model.config,
        prompt=["1 2 3 4 5 6 7 8"],
        prompt_len=8,
        ctx_len=32,
    )

    hf_tokens = api_runner.run_hf_model_on_pytorch(hf_model)
    qeff_pytorch_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
    onnx_path = qeff_model.export(tmp_path)
    ort_tokens = api_runner.run_kv_model_on_ort(onnx_path)

    hf_tokens = np.asarray(hf_tokens).reshape(1, -1)
    qeff_pytorch_tokens = np.asarray(qeff_pytorch_tokens).reshape(1, -1)
    ort_tokens = np.asarray(ort_tokens).reshape(1, -1)

    assert np.array_equal(hf_tokens, qeff_pytorch_tokens)
    assert np.array_equal(qeff_pytorch_tokens, ort_tokens)
