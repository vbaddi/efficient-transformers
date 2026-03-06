# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import numpy as np
import torch
from transformers import Qwen3_5ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

torch.manual_seed(42)


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    vocab_size = 64
    padding_side = "right"
    max_length = 32

    @staticmethod
    def _encode_prompt(prompt):
        return [int(tok) for tok in prompt.split()]

    def __call__(self, prompt, return_tensors="pt", padding=True, max_length=None):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        encoded = [self._encode_prompt(item) for item in prompts]

        if max_length is not None:
            # Use explicit max_length for padding target
            pad_target = max_length
        elif padding:
            pad_target = max(len(item) for item in encoded)
        else:
            pad_target = None

        input_ids = []
        attention_mask = []
        for item in encoded:
            # Truncate if longer than max_length
            if max_length is not None:
                item = item[:max_length]
            pad_len = (pad_target - len(item)) if pad_target is not None else 0
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


model_name = "Qwen/Qwen3.5-0.8B-Base"
tokenizer = DummyTokenizer()
layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
hf_model = Qwen3_5ForCausalLM(_make_config(num_hidden_layers=4, layer_types=layer_types)).eval()
# hf_model = Qwen3_5ForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

runner = ApiRunner(
    batch_size=1,
    tokenizer=tokenizer,
    config=hf_model.config,
    prompt=["1 2 4"],
    prompt_len=32,
    ctx_len=128,
)

# PyTorch HF Model

hf_tokens = runner.run_hf_model_on_pytorch(hf_model)

# PyTorch (KV) output

qeff_model = QEFFAutoModelForCausalLM(hf_model)

pt_tokens = runner.run_kv_model_on_pytorch(qeff_model.model)

hf_tokens = np.asarray(hf_tokens).reshape(1, -1)
pt_tokens = np.asarray(pt_tokens).reshape(1, -1)
assert np.array_equal(hf_tokens, pt_tokens)

onnx_path = qeff_model.export(verbose=True)
ort_tokens = runner.run_kv_model_on_ort(onnx_path)
# print(ort_tokens)

# qeff_model.compile(
#     prefill_seq_len=8,
#     ctx_len=32,
#     # convert_to_fp16=False,
#     use_onnx_subfunctions=False
# )
# print("compile done")
# print("QEff Transformed Onnx Model Outputs(AIC Backend)")
# output = qeff_model.generate(prompts=["1 2 3 4 5 6 7 8"], tokenizer=tokenizer, automation=True)
# print(output)
# # print(output.generated_ids)
