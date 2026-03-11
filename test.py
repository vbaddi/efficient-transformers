# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

model_name = "Qwen/Qwen3.5-0.8B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)

runner = ApiRunner(
    batch_size=1, tokenizer=tokenizer, config=hf_model.config, prompt=["My name is"], prompt_len=32, ctx_len=128
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
