# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from transformers import MistralForCausalLM

import QEfficient
from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.run_utils import ApiRunner

model_id = "mistralai/Mistral-7B-v0.1"
model = MistralForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, use_cache=True, attn_implementation="eager"
)
# model = MistralForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, use_cache=True, attn_implementation="eager")
model.eval()
model.to(torch.float32)

tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_id)
config = model.config
batch_size = len(Constants.INPUT_STR)
api_runner = ApiRunner(
    batch_size,
    tokenizer,
    config,
    ["[INST] What is your favourite condiment? [/INST]"],
    16,
    4096,
)
pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model)
qeff_model = QEFFAutoModelForCausalLM(model)
# pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
onnx_model_path = qeff_model.export()
# ort_tokens = api_runner.run_kv_model_on_ort(onnx_model_path, is_tlm=False)

qpc_path = qeff_model.compile(
    prefill_seq_len=128,
    ctx_len=4096,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=1,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
)
print(f"qpc path is {qpc_path}")
# exec_info = QEfficient.cloud_ai_100_exec_kv(
#     tokenizer=tokenizer, qpc_path=qpc_path, prompt="Who are you?", generation_len=8196, device_id=[0]
# )

exec_info = QEfficient.cloud_ai_100_exec_kv(tokenizer=tokenizer, qpc_path=qpc_path, prompt="[INST] What is your favourite condiment? [/INST]", generation_len=8192, device_id=[0])
