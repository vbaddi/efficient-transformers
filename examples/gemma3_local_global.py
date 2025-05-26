# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from transformers import AutoConfig, AutoModelForCausalLM, Gemma3ForCausalLM, AutoTokenizer
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.run_utils import ApiRunner

def add_named_scopes(model):
    for name, module in model.named_modules():
        if isinstance(module, Gemma3RMSNorm):
            module._onnx_scope_name = f"/{name}"

torch.manual_seed(42)
model_id = "google/gemma-3-4b-it"
model = Gemma3ForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, use_cache=True, attn_implementation="eager", sliding_window=6, num_hidden_layers=6)
model.eval()

tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_id)
config = model.config
batch_size = len(Constants.INPUT_STR)

api_runner = ApiRunner(
    batch_size,
    tokenizer,
    config, 
    ["Hi"],
    4,
    10
)
pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model)
qeff_model = QEFFAutoModelForCausalLM(model)
pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)

# add_named_scopes(qeff_model.model)
# onnx_model_path = qeff_model.export()
# ort_tokens = api_runner.run_kv_model_on_ort(onnx_model_path, is_tlm=False)
# assert (pytorch_kv_tokens == ort_tokens).all(), "Tokens don't match for ONNXRT output and PyTorch output."

# qpc_path = qeff_model.compile(
#         prefill_seq_len=128,
#         ctx_len=2048,
#         num_cores=16,
#         mxfp6_matmul=False,
#         mxint8_kv_cache=False,
#         num_devices=1,
#         mos=1,
#         aic_enable_depth_first=True,
#         num_speculative_tokens=None,
#         node_precision_info="fp32_nodes_rmsnorm.yaml"
# )
# print(f'qpc path is {qpc_path}')

# exec_info = qeff_model.generate(tokenizer, prompts="Once upon a time in Mumbai, ", device_ids=[0])