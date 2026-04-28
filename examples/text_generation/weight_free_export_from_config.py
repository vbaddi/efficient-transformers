# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient.exporter.weight_free import load_weight_free_ort_inputs
from QEfficient.exporter.weight_spec import resolve_weight_spec_path
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

# model_name = "meta-llama/Llama-3.2-1B"
# model_name = "gpt2"
model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
print(config)

runner = ApiRunner(
    batch_size=1,
    tokenizer=tokenizer,
    config=config,
    prompt=["My name is"],
    prompt_len=8,
    ctx_len=32,
)

with init_empty_weights():
    meta_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")

qeff_model = QEFFAutoModelForCausalLM(
    meta_model,
    pretrained_model_name_or_path=model_name,
)

export_dir = Path("test_models/weightfree_from_config")
for stale_export in export_dir.parent.glob(export_dir.name + "-*"):
    shutil.rmtree(stale_export, ignore_errors=True)

onnx_path = Path(
    qeff_model.export(
        export_dir=export_dir,
        use_dynamo=True,
        use_onnx_subfunctions=True,
        use_weight_free_export=True,
        offload_pt_weights=False,
    )
)
weight_spec_path = resolve_weight_spec_path(onnx_path)

session = ort.InferenceSession(str(onnx_path))
ort_inputs = load_weight_free_ort_inputs(weight_spec_path, runner.input_handler.prepare_ort_inputs())
ort_outputs = runner.run_ort_session(ort_inputs, session)
ort_outputs = runner.input_handler.update_ort_outputs(ort_outputs)

generated_ids = []
for _ in range(1, runner.gen_len):
    generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
    ort_inputs = runner.input_handler.update_ort_inputs(ort_inputs, ort_outputs)
    ort_inputs = load_weight_free_ort_inputs(weight_spec_path, ort_inputs)
    ort_outputs = runner.run_ort_session(ort_inputs, session)
    ort_outputs = runner.input_handler.update_ort_outputs(ort_outputs)

generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
generated_ids = np.concatenate(generated_ids, axis=1)
generated_text = runner.input_handler.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(f"Weight-free ONNX: {onnx_path}")
print(f"Weight spec: {weight_spec_path}")
print(generated_text)
