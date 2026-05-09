# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient.exporter.weight_free import _default_weights_roots, load_weight_free_ort_inputs
from QEfficient.exporter.weight_spec import CheckpointFile, load_weight_spec, resolve_weight_spec_path, save_weight_spec
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner


def convert_checkpoint_to_fp32(onnx_path: Path, weight_spec_path: Path) -> None:
    """
    Load each safetensors checkpoint file, cast all tensors to FP32,
    save next to the ONNX, and update weight_spec.json to point there.

    This ensures the compiler sees matching dtypes between the ONNX (FLOAT)
    and the safetensors files (also FLOAT after conversion).
    """
    spec = load_weight_spec(weight_spec_path)
    export_dir = onnx_path.parent
    candidate_roots = _default_weights_roots(weight_spec_path, spec)

    new_checkpoint_files = []
    for idx, ckpt_file in enumerate(spec.checkpoint_files):
        rel_path = Path(ckpt_file.path)
        abs_path = rel_path if rel_path.is_absolute() else None
        if abs_path is None:
            for root in candidate_roots:
                candidate = root / rel_path
                if candidate.exists():
                    abs_path = candidate
                    break
        if abs_path is None or not abs_path.exists():
            raise FileNotFoundError(f"Cannot resolve checkpoint file: {ckpt_file.path}")

        tensors = load_file(str(abs_path))
        fp32_tensors = {k: v.to(torch.float32) for k, v in tensors.items()}

        out_name = f"model_{idx:04d}.safetensors" if len(spec.checkpoint_files) > 1 else "model.safetensors"
        save_file(fp32_tensors, str(export_dir / out_name))
        new_checkpoint_files.append(CheckpointFile(path=out_name, format="safetensors"))
        print(f"  {abs_path.name}  ({next(iter(tensors.values())).dtype})  →  {out_name}  (float32)")

    spec.checkpoint_files = new_checkpoint_files
    save_weight_spec(weight_spec_path, spec)

    # Sync aic_weightspec metadata in the ONNX so the compiler sees the same
    # (local FP32) paths as weight_spec.json.
    updated_json = json.dumps(json.loads(weight_spec_path.read_text()), separators=(",", ":"), sort_keys=True)
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    for entry in onnx_model.metadata_props:
        if entry.key == "aic_weightspec":
            entry.value = updated_json
            break
    tmp = onnx_path.with_suffix(onnx_path.suffix + ".tmp")
    onnx.save(onnx_model, str(tmp))
    tmp.replace(onnx_path)


# model_name = "meta-llama/Llama-3.3-70B-Instruct"
model_name = "meta-llama/Llama-3.2-1B"
# model_name = "gpt2"
# model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
# config.num_hidden_layers = 2
config.torch_dtype = torch.float32
print(config)

CONTINUOUS_BATCHING = False
FULL_BATCH_SIZE = 4  # slots in the KV cache; active batch_size stays at 1 here # NOT VERIFIED, WIP

runner = ApiRunner(
    batch_size=1,
    tokenizer=tokenizer,
    config=config,
    prompt=["My name is"],
    prompt_len=8,
    ctx_len=32,
    full_batch_size=FULL_BATCH_SIZE if CONTINUOUS_BATCHING else None,
)

with init_empty_weights():
    meta_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")

qeff_model = QEFFAutoModelForCausalLM(
    meta_model,
    pretrained_model_name_or_path=model_name,
    continuous_batching=CONTINUOUS_BATCHING,
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

print("Converting checkpoint to FP32 ...")
convert_checkpoint_to_fp32(onnx_path, weight_spec_path)

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
