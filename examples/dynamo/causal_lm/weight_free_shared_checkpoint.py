# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Export decode and expert-parallel prefill using one prepared checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ["QEFF_WF_LAYOUT_TRANSFORMS"] = "1"

import numpy as np
import torch
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.exporter.weight_free import load_weight_free_ort_inputs, resolve_weight_spec_path
from QEfficient.transformers.moe.weights import _pack_expert_parallel_tensor


def export_graph(model, **kwargs) -> Path:
    """Run the export portion of ``compile()`` without invoking qaic-compile."""
    onnx_path = model.get_onnx_path(dynamo=True, use_onnx_subfunctions=True, offload_pt_weights=False, **kwargs)
    return resolve_weight_spec_path(Path(onnx_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="tiny-random/qwen3-moe")
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--ctx-len", type=int, default=128)
    parser.add_argument("--prefill-seq-len", type=int, default=32)
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--num-devices", type=int, default=2)
    parser.add_argument("--mdp-num-partitions", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("wf_shared_checkpoint"))
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    if args.num_hidden_layers > 0:
        config.num_hidden_layers = args.num_hidden_layers
    model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_name, config=config, trust_remote_code=True, weight_free=True
    )

    decode_spec = export_graph(
        model,
        prefill_only=False,
        retain_full_kv=True,
        specializations=[{"batch_size": 1, "seq_len": 1, "ctx_len": args.ctx_len}],
        aic_num_cores=args.num_cores,
    )
    prefill_spec = export_graph(
        model,
        prefill_only=True,
        enable_chunking=True,
        retain_full_kv=True,
        specializations=[{"batch_size": 1, "seq_len": args.prefill_seq_len, "ctx_len": args.ctx_len}],
        aic_num_cores=args.num_cores,
        num_devices=args.num_devices,
        mdp_num_partitions=args.mdp_num_partitions,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": 16}},
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(decode_spec, args.output_dir / "decode_weight_spec.json")
    shutil.copy(prefill_spec, args.output_dir / "prefill_weight_spec.json")

    decode = json.loads(decode_spec.read_text())
    prefill = json.loads(prefill_spec.read_text())
    transformed = [entry for entry in prefill["inputs"] if entry.get("transform")]
    print(f"decode  spec v{decode['version']}: {decode_spec}")
    print(f"prefill spec v{prefill['version']}: {prefill_spec}")
    print(f"shared prepared checkpoint: {decode['model_id'] == prefill['model_id']} -> {prefill['model_id']}")
    print(f"prefill inputs with a transform: {len(transformed)} of {len(prefill['inputs'])}")
    if decode["model_id"] != prefill["model_id"] or not transformed:
        raise SystemExit("FAILED: prefill and decode do not share one checkpoint with v6 transforms.")

    decode_weights = load_weight_free_ort_inputs(decode_spec, {})
    prefill_weights = load_weight_free_ort_inputs(prefill_spec, {})
    decode_name_by_key = {entry["location"]["key"]: entry["name"] for entry in decode["inputs"]}
    for entry in transformed:
        decode_name = decode_name_by_key.get(entry["location"]["key"])
        if decode_name is None:
            raise SystemExit(f"FAILED: decode spec does not read checkpoint key {entry['location']['key']}.")
        num_pipeline_stages, num_parallelized_experts = entry["transform"][0]["shape"][:2]
        expected = _pack_expert_parallel_tensor(
            torch.from_numpy(decode_weights[decode_name]),
            num_pipeline_stages=num_pipeline_stages,
            num_parallelized_experts=num_parallelized_experts,
        ).data.numpy()
        if not np.array_equal(prefill_weights[entry["name"]], expected):
            raise SystemExit(f"FAILED: {entry['name']} does not match expert-parallel packing.")
    print(f"OK: {len(transformed)} transformed inputs match the model's expert-parallel packing.")
    print(f"Specs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
