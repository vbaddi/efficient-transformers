# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Full Qwen3.5-35B-A3B weight-free blocked-QPC evaluation.

This entry point keeps the encoder, blocked prefill, and folded decode exports
separate so their export and compile timings can be reported independently.
Only the blocked prefill graph omits ``-convert-to-fp16``: its narrowed FP32
regions must survive compilation. The encoder and decode graphs use the
default conversion flag.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.transformers.models import modeling_auto

DEFAULT_MODEL = "/home/huggingface_hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754"
DEFAULT_OUT = "/home/vbaddi/.claude/jobs/7eb96c77/tmp/qwen35_full_weightfree_blocked"

CTX_LEN = 16 * 1024
PREFILL_SEQ_LEN = 64
GDN_CHUNK_SIZE = 64
IMAGE_HEIGHT = 354
IMAGE_WIDTH = 536

PREFILL_QAIC_CONFIG = {
    "blocking_mode": "prefill_online",
    "num_kv_blocks": 16,
    "num_q_blocks": 4,
    "n_rep_chunk": 1,
    "skip_kv": True,
    "gdn_chunk_size": GDN_CHUNK_SIZE,
    "gdn_full_state_update": True,
    "moe_config": {
        "flavour": "expert_parallel",
        "cores_per_expert": 1,
        "tree_reduce": False,
        "expert_parallel_chunk_size": 256,
    },
}

DECODE_QAIC_CONFIG = {
    "blocking_mode": "kv_batch_fold",
    "num_kv_blocks": 64,
    "gdn_num_head_blocks": 8,
    "skip_kv": True,
    "moe_config": {
        "flavour": "expert_parallel",
        "cores_per_expert": 1,
        "tree_reduce": False,
        "expert_parallel_chunk_size": 256,
    },
}


def _model_config(model_id: str, num_layers: int | None = None):
    config = AutoConfig.from_pretrained(model_id)
    config.dtype = "float16"
    config.torch_dtype = torch.float16
    if num_layers is not None:
        config.text_config.num_hidden_layers = num_layers
    return config


def _new_qeff(model_id: str, num_layers: int | None = None):
    return QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        config=_model_config(model_id, num_layers),
        attn_implementation="eager",
        dtype=torch.float16,
        kv_offload=True,
        continuous_batching=True,
        weight_free=True,
        layerwise=False,
    )


def _size_bytes(path: str | Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _qpc_stats(qpc_path: str | Path) -> dict:
    qpc_path = Path(qpc_path)
    program = qpc_path / "programqpc.bin"
    return {
        "qpc_path": str(qpc_path),
        "qpc_bytes": _size_bytes(qpc_path),
        "programqpc_bytes": program.stat().st_size if program.exists() else None,
    }


def _compile_one(
    name: str,
    model_id: str,
    out_dir: Path,
    num_layers: int | None = None,
    omit_convert_to_fp16: bool = False,
    **kwargs,
) -> dict:
    print(f"\n=== {name} ===", flush=True)
    print(f"compile kwargs: {kwargs}", flush=True)
    model = _new_qeff(model_id, num_layers)
    start = perf_counter()
    original_should_convert = modeling_auto._should_convert_to_fp16
    if omit_convert_to_fp16:
        modeling_auto._should_convert_to_fp16 = lambda target_dtype, compiler_options: False
        print(f"{name}: -convert-to-fp16 is NOT passed", flush=True)
    try:
        result = model.compile(compile_dir=str(out_dir), **kwargs)
    finally:
        modeling_auto._should_convert_to_fp16 = original_should_convert
    elapsed = perf_counter() - start
    result_dict = dict(result) if isinstance(result, dict) else {"qpc_path": result}
    result_dict["wall_time_seconds"] = round(elapsed, 3)
    for key, value in list(result_dict.items()):
        if value is not None and (key.endswith("qpc_path") or key.endswith("onnx_path")):
            result_dict[f"{key}_stats"] = _qpc_stats(value) if "qpc" in key else {"path": str(value)}
    print(json.dumps(result_dict, indent=2, default=str), flush=True)
    return result_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument("--with-convert-to-fp16", action="store_true")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--no-subfunctions", action="store_true")
    parser.add_argument("--prefill-seq-len", type=int, default=PREFILL_SEQ_LEN)
    parser.add_argument("--gdn-chunk-size", type=int, default=GDN_CHUNK_SIZE)
    parser.add_argument("--image-height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--image-width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--vision-qpc", default=None, help="reuse a previously compiled vision QPC")
    args = parser.parse_args()

    prefill_seq_len = args.prefill_seq_len
    gdn_chunk_size = args.gdn_chunk_size
    image_height = args.image_height
    image_width = args.image_width
    prefill_qaic_config = dict(PREFILL_QAIC_CONFIG)
    prefill_qaic_config["gdn_chunk_size"] = gdn_chunk_size

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    language_common = {
        "batch_size": 1,
        "full_batch_size": 1,
        "kv_cache_batch_size": 1,
        "ctx_len": CTX_LEN,
        "height": image_height,
        "width": image_width,
        "num_cores": 16,
        "use_onnx_subfunctions": not args.no_subfunctions,
        "retain_full_kv": True,
        "split_model_io": True,
        "user_tiled": True,
        "mxint8_kv_cache": True,
        "kv_cache_prefix": "vllmKvCache",
        "offload_pt_weights": False,
        "allow_mxint8_mdp_io": False,
    }
    encoder_common = {
        "batch_size": 1,
        "full_batch_size": 1,
        "prefill_seq_len": prefill_seq_len,
        "ctx_len": CTX_LEN,
        "height": image_height,
        "width": image_width,
        "num_cores": 16,
        "num_devices": 1,
        "mos": 1,
        "mxfp6_matmul": False,
        "mxint8_kv_cache": False,
        # Depth-first encoder compilation hits a QAIC Conv3D bias assertion at
        # the requested 2048x2048 image shape; keep the large-image path on
        # the non-DFS compiler route.
        "aic_enable_depth_first": False,
        "skip_vision": False,
        "skip_lang": True,
        "split_model_io": True,
        "use_onnx_subfunctions": not args.no_subfunctions,
    }

    report = {
        "model_id": args.model_id,
        "weight_free": True,
        "continuous_batching": True,
        "ctx_len": CTX_LEN,
        "prefill_seq_len": prefill_seq_len,
        "gdn_chunk_size": gdn_chunk_size,
        "image_height": image_height,
        "image_width": image_width,
        "prefill_omits_convert_to_fp16": not args.with_convert_to_fp16,
        "prefill_only_run": args.prefill_only,
        "modules": {},
    }

    if not args.decode_only:
        if not args.prefill_only:
            if args.vision_qpc:
                report["modules"]["encoder"] = {
                    "vision_qpc_path": args.vision_qpc,
                    "reused": True,
                }
                print(f"Reusing encoder QPC: {args.vision_qpc}", flush=True)
            else:
                report["modules"]["encoder"] = _compile_one(
                    "encoder",
                    args.model_id,
                    out_dir / "encoder",
                    num_layers=args.num_layers,
                    # QAIC's weight-free Conv3D path requires the encoder to
                    # remain FP32; -convert-to-fp16 triggers the biasC assertion.
                    omit_convert_to_fp16=True,
                    **encoder_common,
                    vision_batch_size=1,
                    prefill_only=None,
                )
        report["modules"]["prefill"] = _compile_one(
            "prefill",
            args.model_id,
            out_dir / "prefill",
            num_layers=args.num_layers,
            omit_convert_to_fp16=not args.with_convert_to_fp16,
            **language_common,
            prefill_seq_len=prefill_seq_len,
            num_devices=1,
            mxfp6_matmul=True,
            aic_enable_depth_first=False,
            prefill_only=True,
            enable_chunking=True,
            skip_vision=True,
            qaic_config=prefill_qaic_config,
        )
    if args.decode_only or not args.skip_decode:
        report["modules"]["decode"] = _compile_one(
            "decode",
            args.model_id,
            out_dir / "decode",
            num_layers=args.num_layers,
            **language_common,
            prefill_seq_len=1,
            num_devices=2,
            mxfp6_matmul=True,
            aic_enable_depth_first=False,
            prefill_only=False,
            enable_chunking=False,
            skip_vision=True,
            qaic_config=DECODE_QAIC_CONFIG,
        )

    report_path = out_dir / "compile_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport: {report_path}", flush=True)


if __name__ == "__main__":
    main()
