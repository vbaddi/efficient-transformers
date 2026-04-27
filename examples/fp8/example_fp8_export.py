# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
FP8 ONNX Export Example
"""

import argparse
import copy

import torch
from transformers import AutoModelForCausalLM

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils import hf_download

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a Causal LM (including FP8 models) to ONNX using QEfficient.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="neuralmagic/Llama-3.2-3B-Instruct-FP8",
        help="HuggingFace model card name or local path to the model.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help=(
            "Number of transformer layers to load. "
            "Reduces memory and export time during development/testing. "
            "Omit to load the full model."
        ),
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float16",
        choices=list(_DTYPE_MAP.keys()),
        help="Torch dtype to use when loading model weights.",
    )
    parser.add_argument(
        "--use-onnx-subfunctions",
        action="store_true",
        default=False,
        help=(
            "Export the model with ONNX sub-functions. "
            "This can significantly reduce export and compile time for large models."
        ),
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Directory where the exported ONNX graph will be saved. Defaults to QEfficient's cache directory.",
    )
    return parser.parse_args()


def load_model(model_name: str, num_layers: int | None, torch_dtype):
    torch.manual_seed(42)

    model_path = hf_download(
        repo_id=model_name,
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )

    load_kwargs = dict(
        use_cache=True,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
    )
    if num_layers is not None:
        load_kwargs["num_hidden_layers"] = num_layers

    model_hf = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model_hf.eval()
    return model_hf


def export_to_onnx(
    model_name: str,
    num_layers: int | None = None,
    torch_dtype=torch.float16,
    use_onnx_subfunctions: bool = False,
    export_dir: str | None = None,
) -> str:

    replace_transformers_quantizers()

    print(f"Loading model: {model_name}")
    if num_layers is not None:
        print(f"  num_hidden_layers overridden to: {num_layers}")
    print(f"  torch_dtype: {torch_dtype}")

    model_hf = load_model(model_name, num_layers, torch_dtype)

    qeff_model = QEFFAutoModelForCausalLM(
        copy.deepcopy(model_hf),
        pretrained_model_name_or_path=model_name,
    )

    print(f"\nExporting to ONNX (use_onnx_subfunctions={use_onnx_subfunctions}) ...")
    export_kwargs = {"use_onnx_subfunctions": use_onnx_subfunctions}
    if export_dir is not None:
        export_kwargs["export_dir"] = export_dir

    onnx_path = qeff_model.export(**export_kwargs)
    print(f"\nONNX export complete.\n  ONNX model saved to: {onnx_path}")
    return onnx_path


def main():
    args = parse_args()
    torch_dtype = _DTYPE_MAP[args.torch_dtype]

    export_to_onnx(
        model_name=args.model_name,
        num_layers=args.num_layers,
        torch_dtype=torch_dtype,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
        export_dir=args.export_dir,
    )


if __name__ == "__main__":
    main()
