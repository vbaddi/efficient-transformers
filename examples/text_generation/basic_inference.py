# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import os

import torch
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Basic text generation inference")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Input prompt")
    parser.add_argument("--prefill-seq-len", type=int, default=1, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=128, help="Context length")
    parser.add_argument("--generation-len", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument("--enable-mla", action="store_true", help="Enable GLM MLA absorption/compressed cache path")
    parser.add_argument("--use-onnx-subfunctions", action="store_true", help="Enable ONNX subfunctions during export")
    parser.add_argument(
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        default=None,
        help="Device IDs (comma-separated) e.g. [0,1]",
    )
    args = parser.parse_args()

    if args.enable_mla:
        os.environ["QEFF_ENABLE_GLM4_MLA_ABSORPTION"] = "1"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32)
    print(model.model.model.enable_mla)
    print(model.model.model.mla_absorption_config)

    # Compile the model
    qpc_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=(1 if args.device_group is None else len(args.device_group)),
        use_onnx_subfunctions=args.use_onnx_subfunctions,
    )
    print(f"Model compiled to: {qpc_path}")

    # Generate text
    exec_info = model.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt],
        device_id=args.device_group,
        generation_len=args.generation_len,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {exec_info.generated_texts[0]}")


if __name__ == "__main__":
    main()
