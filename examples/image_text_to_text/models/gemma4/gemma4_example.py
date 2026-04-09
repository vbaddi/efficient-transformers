# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse

import numpy as np
import torch
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText


def normalize_generated_ids(generated_ids):
    array = np.asarray(generated_ids)
    if array.dtype == object:
        array = np.asarray([np.asarray(row).reshape(-1) for row in generated_ids], dtype=np.int64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return array.astype(np.int64, copy=False)


def build_inputs(processor, prompt: str, image_url: str | None):
    if image_url is None:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_url},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
    return inputs


def run_model(
    model_name: str,
    prompt: str,
    image_url: str | None,
    skip_vision: bool,
    prefill_seq_len: int,
    ctx_len: int,
    generation_len: int,
    num_cores: int,
    num_devices: int,
    mos: int,
    aic_enable_depth_first: bool,
    enable_npi: bool,
):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        kv_offload=True,
        skip_vision=skip_vision,
        dtype="float32",
    )

    model.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        skip_vision=skip_vision,
        use_onnx_subfunctions=True,
        lang_use_onnx_subfunctions=True,
        vision_use_onnx_subfunctions=False,
        node_precision_info=enable_npi,
        vision_node_precision_info=False,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        mos=mos,
        aic_enable_depth_first=aic_enable_depth_first,
    )

    inputs = build_inputs(processor, prompt, None if skip_vision else image_url)
    output = model.generate(inputs=inputs, processor=processor, generation_len=generation_len)
    generated_ids = normalize_generated_ids(output.generated_ids)[:, :generation_len]

    print("Generated IDs:")
    print(generated_ids.tolist())
    print("Generated Text:")
    print(processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    print(output)


def main():
    parser = argparse.ArgumentParser(description="Gemma4 text-only or image+text QAic example")
    parser.add_argument("--model-name", type=str, default="tiny-random/gemma-4-dense")
    parser.add_argument("--prompt", type=str, default="What is shown in this image?")
    parser.add_argument(
        "--image-url",
        type=str,
        default="https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/Demos/sample-data/GoldenGate.png",
    )
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--prefill-seq-len", type=int, default=32)
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--generation-len", type=int, default=32)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--mos", type=int, default=1)
    parser.add_argument("--aic-enable-depth-first", action="store_true")
    parser.add_argument("--disable-npi", action="store_true")
    args = parser.parse_args()

    run_model(
        model_name=args.model_name,
        prompt=args.prompt,
        image_url=args.image_url,
        skip_vision=args.skip_vision,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        generation_len=args.generation_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        mos=args.mos,
        aic_enable_depth_first=args.aic_enable_depth_first,
        enable_npi=not args.disable_npi,
    )


if __name__ == "__main__":
    main()
