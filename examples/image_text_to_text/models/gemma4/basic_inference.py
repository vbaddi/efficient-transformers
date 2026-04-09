# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse

import numpy as np
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

    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def main():
    parser = argparse.ArgumentParser(description="Gemma4 VLM basic inference")
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
    parser.add_argument("--vision-use-onnx-subfunctions", action="store_true")
    parser.add_argument("--disable-lang-subfunctions", action="store_true")
    parser.add_argument("--disable-npi", action="store_true")
    parser.add_argument("--enable-vision-npi", action="store_true")
    parser.add_argument("--aic-enable-depth-first", action="store_true")
    args = parser.parse_args()

    image_url = None if args.skip_vision else args.image_url
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        kv_offload=True,
        skip_vision=args.skip_vision,
        dtype="float32",
    )

    model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        skip_vision=args.skip_vision,
        use_onnx_subfunctions=True,
        lang_use_onnx_subfunctions=not args.disable_lang_subfunctions,
        vision_use_onnx_subfunctions=args.vision_use_onnx_subfunctions,
        node_precision_info=not args.disable_npi,
        vision_node_precision_info=args.enable_vision_npi,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=args.aic_enable_depth_first,
    )

    inputs = build_inputs(processor, args.prompt, image_url)
    output = model.generate(inputs=inputs, processor=processor, generation_len=args.generation_len)
    generated_ids = normalize_generated_ids(output.generated_ids)[:, : args.generation_len]

    print("Generated IDs:")
    print(generated_ids.tolist())
    print("Generated Text:")
    print(processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    print(output)


if __name__ == "__main__":
    main()
