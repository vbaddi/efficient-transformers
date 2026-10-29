# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Run a weight-free Qwen3.5 vision/prefill/decode QPC end to end.

The QPCs are compiled separately for the vision encoder, blocked prefill, and
decode paths.  This script loads the processor inputs, runs the three QPCs,
prints stage timings, and decodes the generated token IDs.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import requests
import torch
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

DEFAULT_MODEL = "/home/huggingface_hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754"
DEFAULT_IMAGE = "https://picsum.photos/id/237/536/354"
DEFAULT_PROMPT = "Describe all the colors seen in the image."
DEFAULT_PREFILL_SEQ_LEN = 512

DEFAULT_VISION_QPC = (
    "/home/vbaddi/.claude/jobs/7eb96c77/tmp/qwen35_full_weightfree_final/encoder/qpc-e6766aa92ce8606e/qpc"
)
DEFAULT_PREFILL_QPC = (
    "/home/vbaddi/.claude/jobs/qwen35_weightfree_e2e_512_354x536_reuse_encoder/prefill/qpc-00dd820380621b45/qpc"
)
DEFAULT_DECODE_QPC = (
    "/home/vbaddi/.claude/jobs/7eb96c77/tmp/qwen35_full_weightfree_final/decode/qpc-4526ddbc60a8b366/qpc"
)


def _update_retained(target, outputs):
    suffix = "_RetainedState"
    for name, value in outputs.items():
        if name.endswith(suffix):
            target[name[: -len(suffix)]] = np.array(value, copy=True)


def _initialize_decode_states(session):
    states = {}
    prefixes = ("past_key.", "past_value.", "conv_state.", "recurrent_state.")
    for name in session.input_names:
        base = name.rsplit("/", 1)[-1]
        if not base.startswith(prefixes):
            continue
        binding = session.bindings[session.binding_index_map[name]]
        dtype = session.aic_to_np_dtype_mapping[binding.type]
        states[name] = np.zeros(tuple(binding.dims), dtype=dtype)
    return states


def _next_token(logits):
    logits = np.asarray(logits)
    if logits.ndim == 2:
        logits = logits[:, None, :]
    return logits[:, -1:, :].argmax(-1).astype(np.int64)


def _load_image(source):
    if source.startswith(("http://", "https://")):
        return Image.open(requests.get(source, stream=True, timeout=30).raw).convert("RGB")
    return Image.open(source).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--generation-len", type=int, default=32)
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_PREFILL_SEQ_LEN)
    parser.add_argument("--vision-qpc", default=DEFAULT_VISION_QPC)
    parser.add_argument("--prefill-qpc", default=DEFAULT_PREFILL_QPC)
    parser.add_argument("--decode-qpc", default=DEFAULT_DECODE_QPC)
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--resize-height", type=int, default=None)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_id)
    config.torch_dtype = "float32"
    layer_types = list(getattr(config.text_config, "layer_types", []))
    config.text_config.layer_types = layer_types[: config.text_config.num_hidden_layers]
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        weight_free=True,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id)

    vision_session = QAICInferenceSession(args.vision_qpc)
    prefill_session = QAICInferenceSession(args.prefill_qpc)
    decode_session = QAICInferenceSession(args.decode_qpc)

    image = _load_image(args.image)
    source_size = image.size
    if args.resize_width is not None or args.resize_height is not None:
        if args.resize_width is None or args.resize_height is None:
            parser.error("--resize-width and --resize-height must be provided together")
        image = image.resize((args.resize_width, args.resize_height), Image.Resampling.BICUBIC)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info([messages])
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=inputs, prefill_seq_len=args.prefill_seq_len, batch_size=1
    )

    input_ids_length = inputs["input_ids"].shape[1]
    padded_len = ((input_ids_length + args.prefill_seq_len - 1) // args.prefill_seq_len) * args.prefill_seq_len
    inputs["input_ids"] = torch.nn.functional.pad(
        inputs["input_ids"], (0, padded_len - input_ids_length), "constant", 1
    )
    inputs["attention_mask"] = torch.nn.functional.pad(
        inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
    )
    inputs = {name: np.asarray(value) for name, value in inputs.items()}

    vision_inputs = {name: value for name, value in inputs.items() if name in vision_session.input_names}
    if "pixel_values" in vision_inputs:
        vision_inputs["pixel_values"] = vision_inputs["pixel_values"].astype(np.float16)

    vision_start = time.perf_counter()
    vision_outputs = vision_session.run(vision_inputs)
    vision_time = time.perf_counter() - vision_start

    lang_inputs = {
        name: value
        for name, value in inputs.items()
        if name not in vision_inputs and name in prefill_session.input_names
    }
    if "position_ids" in inputs:
        lang_inputs["position_ids"] = inputs["position_ids"]
        lang_inputs.pop("attention_mask", None)
    if "image_idx" in prefill_session.input_names:
        lang_inputs["image_idx"] = np.array([[0]], dtype=np.int64)
    if "batch_index" in prefill_session.input_names:
        lang_inputs["batch_index"] = np.array([[0]], dtype=np.int64)

    prefill_session.set_buffers(vision_outputs)
    chunk_inputs = dict(lang_inputs)
    prefill_output = None
    prefill_start = time.perf_counter()
    num_chunks = padded_len // args.prefill_seq_len
    for chunk_idx in range(num_chunks):
        lower = chunk_idx * args.prefill_seq_len
        upper = lower + args.prefill_seq_len
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, lower:upper]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][..., lower:upper]
        prefill_output = prefill_session.run(chunk_inputs)
        _update_retained(chunk_inputs, prefill_output)
        if "image_idx_output" in prefill_output:
            chunk_inputs["image_idx"] = prefill_output["image_idx_output"]
    prefill_time = time.perf_counter() - prefill_start

    decode_inputs = _initialize_decode_states(decode_session)
    _update_retained(decode_inputs, prefill_output)
    if "image_idx" in decode_session.input_names:
        decode_inputs["image_idx"] = prefill_output["image_idx_output"]
    if "vision_embeds" in decode_session.input_names:
        decode_inputs["vision_embeds"] = prefill_output["vision_embeds_RetainedState"]
    decode_inputs["input_ids"] = _next_token(prefill_output["logits"])
    decode_inputs["position_ids"] = np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1

    generated = [decode_inputs["input_ids"].copy()]
    decode_start = time.perf_counter()
    for _ in range(args.generation_len - 1):
        decode_output = decode_session.run(decode_inputs)
        _update_retained(decode_inputs, decode_output)
        decode_inputs["input_ids"] = _next_token(decode_output["logits"])
        decode_inputs["position_ids"] += 1
        if "image_idx_output" in decode_output:
            decode_inputs["image_idx"] = decode_output["image_idx_output"]
        if "vision_embeds_RetainedState" in decode_output:
            decode_inputs["vision_embeds"] = decode_output["vision_embeds_RetainedState"]
        generated.append(decode_inputs["input_ids"].copy())
    decode_time = time.perf_counter() - decode_start

    generated_ids = np.asarray(generated).reshape(-1).tolist()
    print(f"prompt={args.prompt!r}")
    print(f"image={args.image} source_size={source_size} model_image_size={image.size}")
    print(f"input_tokens={input_ids_length} padded_tokens={padded_len} chunks={num_chunks} thinking_disabled=True")
    print(f"vision_time_s={vision_time:.4f}")
    print(f"prefill_time_s={prefill_time:.4f}")
    print(f"decode_time_s={decode_time:.4f} decode_tokens={len(generated_ids)}")
    print(f"generated_token_ids={generated_ids}")
    print(f"generated_text={tokenizer.decode(generated_ids, skip_special_tokens=False)!r}")


if __name__ == "__main__":
    main()
