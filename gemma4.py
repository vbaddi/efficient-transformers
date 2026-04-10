import argparse
import copy
import json
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
import onnxruntime as ort
import requests
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.base.onnx_transforms import FP16ClipTransform
from QEfficient.utils.cache import QEFF_HOME
from QEfficient.utils.run_utils import ApiRunner

torch.manual_seed(42)


def build_example_artifact_dir(model_name: str, kind: str) -> Path:
    safe_model_name = model_name.replace("/", "--")
    artifact_dir = QEFF_HOME / "example_runs" / safe_model_name / f"{kind}-{uuid4().hex[:12]}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def print_perf_stats(exec_info):
    if not hasattr(exec_info, "perf_metrics"):
        return
    print("\n========================= Performance Stats =========================")
    print(exec_info)
    print("=====================================================================")


def normalize_generated_ids(generated_ids):
    array = np.asarray(generated_ids)
    if array.dtype == object:
        array = np.asarray([np.asarray(row).reshape(-1) for row in generated_ids], dtype=np.int64)
    array = np.asarray(array)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return array.astype(np.int64, copy=False)


def parse_device_group(device_ids):
    device_ids = device_ids.strip()
    if not device_ids:
        return None
    return [int(x) for x in device_ids.strip("[]").split(",") if x.strip()]


def resolve_effective_ctx_len(requested_ctx_len: int, prompt_len: int, generation_len: int) -> int:
    required_ctx_len = prompt_len + generation_len
    if required_ctx_len > requested_ctx_len:
        raise ValueError(
            f"Requested ctx_len={requested_ctx_len} is too small for prompt_len={prompt_len} "
            f"and generation_len={generation_len}. Need at least {required_ctx_len}."
        )
    return requested_ctx_len


def resolve_effective_prefill_seq_len(requested_prefill_seq_len: int, inputs, use_image: bool) -> int:
    del inputs, use_image
    return requested_prefill_seq_len


def resolve_subfunction_settings(args, use_image: bool, is_moe: bool):
    if use_image:
        effective_vision_subfunctions = bool(args.vision_use_onnx_subfunctions and not args.disable_vision_subfunctions)
        effective_lang_subfunctions = not args.disable_lang_subfunctions and (
            args.use_onnx_subfunctions or args.lang_use_onnx_subfunctions or not is_moe
        )
        effective_use_onnx_subfunctions = effective_vision_subfunctions or effective_lang_subfunctions
    else:
        effective_vision_subfunctions = None
        effective_lang_subfunctions = not args.disable_lang_subfunctions and (
            args.use_onnx_subfunctions or args.lang_use_onnx_subfunctions or not is_moe
        )
        effective_use_onnx_subfunctions = effective_lang_subfunctions

    return effective_use_onnx_subfunctions, effective_vision_subfunctions, effective_lang_subfunctions


def is_gemma4_moe_config(config) -> bool:
    return bool(
        getattr(config, "enable_moe_block", False)
        or (getattr(config, "num_experts", 0) or 0) > 0
        or (getattr(config, "moe_intermediate_size", 0) or 0) > 0
    )


def build_messages(system_prompt: str, user_prompt: str, use_image: bool):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if use_image:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": user_prompt})
    return messages


def load_image(image_source: str):
    if image_source.startswith(("http://", "https://")):
        response = requests.get(image_source, stream=True, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_source).convert("RGB")


def prepare_inputs(processor, system_prompt: str, user_prompt: str, image_source: str | None):
    use_image = image_source is not None
    messages = build_messages(system_prompt, user_prompt, use_image)
    if use_image:
        rendered_messages = copy.deepcopy(messages)
        rendered_messages[-1]["content"][0]["url"] = image_source
        rendered_prompt = processor.tokenizer.apply_chat_template(
            rendered_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=rendered_prompt, images=load_image(image_source), return_tensors="pt")
    else:
        rendered_prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
    return rendered_prompt, inputs


def load_gemma4_text_model(model_name: str):
    tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True).tokenizer
    if hasattr(tokenizer, "model_input_names"):
        tokenizer.model_input_names = ["input_ids", "attention_mask"]

    full_model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=False,
        torch_dtype=torch.float32,
    ).to(torch.float32)
    full_model.eval()

    text_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True).text_config
    text_model = AutoModelForCausalLM.from_config(
        text_config,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(torch.float32)
    text_model.model.load_state_dict(full_model.model.language_model.state_dict())
    text_model.lm_head.load_state_dict(full_model.lm_head.state_dict())
    text_model = text_model.to(torch.float32)
    text_model.eval()
    return tokenizer, text_model


def run_hf_verification(
    model_name: str, inputs, generation_len: int, reference_dtype: str, use_image: bool, hf_model=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if reference_dtype == "fp16" else torch.float32
    if hf_model is not None:
        model = hf_model.to(device=device, dtype=dtype)
    elif use_image:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=False,
            torch_dtype=dtype,
        ).to(device=device, dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=False,
            torch_dtype=dtype,
        ).to(device=device, dtype=dtype)
    model.eval()
    model_inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(**model_inputs, max_new_tokens=generation_len, do_sample=False)
    prompt_len = model_inputs["input_ids"].shape[1]
    return outputs[:, prompt_len:].cpu()


def build_text_ort_tokens(
    tokenizer,
    config,
    rendered_prompt: str,
    onnx_path: str,
    generation_len: int,
    prefill_seq_len: int,
    ctx_len: int,
):
    def retained(outputs: dict[str, np.ndarray], name: str):
        retained_name = f"{name}_RetainedState"
        if retained_name in outputs:
            return outputs[retained_name]
        return outputs[f"{name}_InternalRetainedState"]

    encoded = tokenizer(rendered_prompt, return_tensors="np")
    input_ids = encoded["input_ids"].astype(np.int64, copy=False)
    attention_mask = encoded["attention_mask"].astype(np.int64, copy=False)
    input_len = int(input_ids.shape[1])
    padded_len = int(np.ceil(input_len / prefill_seq_len) * prefill_seq_len)
    pad_len = padded_len - input_len
    if pad_len:
        input_ids = np.pad(input_ids, ((0, 0), (0, pad_len)), constant_values=tokenizer.pad_token_id)
        attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_len)), constant_values=0)
    position_ids = np.where(attention_mask, np.arange(padded_len, dtype=np.int64), -1)

    past_key_values = []
    for layer_type in config.layer_types:
        if layer_type == "sliding_attention":
            n_heads = config.num_key_value_heads
            d_head = config.head_dim
            layer_seq_len = min(config.sliding_window, ctx_len)
        else:
            use_alternative_attention = getattr(config, "attention_k_eq_v", False)
            n_heads = (
                config.num_global_key_value_heads
                if use_alternative_attention and getattr(config, "num_global_key_value_heads", None) is not None
                else config.num_key_value_heads
            )
            d_head = config.global_head_dim if getattr(config, "global_head_dim", None) else config.head_dim
            layer_seq_len = ctx_len
        cache_shape = (1, n_heads, layer_seq_len, d_head)
        past_key_values.append(
            (
                np.zeros(cache_shape, dtype=np.float32),
                np.zeros(cache_shape, dtype=np.float32),
            )
        )

    session = ort.InferenceSession(str(onnx_path))
    inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
    }
    for i, (key, value) in enumerate(past_key_values):
        inputs[f"past_key.{i}"] = key
        inputs[f"past_value.{i}"] = value

    num_layers = len(past_key_values)
    outputs = None
    for chunk_start in range(0, padded_len, prefill_seq_len):
        chunk_end = chunk_start + prefill_seq_len
        chunk_inputs = {
            "input_ids": inputs["input_ids"][:, chunk_start:chunk_end],
            "position_ids": inputs["position_ids"][:, chunk_start:chunk_end],
        }
        for i in range(num_layers):
            chunk_inputs[f"past_key.{i}"] = inputs[f"past_key.{i}"]
            chunk_inputs[f"past_value.{i}"] = inputs[f"past_value.{i}"]
        outputs = run_ort_session(session, chunk_inputs)
        for i in range(num_layers):
            inputs[f"past_key.{i}"] = retained(outputs, f"past_key.{i}")
            inputs[f"past_value.{i}"] = retained(outputs, f"past_value.{i}")

    if outputs is None:
        raise RuntimeError("No text prefill chunk was executed.")

    generated = [outputs["logits"].argmax(-1).astype(np.int64).reshape(-1, 1)]
    decode_inputs = {
        "input_ids": generated[0],
        "position_ids": np.max(position_ids, axis=1, keepdims=True) + 1,
    }
    for i in range(num_layers):
        decode_inputs[f"past_key.{i}"] = inputs[f"past_key.{i}"]
        decode_inputs[f"past_value.{i}"] = inputs[f"past_value.{i}"]

    for _ in range(1, generation_len):
        outputs = run_ort_session(session, decode_inputs)
        next_token = outputs["logits"].argmax(-1).astype(np.int64)
        generated.append(next_token.reshape(-1, 1))
        decode_inputs["input_ids"] = next_token
        decode_inputs["position_ids"] = decode_inputs["position_ids"] + 1
        for i in range(num_layers):
            decode_inputs[f"past_key.{i}"] = retained(outputs, f"past_key.{i}")
            decode_inputs[f"past_value.{i}"] = retained(outputs, f"past_value.{i}")

    return np.concatenate(generated, axis=1)


def run_ort_session(session: ort.InferenceSession, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    input_names = {x.name for x in session.get_inputs()}
    feed = {k: v for k, v in inputs.items() if k in input_names}
    output_names = [x.name for x in session.get_outputs()]
    outputs = session.run(output_names, feed)
    return dict(zip(output_names, outputs))


def pad_prefill_inputs(inputs, padded_len: int, pad_token_id: int):
    input_ids = inputs["input_ids"]
    _, input_len = input_ids.shape
    pad_len = padded_len - input_len

    if pad_len == 0:
        return inputs

    padded = dict(inputs)
    padded["input_ids"] = torch.nn.functional.pad(input_ids, (0, pad_len), "constant", pad_token_id)
    padded["attention_mask"] = torch.nn.functional.pad(inputs["attention_mask"], (0, pad_len), "constant", 0)
    padded["mm_token_type_ids"] = torch.nn.functional.pad(inputs["mm_token_type_ids"], (0, pad_len), "constant", 0)
    return padded


def build_initial_decoder_inputs(model, inputs, vision_embeds, ctx_len: int):
    lang_cfg = model.model.model.language_model.config
    seq_len = inputs["input_ids"].shape[1]
    past_key_values = model.model.get_dummy_pkv_cache(lang_cfg, batch_size=1, seq_len=ctx_len)
    position_ids = torch.where(
        inputs["attention_mask"].bool(),
        inputs["attention_mask"].cumsum(1) - 1,
        torch.full_like(inputs["attention_mask"], -1),
    )

    decoder_inputs = {
        "input_ids": inputs["input_ids"].cpu().numpy(),
        "vision_embeds": vision_embeds.astype(np.float32),
        "position_ids": position_ids.cpu().numpy(),
        "image_idx": np.zeros((1, 1), dtype=np.int64),
        "mm_token_type_ids": inputs["mm_token_type_ids"].cpu().numpy(),
    }
    for i, (key, value) in enumerate(past_key_values):
        decoder_inputs[f"past_key.{i}"] = key.cpu().numpy()
        decoder_inputs[f"past_value.{i}"] = value.cpu().numpy()
    return decoder_inputs


def update_decoder_inputs(inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray], num_layers: int):
    def retained(name: str):
        retained_name = f"{name}_RetainedState"
        if retained_name in outputs:
            return outputs[retained_name]
        return outputs[f"{name}_InternalRetainedState"]

    next_token = outputs["logits"].argmax(-1).astype(np.int64)
    next_inputs = {
        "input_ids": next_token,
        "position_ids": np.max(inputs["position_ids"], axis=1, keepdims=True) + 1,
        "image_idx": outputs["image_idx_output"],
        "vision_embeds": outputs.get(
            "vision_embeds_RetainedState",
            outputs.get("vision_embeds_InternalRetainedState", inputs["vision_embeds"]),
        ),
        "mm_token_type_ids": np.zeros_like(next_token, dtype=inputs["mm_token_type_ids"].dtype),
    }
    for i in range(num_layers):
        next_inputs[f"past_key.{i}"] = retained(f"past_key.{i}")
        next_inputs[f"past_value.{i}"] = retained(f"past_value.{i}")
    return next_inputs


def pad_prefill_input_last_dim(array: np.ndarray, target_seq_len: int, pad_value: int) -> np.ndarray:
    pad_amount = target_seq_len - array.shape[-1]
    if pad_amount <= 0:
        return array
    pad_width = [(0, 0)] * array.ndim
    pad_width[-1] = (0, pad_amount)
    return np.pad(array, pad_width, mode="constant", constant_values=pad_value)


def build_vlm_ort_tokens(
    model, onnx_paths, inputs, generation_len: int, ctx_len: int, prefill_seq_len: int, pad_token_id: int
):
    def retained(outputs: dict[str, np.ndarray], name: str):
        retained_name = f"{name}_RetainedState"
        if retained_name in outputs:
            return outputs[retained_name]
        return outputs[f"{name}_InternalRetainedState"]

    vision_session = ort.InferenceSession(str(onnx_paths[0]))
    decoder_session = ort.InferenceSession(str(onnx_paths[1]))

    vision_outputs = run_ort_session(
        vision_session,
        {
            "pixel_values": inputs["pixel_values"].cpu().numpy(),
            "image_position_ids": inputs["image_position_ids"].cpu().numpy(),
        },
    )
    vision_embeds = vision_outputs["vision_embeds"]

    available_prefill_seq_lens = sorted(model.model.get_prefill_seq_lens(prefill_seq_len, kv_offload=True))
    chunk_seq_lens = model.model.resolve_multimodal_prefill_chunk_lens(
        inputs, prefill_seq_len, available_prefill_seq_lens
    )
    actual_chunk_seq_lens = []
    remaining_tokens = int(inputs["input_ids"].shape[1])
    for chunk_seq_len in chunk_seq_lens:
        if remaining_tokens <= 0:
            break
        actual_chunk_seq_lens.append(min(chunk_seq_len, remaining_tokens))
        remaining_tokens -= actual_chunk_seq_lens[-1]

    decoder_state = build_initial_decoder_inputs(model, inputs, vision_embeds, ctx_len=ctx_len)
    num_layers = model.model.model.language_model.config.num_hidden_layers
    chunk_image_idx = np.array([[0]], dtype=np.int64)
    outputs = None

    chunk_start = 0
    for chunk_seq_len in actual_chunk_seq_lens:
        chunk_end = chunk_start + chunk_seq_len
        target_seq_len = next((seq_len for seq_len in available_prefill_seq_lens if seq_len >= chunk_seq_len), None)
        if target_seq_len is None:
            raise ValueError(
                f"No prefill bucket can serve logical multimodal chunk length {chunk_seq_len}; "
                f"available={available_prefill_seq_lens}"
            )
        chunk_inputs = {
            "input_ids": pad_prefill_input_last_dim(
                decoder_state["input_ids"][:, chunk_start:chunk_end], target_seq_len, pad_token_id
            ),
            "position_ids": pad_prefill_input_last_dim(
                decoder_state["position_ids"][:, chunk_start:chunk_end], target_seq_len, -1
            ),
            "vision_embeds": decoder_state["vision_embeds"],
            "image_idx": chunk_image_idx,
            "mm_token_type_ids": pad_prefill_input_last_dim(
                decoder_state["mm_token_type_ids"][:, chunk_start:chunk_end], target_seq_len, 0
            ),
        }
        for i in range(num_layers):
            chunk_inputs[f"past_key.{i}"] = decoder_state[f"past_key.{i}"]
            chunk_inputs[f"past_value.{i}"] = decoder_state[f"past_value.{i}"]

        outputs = run_ort_session(decoder_session, chunk_inputs)
        chunk_image_idx = outputs["image_idx_output"]
        if "vision_embeds_RetainedState" in outputs:
            decoder_state["vision_embeds"] = outputs["vision_embeds_RetainedState"]
        elif "vision_embeds_InternalRetainedState" in outputs:
            decoder_state["vision_embeds"] = outputs["vision_embeds_InternalRetainedState"]
        for i in range(num_layers):
            decoder_state[f"past_key.{i}"] = retained(outputs, f"past_key.{i}")
            decoder_state[f"past_value.{i}"] = retained(outputs, f"past_value.{i}")
        chunk_start = chunk_end

    if outputs is None:
        raise RuntimeError("No multimodal prefill chunk was executed.")

    decoder_inputs = {
        "input_ids": outputs["logits"].argmax(-1).astype(np.int64),
        "position_ids": np.max(decoder_state["position_ids"], axis=1, keepdims=True) + 1,
        "vision_embeds": decoder_state["vision_embeds"],
        "image_idx": chunk_image_idx,
        "mm_token_type_ids": np.zeros_like(
            outputs["logits"].argmax(-1), dtype=decoder_state["mm_token_type_ids"].dtype
        ),
    }
    for i in range(num_layers):
        decoder_inputs[f"past_key.{i}"] = decoder_state[f"past_key.{i}"]
        decoder_inputs[f"past_value.{i}"] = decoder_state[f"past_value.{i}"]

    generated = [decoder_inputs["input_ids"].reshape(-1, 1)]
    for _ in range(1, generation_len):
        outputs = run_ort_session(decoder_session, decoder_inputs)
        generated.append(outputs["logits"].argmax(-1).astype(np.int64).reshape(-1, 1))
        decoder_inputs = update_decoder_inputs(decoder_inputs, outputs, num_layers)
    return np.concatenate(generated, axis=1)


def build_skip_vision_text_ort_tokens(
    model,
    decoder_onnx_path,
    inputs,
    generation_len: int,
    ctx_len: int,
    prefill_seq_len: int,
    pad_token_id: int,
):
    decoder_session = ort.InferenceSession(str(decoder_onnx_path))
    lang_cfg = model.model.model.language_model.config
    mm_tokens_per_image = getattr(model.model.config, "mm_tokens_per_image", 256)
    hidden_size = lang_cfg.hidden_size

    decoder_inputs = {
        "input_ids": inputs["input_ids"].cpu().numpy(),
        "vision_embeds": np.zeros((1, mm_tokens_per_image, hidden_size), dtype=np.float32),
        "position_ids": torch.where(
            inputs["attention_mask"].bool(),
            inputs["attention_mask"].cumsum(1) - 1,
            torch.full_like(inputs["attention_mask"], -1),
        )
        .cpu()
        .numpy(),
        "image_idx": np.zeros((1, 1), dtype=np.int64),
        "mm_token_type_ids": np.zeros_like(inputs["input_ids"].cpu().numpy(), dtype=np.int64),
    }
    past_key_values = model.model.get_dummy_pkv_cache(lang_cfg, batch_size=1, seq_len=ctx_len)
    for i, (key, value) in enumerate(past_key_values):
        decoder_inputs[f"past_key.{i}"] = key.cpu().numpy()
        decoder_inputs[f"past_value.{i}"] = value.cpu().numpy()

    num_layers = lang_cfg.num_hidden_layers
    input_len = int(inputs["input_ids"].shape[1])
    padded_len = int(np.ceil(input_len / prefill_seq_len) * prefill_seq_len)
    pad_len = padded_len - input_len
    if pad_len:
        decoder_inputs["input_ids"] = np.pad(
            decoder_inputs["input_ids"],
            ((0, 0), (0, pad_len)),
            constant_values=pad_token_id,
        )
        decoder_inputs["position_ids"] = np.pad(
            decoder_inputs["position_ids"],
            ((0, 0), (0, pad_len)),
            constant_values=-1,
        )
        decoder_inputs["mm_token_type_ids"] = np.pad(
            decoder_inputs["mm_token_type_ids"],
            ((0, 0), (0, pad_len)),
            constant_values=0,
        )

    outputs = None
    for chunk_start in range(0, padded_len, prefill_seq_len):
        chunk_end = chunk_start + prefill_seq_len
        chunk_inputs = {
            "input_ids": decoder_inputs["input_ids"][:, chunk_start:chunk_end],
            "position_ids": decoder_inputs["position_ids"][:, chunk_start:chunk_end],
            "vision_embeds": decoder_inputs["vision_embeds"],
            "image_idx": decoder_inputs["image_idx"],
            "mm_token_type_ids": decoder_inputs["mm_token_type_ids"][:, chunk_start:chunk_end],
        }
        for i in range(num_layers):
            chunk_inputs[f"past_key.{i}"] = decoder_inputs[f"past_key.{i}"]
            chunk_inputs[f"past_value.{i}"] = decoder_inputs[f"past_value.{i}"]

        outputs = run_ort_session(decoder_session, chunk_inputs)
        decoder_inputs = update_decoder_inputs(chunk_inputs, outputs, num_layers)

    if outputs is None:
        raise RuntimeError("No text prefill chunk was executed for skip_vision Gemma4 decoder.")

    generated = [decoder_inputs["input_ids"].reshape(-1, 1)]
    for _ in range(1, generation_len):
        outputs = run_ort_session(decoder_session, decoder_inputs)
        generated.append(outputs["logits"].argmax(-1).astype(np.int64).reshape(-1, 1))
        decoder_inputs = update_decoder_inputs(decoder_inputs, outputs, num_layers)
    return np.concatenate(generated, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Gemma4 text-only or image+text QAic example.")
    parser.add_argument("--model-name", type=str, default="tiny-random/gemma-4-dense")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--prompt", type=str, default="Hi, Who are you?")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--prefill-seq-len", type=int, default=32)
    parser.add_argument("--ctx-len", type=int, default=128)
    parser.add_argument("--generation-len", type=int, default=16)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--device-group", type=parse_device_group, default=None)
    parser.add_argument("--vision-device-group", type=parse_device_group, default=None)
    parser.add_argument("--lang-device-group", type=parse_device_group, default=None)
    parser.add_argument("--vision-num-devices", type=int, default=None)
    parser.add_argument("--lang-num-devices", type=int, default=None)
    parser.add_argument("--use-onnx-subfunctions", action="store_true")
    parser.add_argument("--vision-use-onnx-subfunctions", action="store_true")
    parser.add_argument("--lang-use-onnx-subfunctions", action="store_true")
    parser.add_argument("--disable-vision-subfunctions", action="store_true")
    parser.add_argument("--disable-lang-subfunctions", action="store_true")
    parser.add_argument("--mxfp6-matmul", action="store_true")
    parser.add_argument("--mxint8-kv-cache", action="store_true")
    parser.add_argument("--aic-enable-depth-first", action="store_true")
    parser.add_argument("--mos", type=int, default=None)
    parser.add_argument("--enable-fp16clip", action="store_true")
    parser.add_argument("--enable-npi", action="store_true")
    parser.add_argument("--disable-npi", action="store_true")
    parser.add_argument("--verify-runtime", action="store_true")
    parser.add_argument("--verify-ort", action="store_true")
    parser.add_argument("--reference-dtype", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()

    use_image = args.image is not None
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = processor.tokenizer
    rendered_prompt, inputs = prepare_inputs(processor, args.system_prompt, args.prompt, args.image)
    prompt_len = int(inputs["input_ids"].shape[1])
    effective_ctx_len = resolve_effective_ctx_len(args.ctx_len, prompt_len, args.generation_len)
    effective_prefill_seq_len = resolve_effective_prefill_seq_len(args.prefill_seq_len, inputs, use_image)
    effective_fp16clip = args.enable_fp16clip or use_image
    effective_convert_to_fp16 = True
    npi_mode = "enabled" if args.enable_npi else "disabled" if args.disable_npi else "auto"
    runtime_lang_device_group = args.lang_device_group or args.device_group
    runtime_vision_device_group = args.vision_device_group or args.device_group
    if runtime_lang_device_group is None and args.lang_num_devices is not None:
        if use_image:
            vision_count = args.vision_num_devices or 1
            runtime_lang_device_group = list(range(vision_count, vision_count + args.lang_num_devices))
        else:
            runtime_lang_device_group = list(range(args.lang_num_devices))
    if use_image and runtime_vision_device_group is None and args.vision_num_devices is not None:
        runtime_vision_device_group = list(range(args.vision_num_devices))
    hf_inputs = {k: v.clone() if hasattr(v, "clone") else v for k, v in inputs.items()}
    hf_ids = None
    ort_ids = None
    onnx_paths = None
    export_dir = build_example_artifact_dir(args.model_name, "export")
    compile_dir = build_example_artifact_dir(args.model_name, "compile")

    if use_image:
        model = QEFFAutoModelForImageTextToText.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            kv_offload=True,
            skip_vision=False,
            dtype="float32",
        )
        lang_config = getattr(getattr(model.model, "model", None), "language_model", None).config
        is_moe = is_gemma4_moe_config(lang_config)
        (
            effective_use_onnx_subfunctions,
            effective_vision_subfunctions,
            effective_lang_subfunctions,
        ) = resolve_subfunction_settings(args, use_image, is_moe)
        if not effective_fp16clip:
            model.lang_model._onnx_transforms = [
                t for t in model.lang_model._onnx_transforms if t is not FP16ClipTransform
            ]
            if getattr(model, "vision_model", None) is not None:
                model.vision_model._onnx_transforms = [
                    t for t in model.vision_model._onnx_transforms if t is not FP16ClipTransform
                ]
        if args.verify_runtime:
            hf_ids = run_hf_verification(args.model_name, hf_inputs, args.generation_len, args.reference_dtype, True)

        onnx_paths = model.export(
            export_dir=export_dir,
            use_onnx_subfunctions=effective_use_onnx_subfunctions,
            vision_use_onnx_subfunctions=effective_vision_subfunctions,
            lang_use_onnx_subfunctions=effective_lang_subfunctions,
        )
        if args.verify_ort:
            ort_ids = build_vlm_ort_tokens(
                model,
                onnx_paths,
                inputs,
                args.generation_len,
                ctx_len=effective_ctx_len,
                prefill_seq_len=effective_prefill_seq_len,
                pad_token_id=tokenizer.pad_token_id,
            )
        compile_kwargs = dict(
            vision_onnx_path=str(onnx_paths[0]),
            lang_onnx_path=str(onnx_paths[1]),
            prefill_seq_len=effective_prefill_seq_len,
            ctx_len=effective_ctx_len,
            num_devices=(
                args.lang_num_devices
                or (len(runtime_lang_device_group) if runtime_lang_device_group is not None else 1)
            ),
            vision_num_devices=(
                args.vision_num_devices
                or (len(runtime_vision_device_group) if runtime_vision_device_group is not None else 1)
            ),
            lang_num_devices=(
                args.lang_num_devices
                or (len(runtime_lang_device_group) if runtime_lang_device_group is not None else 1)
            ),
            num_cores=args.num_cores,
            mxfp6_matmul=args.mxfp6_matmul,
            mxint8_kv_cache=args.mxint8_kv_cache,
            aic_enable_depth_first=args.aic_enable_depth_first,
            use_onnx_subfunctions=effective_use_onnx_subfunctions,
            vision_use_onnx_subfunctions=effective_vision_subfunctions,
            lang_use_onnx_subfunctions=effective_lang_subfunctions,
        )
        if npi_mode == "enabled":
            compile_kwargs["node_precision_info"] = True
        elif npi_mode == "disabled":
            compile_kwargs["node_precision_info"] = False
            compile_kwargs["vision_node_precision_info"] = False
        if args.mos is not None:
            compile_kwargs["mos"] = args.mos

        qpc_path = model.compile(compile_dir=compile_dir, **compile_kwargs)
        exec_info = model.generate(
            inputs=inputs,
            device_ids=args.device_group,
            vision_device_ids=runtime_vision_device_group,
            lang_device_ids=runtime_lang_device_group,
            generation_len=args.generation_len,
        )
        print_perf_stats(exec_info)
    else:
        tokenizer, hf_text_model = load_gemma4_text_model(args.model_name)
        model = QEFFAutoModelForImageTextToText.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            kv_offload=True,
            skip_vision=True,
            dtype="float32",
        )
        if not effective_fp16clip:
            model.lang_model._onnx_transforms = [
                t for t in model.lang_model._onnx_transforms if t is not FP16ClipTransform
            ]
        (
            effective_text_subfunctions,
            effective_vision_subfunctions,
            effective_lang_subfunctions,
        ) = resolve_subfunction_settings(args, use_image, False)
        if args.verify_runtime:
            hf_ids = run_hf_verification(
                args.model_name,
                hf_inputs,
                args.generation_len,
                args.reference_dtype,
                False,
                hf_model=hf_text_model,
            )

        onnx_paths = model.export(
            export_dir=export_dir,
            skip_vision=True,
            use_onnx_subfunctions=effective_text_subfunctions,
            lang_use_onnx_subfunctions=effective_text_subfunctions,
        )
        onnx_path = onnx_paths[1] if isinstance(onnx_paths, (list, tuple)) else onnx_paths
        if args.verify_ort:
            ort_ids = build_skip_vision_text_ort_tokens(
                model=model,
                decoder_onnx_path=str(onnx_path),
                inputs=inputs,
                generation_len=args.generation_len,
                prefill_seq_len=effective_prefill_seq_len,
                ctx_len=effective_ctx_len,
                pad_token_id=tokenizer.pad_token_id,
            )
        compile_kwargs = dict(
            skip_vision=True,
            lang_onnx_path=str(onnx_path),
            prefill_seq_len=effective_prefill_seq_len,
            ctx_len=effective_ctx_len,
            num_devices=(
                args.lang_num_devices
                or (len(runtime_lang_device_group) if runtime_lang_device_group is not None else 1)
            ),
            lang_num_devices=(
                args.lang_num_devices
                or (len(runtime_lang_device_group) if runtime_lang_device_group is not None else 1)
            ),
            num_cores=args.num_cores,
            mxfp6_matmul=args.mxfp6_matmul,
            mxint8_kv_cache=args.mxint8_kv_cache,
            aic_enable_depth_first=args.aic_enable_depth_first,
            use_onnx_subfunctions=effective_text_subfunctions,
            lang_use_onnx_subfunctions=effective_text_subfunctions,
        )
        if npi_mode == "enabled":
            compile_kwargs["node_precision_info"] = True
        elif npi_mode == "disabled":
            compile_kwargs["node_precision_info"] = False
        if args.mos is not None:
            compile_kwargs["mos"] = args.mos

        qpc_path = model.compile(compile_dir=compile_dir, **compile_kwargs)
        exec_info = model.generate(
            prompts=[rendered_prompt],
            tokenizer=tokenizer,
            device_ids=args.device_group,
            lang_device_ids=runtime_lang_device_group,
            generation_len=args.generation_len,
        )
        onnx_paths = [str(onnx_path)]
        effective_use_onnx_subfunctions = effective_text_subfunctions
        effective_lang_subfunctions = effective_use_onnx_subfunctions

    qeff_ids = normalize_generated_ids(exec_info.generated_ids)[:, : args.generation_len]
    qeff_text = tokenizer.batch_decode(qeff_ids, skip_special_tokens=True)

    print("\nRendered prompt:")
    print(rendered_prompt)
    print("\nCompile settings:")
    print(
        json.dumps(
            {
                "image_mode": use_image,
                "skip_vision": not use_image,
                "kv_offload": True,
                "use_onnx_subfunctions": effective_use_onnx_subfunctions,
                "vision_use_onnx_subfunctions": effective_vision_subfunctions,
                "lang_use_onnx_subfunctions": effective_lang_subfunctions,
                "vision_num_devices": (
                    args.vision_num_devices
                    or (len(runtime_vision_device_group) if runtime_vision_device_group is not None else None)
                ),
                "lang_num_devices": (
                    args.lang_num_devices
                    or (len(runtime_lang_device_group) if runtime_lang_device_group is not None else None)
                ),
                "mxfp6_matmul": args.mxfp6_matmul,
                "mxint8_kv_cache": args.mxint8_kv_cache,
                "npi_mode": npi_mode,
                "fp16clip_enabled": effective_fp16clip,
                "convert_to_fp16": effective_convert_to_fp16,
                "prompt_len": prompt_len,
                "requested_prefill_seq_len": args.prefill_seq_len,
                "effective_prefill_seq_len": effective_prefill_seq_len,
                "requested_ctx_len": args.ctx_len,
                "effective_ctx_len": effective_ctx_len,
                "onnx_paths": [str(x) for x in onnx_paths] if onnx_paths is not None else None,
                "qpc_path": [str(x) for x in qpc_path] if isinstance(qpc_path, (list, tuple)) else str(qpc_path),
            },
            indent=2,
        )
    )

    print("\nQEff generated ids:")
    print(qeff_ids.tolist())
    print("QEff generated text:")
    print(qeff_text)

    if args.verify_ort:
        print("\nORT generated ids:")
        print(ort_ids.tolist())
        print("ORT generated text:")
        print(tokenizer.batch_decode(ort_ids, skip_special_tokens=True))

    if args.verify_runtime:
        hf_text = tokenizer.batch_decode(hf_ids, skip_special_tokens=True)
        qeff_prefix = qeff_ids[:, : hf_ids.shape[1]]
        print("\nHF vs QEff parity:")
        print(
            json.dumps(
                {
                    "image_mode": use_image,
                    "rendered_prompt": rendered_prompt,
                    "reference_dtype": args.reference_dtype,
                    "hf_ids": hf_ids.tolist(),
                    "qeff_ids_prefix": qeff_prefix.tolist(),
                    "hf_text": hf_text,
                    "qeff_text_prefix": tokenizer.batch_decode(qeff_prefix, skip_special_tokens=True),
                    "match": bool(np.array_equal(hf_ids, qeff_prefix)),
                },
                indent=2,
            )
        )
        if args.verify_ort:
            print("\nHF vs ORT parity:")
            print(
                json.dumps(
                    {
                        "image_mode": use_image,
                        "rendered_prompt": rendered_prompt,
                        "reference_dtype": args.reference_dtype,
                        "hf_ids": hf_ids.tolist(),
                        "ort_ids": ort_ids.tolist(),
                        "hf_text": hf_text,
                        "ort_text": tokenizer.batch_decode(ort_ids, skip_special_tokens=True),
                        "match": bool(np.array_equal(hf_ids, ort_ids)),
                    },
                    indent=2,
                )
            )
            print("\nORT vs QEff parity:")
            print(
                json.dumps(
                    {
                        "image_mode": use_image,
                        "ort_ids": ort_ids.tolist(),
                        "qeff_ids": qeff_ids.tolist(),
                        "ort_text": tokenizer.batch_decode(ort_ids, skip_special_tokens=True),
                        "qeff_text": qeff_text,
                        "match": bool(np.array_equal(ort_ids, qeff_ids)),
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
