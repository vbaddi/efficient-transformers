# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
from typing import Optional

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM


def parse_device_group(raw: Optional[str]):
    if raw is None:
        return None
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def first_text(generated_texts):
    out = generated_texts
    while isinstance(out, list):
        if not out:
            return ""
        out = out[0]
    return str(out)


def print_turbo_metrics(metrics):
    line = "=" * 69
    print(f"\n{line}")
    print("TurboQuant Metrics")
    print(f"requested         : {metrics.get('requested')}")
    print(f"host_kv_mode      : {metrics.get('host_kv_mode')}")
    print(f"supported         : {metrics.get('supported')}")
    print(f"reason            : {metrics.get('reason')}")
    print(f"total_bits        : {metrics.get('total_bits')}")
    ratio = metrics.get("compression_ratio")
    if ratio is not None:
        print(f"kv fp16 size      : {metrics.get('fp16_mb', 0.0):.2f} MB")
        print(f"kv turbo size     : {metrics.get('compressed_mb', 0.0):.2f} MB")
        print(f"compression ratio : {ratio:.2f}x")
    print(line)


def main():
    parser = argparse.ArgumentParser(description="Llama TurboQuant sample on QAIC")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HF model id/path")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain in one paragraph how prefill and decode differ in LLM inference.",
        help="Input prompt",
    )
    parser.add_argument("--prefill-seq-len", type=int, default=32, help="Prefill sequence length for compile")
    parser.add_argument("--ctx-len", type=int, default=128, help="Context length for compile")
    parser.add_argument("--generation-len", type=int, default=64, help="Number of output tokens to generate")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of QAIC cores")
    parser.add_argument("--device-group", type=str, default=None, help="Device ids, e.g. [0] or 0,1")
    parser.add_argument("--mxfp6", action="store_true", help="Enable MXFP6 weights during compile")
    parser.add_argument("--mxint8-kv-cache", action="store_true", help="Enable MXINT8 KV cache during compile")
    parser.add_argument("--turbo-total-bits", type=int, default=3, choices=[2, 3, 4], help="TurboQuant total bits")
    args = parser.parse_args()

    device_group = parse_device_group(args.device_group)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_name,
        enable_turbo=True,
        turbo_total_bits=args.turbo_total_bits,
    )

    qpc_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=(1 if device_group is None else len(device_group)),
        mxfp6_matmul=args.mxfp6,
        mxint8_kv_cache=args.mxint8_kv_cache,
    )
    print(f"Model compiled to: {qpc_path}")

    exec_info = model.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt],
        device_id=device_group,
        generation_len=args.generation_len,
        stream=False,
    )

    print(f"\nPrompt:\n{args.prompt}")
    print(f"\nGenerated:\n{first_text(exec_info.generated_texts)}")
    print(f"\nTTFT={exec_info.perf_metrics.prefill_time:.3f}s")
    print(f"Decode tok/s={exec_info.perf_metrics.decode_perf * exec_info.batch_size:.2f}")
    print(f"Total tok/s={exec_info.perf_metrics.total_perf * exec_info.batch_size:.2f}")
    print(f"E2E time={exec_info.perf_metrics.total_time:.3f}s")

    print_turbo_metrics(exec_info.turbo_metrics or {})


if __name__ == "__main__":
    main()
