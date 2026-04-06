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


def run_once(model, tokenizer, prompt, generation_len, device_group, enable_turbo, turbo_total_bits):
    model.enable_turbo = enable_turbo
    model.turbo_total_bits = turbo_total_bits
    return model.generate(
        tokenizer=tokenizer,
        prompts=[prompt],
        device_id=device_group,
        generation_len=generation_len,
        stream=False,
    )


def print_summary(args, baseline_info, turbo_info):
    bs_base = max(int(baseline_info.batch_size), 1)
    bs_turbo = max(int(turbo_info.batch_size), 1)

    base_ttft = float(baseline_info.perf_metrics.prefill_time)
    base_decode = float(baseline_info.perf_metrics.decode_perf) * bs_base
    base_total = float(baseline_info.perf_metrics.total_perf) * bs_base
    base_e2e = float(baseline_info.perf_metrics.total_time)

    turbo_ttft = float(turbo_info.perf_metrics.prefill_time)
    turbo_decode = float(turbo_info.perf_metrics.decode_perf) * bs_turbo
    turbo_total = float(turbo_info.perf_metrics.total_perf) * bs_turbo
    turbo_e2e = float(turbo_info.perf_metrics.total_time)

    turbo_metrics = turbo_info.turbo_metrics or {}
    ratio = turbo_metrics.get("compression_ratio")
    ratio_text = f"{ratio:.2f}x" if isinstance(ratio, (int, float)) else "n/a"

    line = "=" * 94
    print(f"\n{line}")
    print("TurboQuant QAIC Benchmark (baseline vs turbo)")
    print(f"model={args.model_name} | prefill_seq_len={args.prefill_seq_len} | ctx_len={args.ctx_len}")
    print(f"generation_len={args.generation_len} | turbo_total_bits={args.turbo_total_bits}")
    print(line)
    print(f"{'Run':<12}{'TTFT (s)':>12}{'Decode tok/s':>16}{'Total tok/s':>16}{'E2E time (s)':>16}{'KV ratio':>12}")
    print(line)
    print(f"{'baseline':<12}{base_ttft:>12.3f}{base_decode:>16.2f}{base_total:>16.2f}{base_e2e:>16.3f}{'-':>12}")
    print(
        f"{'turbo':<12}{turbo_ttft:>12.3f}{turbo_decode:>16.2f}{turbo_total:>16.2f}{turbo_e2e:>16.3f}{ratio_text:>12}"
    )
    print(line)

    if turbo_metrics.get("requested") and not turbo_metrics.get("host_kv_mode"):
        print(f"Turbo mode not active: {turbo_metrics.get('reason', 'unsupported configuration')}")
    if ratio is not None and isinstance(ratio, (int, float)):
        print(
            "Measured host KV sizes: "
            f"fp16={turbo_metrics.get('fp16_mb', 0.0):.2f} MB, "
            f"turbo={turbo_metrics.get('compressed_mb', 0.0):.2f} MB"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark baseline vs TurboQuant on the same QAIC model/config")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HF model id/path")
    parser.add_argument(
        "--prompt", type=str, default="Write two lines about KV cache compression.", help="Input prompt"
    )
    parser.add_argument("--prefill-seq-len", type=int, default=32, help="Prefill sequence length for compile")
    parser.add_argument("--ctx-len", type=int, default=128, help="Context length for compile")
    parser.add_argument("--generation-len", type=int, default=64, help="Number of output tokens to generate")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of QAIC cores")
    parser.add_argument("--device-group", type=str, default=None, help="Device ids, e.g. [0] or 0,1")
    parser.add_argument("--mxfp6", action="store_true", help="Enable MXFP6 weights during compile")
    parser.add_argument("--mxint8-kv-cache", action="store_true", help="Enable MXINT8 KV cache during compile")
    parser.add_argument("--turbo-total-bits", type=int, default=3, choices=[2, 3, 4], help="TurboQuant total bits")
    parser.add_argument("--warmup-runs", type=int, default=0, help="Optional warmup iterations per mode")
    args = parser.parse_args()

    device_group = parse_device_group(args.device_group)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_name,
        enable_turbo=False,
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

    for _ in range(max(args.warmup_runs, 0)):
        run_once(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            generation_len=args.generation_len,
            device_group=device_group,
            enable_turbo=False,
            turbo_total_bits=args.turbo_total_bits,
        )
        run_once(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            generation_len=args.generation_len,
            device_group=device_group,
            enable_turbo=True,
            turbo_total_bits=args.turbo_total_bits,
        )

    baseline_info = run_once(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        generation_len=args.generation_len,
        device_group=device_group,
        enable_turbo=False,
        turbo_total_bits=args.turbo_total_bits,
    )
    turbo_info = run_once(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        generation_len=args.generation_len,
        device_group=device_group,
        enable_turbo=True,
        turbo_total_bits=args.turbo_total_bits,
    )

    print_summary(args, baseline_info, turbo_info)
    print("\nBaseline sample output:\n" + first_text(baseline_info.generated_texts))
    print("\nTurbo sample output:\n" + first_text(turbo_info.generated_texts))


if __name__ == "__main__":
    main()
