# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from QEfficient.utils.layer_benchmarks.artifacts import parse_device_ids
from QEfficient.utils.layer_benchmarks.comparison import compare_qpc_artifacts
from QEfficient.utils.layer_benchmarks.contracts import STAGES, RuntimeOptions
from QEfficient.utils.layer_benchmarks.registry import built_in_benchmarks
from QEfficient.utils.layer_benchmarks.runner import run_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QEfficient module-level benchmark runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List registered layer benchmarks.")

    compare_parser = subparsers.add_parser("compare", help="Compare two compiled benchmark artifact directories.")
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument("--device-group", default="[0]")
    compare_parser.add_argument("--warmup", type=int, default=5)
    compare_parser.add_argument("--iterations", type=int, default=1000)
    compare_parser.add_argument("--samples", type=int, default=3)

    run_parser = subparsers.add_parser("run", help="Prepare, compile, run, or profile a layer benchmark.")
    benchmark_parsers = run_parser.add_subparsers(dest="benchmark", required=True)
    for name, benchmark in built_in_benchmarks().items():
        benchmark_parser = benchmark_parsers.add_parser(name, help=benchmark.description)
        benchmark_parser.add_argument("--stage", choices=STAGES, default="prepare")
        benchmark_parser.add_argument("--artifact-root", type=Path, default=Path(".cache/qeff-layer-benchmarks"))
        benchmark_parser.add_argument("--artifact-dir", type=Path)
        benchmark_parser.add_argument("--seed", type=int, default=1234)
        benchmark_parser.add_argument("--atol", type=float, default=1e-4)
        benchmark_parser.add_argument("--rtol", type=float, default=1e-4)
        benchmark_parser.add_argument("--hw-version", choices=("ai100", "ai200"), default="ai100")
        benchmark_parser.add_argument("--num-cores", "--compile-num-cores", dest="num_cores", type=int)
        benchmark_parser.add_argument("--num-devices", type=int)
        benchmark_parser.add_argument("--device-group", default="[0]")
        benchmark_parser.add_argument("--warmup", "--hw-warmup", dest="warmup", type=int, default=5)
        benchmark_parser.add_argument("--iterations", "--hw-iters", dest="iterations", type=int, default=50)
        benchmark_parser.add_argument(
            "--perf-iterations", "--perf-num-iters", dest="perf_iterations", type=int, default=30
        )
        benchmark_parser.add_argument(
            "--profile-start-iteration",
            "--perf-profile-start-iter",
            dest="profile_start_iteration",
            type=int,
            default=20,
        )
        benchmark_parser.add_argument(
            "--profile-samples", "--perf-num-samples", dest="profile_samples", type=int, default=2
        )
        benchmark_parser.add_argument("--perf-stats-level", type=int, default=70)
        benchmark_parser.add_argument("--extra-compiler-arg", action="append", default=[])
        benchmark_parser.add_argument("--extra-compiler-args", default="")
        benchmark_parser.add_argument("--compiler-lib-dir", type=Path)
        benchmark_parser.add_argument("--enable-mxfp6", action=argparse.BooleanOptionalAction, default=True)
        benchmark_parser.add_argument("--allow-mxint8-mdp-io", action="store_true")
        benchmark_parser.add_argument("--dump-io", action="store_true")
        benchmark_parser.add_argument("--run-compile", action="store_true")
        benchmark_parser.add_argument("--run-hw", action="store_true")
        benchmark_parser.add_argument("--run-perf", action="store_true")
        benchmark_parser.add_argument("--json", action="store_true")
        benchmark.configure_parser(benchmark_parser)
    return parser


def main() -> None:
    args = _parser().parse_args()
    benchmarks = built_in_benchmarks()
    if args.command == "list":
        for name, benchmark in benchmarks.items():
            print(f"{name}: {benchmark.description}")
        return
    if args.command == "compare":
        result = compare_qpc_artifacts(
            args.baseline,
            args.candidate,
            device_ids=parse_device_ids(args.device_group),
            warmup=args.warmup,
            iterations=args.iterations,
            samples=args.samples,
        )
        print(json.dumps(result, indent=2))
        return

    benchmark = benchmarks[args.benchmark]
    config = benchmark.resolved_config(args)
    device_ids = parse_device_ids(args.device_group)
    if args.num_devices is not None and args.num_devices != len(device_ids):
        raise ValueError(f"--num-devices={args.num_devices} disagrees with --device-group={args.device_group}.")
    if args.run_perf:
        args.stage = "profile"
    elif args.run_hw:
        args.stage = "run"
    elif args.run_compile:
        args.stage = "compile"
    num_cores = args.num_cores if args.num_cores is not None else (16 if args.hw_version == "ai100" else 4)
    options = RuntimeOptions(
        stage=args.stage,
        artifact_root=args.artifact_root,
        artifact_dir=args.artifact_dir,
        seed=args.seed,
        atol=args.atol,
        rtol=args.rtol,
        hw_version=args.hw_version,
        num_cores=num_cores,
        device_ids=device_ids,
        warmup=args.warmup,
        iterations=args.iterations,
        perf_iterations=args.perf_iterations,
        profile_start_iteration=args.profile_start_iteration,
        profile_samples=args.profile_samples,
        perf_stats_level=args.perf_stats_level,
        enable_mxfp6=args.enable_mxfp6,
        allow_mxint8_mdp_io=args.allow_mxint8_mdp_io or config.get("kv_cache_dtype") == "mxint8",
        extra_compiler_args=tuple(args.extra_compiler_arg) + tuple(shlex.split(args.extra_compiler_args)),
        compiler_lib_dir=args.compiler_lib_dir,
    )
    run_dir, result = run_benchmark(benchmark, config, options)
    if args.json:
        print((run_dir / "result.json").read_text(), end="")
        return
    print(f"Benchmark: {benchmark.name}")
    print(f"Status   : {result['status']}")
    print(f"Stage    : {options.stage}")
    print(f"Artifacts: {run_dir}")
    parity = result["stages"]["parity"]
    print(f"Parity   : {'PASSED' if all(item['passed'] for item in parity) else 'FAILED'}")
    if options.stage == "run":
        print(f"Latency  : {result['stages']['run']['latency_ms']:.3f} ms")
    print(json.dumps({"result_json": str(run_dir / "result.json")}, indent=2))


if __name__ == "__main__":
    main()
