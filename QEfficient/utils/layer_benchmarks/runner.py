# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import onnx
import onnxscript
import torch

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.layer_benchmarks.artifacts import (
    clean_qpc,
    dump_runner_io,
    emit_compile_assets,
    emit_perf_scripts,
)
from QEfficient.utils.layer_benchmarks.contracts import (
    STAGES,
    LayerBenchmark,
    RuntimeOptions,
    as_output_tuple,
    compare_outputs,
    numpy_outputs,
)

RESULT_SCHEMA_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _run_id(benchmark_name: str, config: Mapping[str, Any], options: RuntimeOptions) -> str:
    identity = {
        "benchmark": benchmark_name,
        "config": config,
        "hardware": {
            "hw_version": options.hw_version,
            "num_cores": options.num_cores,
            "device_ids": options.device_ids,
        },
    }
    encoded = json.dumps(identity, sort_keys=True, default=_json_default).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxscript": onnxscript.__version__,
    }


def _execute(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    result = {
        "command": list(command),
        "returncode": completed.returncode,
        "duration_s": time.perf_counter() - started,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return result


def _compiler_environment(options: RuntimeOptions) -> dict[str, str]:
    environment = dict(os.environ)
    if options.compiler_lib_dir is not None:
        environment["AIC_COMPILER_LIB_DIR"] = str(options.compiler_lib_dir)
        environment["QAIC_COMPILER_LIB"] = str(options.compiler_lib_dir / "libQAicCompiler.so")
    return environment


def _hardware_run(
    qpc_dir: Path,
    inputs: Mapping[str, np.ndarray],
    retained_names: set[str],
    options: RuntimeOptions,
) -> tuple[dict[str, np.ndarray], float]:
    session = QAICInferenceSession(qpc_dir, device_ids=list(options.device_ids))
    feed = {name: value for name, value in inputs.items() if name not in retained_names}
    for _ in range(options.warmup):
        session.run(feed)
    started = time.perf_counter()
    outputs: dict[str, np.ndarray] = {}
    for _ in range(options.iterations):
        outputs = session.run(feed)
    latency_ms = (time.perf_counter() - started) * 1000.0 / options.iterations
    return outputs, latency_ms


def run_benchmark(
    benchmark: LayerBenchmark,
    config: Mapping[str, Any],
    options: RuntimeOptions,
) -> tuple[Path, dict[str, Any]]:
    if options.stage not in STAGES:
        raise ValueError(f"Unknown stage {options.stage!r}; expected one of {STAGES}.")
    torch.manual_seed(options.seed)
    run_dir = options.artifact_dir or (
        options.artifact_root / benchmark.name / _run_id(benchmark.name, config, options)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "benchmark": benchmark.name,
        "run_id": run_dir.name,
        "status": "running",
        "stage": options.stage,
        "config": dict(config),
        "runtime": asdict(options),
        "environment": _environment(),
        "artifacts": {"run_dir": str(run_dir)},
        "stages": {},
    }

    try:
        case = benchmark.build_case(config, options.seed)
        positional_inputs = tuple(case.inputs.values())
        case.module.eval()
        with torch.no_grad():
            reference_outputs = as_output_tuple(case.reference(*positional_inputs))
            candidate_outputs = as_output_tuple(case.module(*positional_inputs))
        comparisons = compare_outputs(reference_outputs, candidate_outputs, atol=options.atol, rtol=options.rtol)
        result["stages"]["parity"] = [asdict(comparison) for comparison in comparisons]
        if not all(comparison.passed for comparison in comparisons):
            raise AssertionError("PyTorch reference/candidate parity failed.")

        onnx_path = run_dir / f"{benchmark.name}.onnx"
        torch.onnx.export(
            case.module,
            positional_inputs,
            str(onnx_path),
            input_names=case.input_names,
            output_names=list(case.output_names),
            dynamic_axes=dict(case.dynamic_axes),
            opset_version=17,
            dynamo=False,
        )
        onnx.checker.check_model(onnx.load(onnx_path, load_external_data=False))
        inputs_np = {name: value.detach().cpu().numpy() for name, value in case.inputs.items()}
        outputs_np = numpy_outputs(case.output_names, candidate_outputs)
        assets = emit_compile_assets(
            run_dir,
            onnx_path=onnx_path,
            inputs=inputs_np,
            output_names=case.output_names,
            specializations=case.specializations,
            custom_io=case.custom_io,
            device_ids=options.device_ids,
            hw_version=options.hw_version,
            num_cores=options.num_cores,
            extra_args=options.extra_compiler_args,
            compiler_lib_dir=options.compiler_lib_dir,
            enable_mxfp6=options.enable_mxfp6,
            allow_mxint8_mdp_io=options.allow_mxint8_mdp_io,
            retained_state=bool(case.retained_names),
        )
        io_json = dump_runner_io(run_dir / "io", inputs_np, outputs_np, case.retained_names)
        perf_scripts = emit_perf_scripts(
            run_dir,
            assets["compile_command"],
            io_json,
            iterations=options.perf_iterations,
            profile_start_iteration=options.profile_start_iteration,
            samples=options.profile_samples,
            stats_level=options.perf_stats_level,
            compiler_lib_dir=options.compiler_lib_dir,
        )
        result["artifacts"].update(
            {
                "onnx": str(onnx_path),
                "io_json": str(io_json),
                "perf_scripts": {name: str(path) for name, path in perf_scripts.items()},
                **assets,
            }
        )
        result["stages"]["prepare"] = {"status": "passed"}

        if options.stage in {"compile", "run"}:
            clean_qpc(Path(assets["qpc_dir"]))
            result["stages"]["compile"] = _execute(assets["compile_command"], env=_compiler_environment(options))

        if options.stage == "run":
            hardware_outputs, latency_ms = _hardware_run(
                Path(assets["qpc_dir"]), inputs_np, case.retained_names, options
            )
            # A newly activated retained-state QPC owns zero-initialized cache
            # buffers; retained inputs from the export fixture are not runtime
            # bindings. Model that state when checking its first/repeated step.
            hardware_reference_inputs = tuple(
                torch.zeros_like(value) if name in case.retained_names else value for name, value in case.inputs.items()
            )
            with torch.no_grad():
                hardware_reference_outputs = as_output_tuple(case.reference(*hardware_reference_inputs))
            visible_reference = {
                name: value
                for name, value in numpy_outputs(case.output_names, hardware_reference_outputs).items()
                if name not in case.retained_names
            }
            hardware_parity = {}
            for name, expected in visible_reference.items():
                if name not in hardware_outputs:
                    raise KeyError(f"Hardware output {name!r} was not returned by the QPC.")
                difference = np.abs(expected.astype(np.float32) - hardware_outputs[name].astype(np.float32))
                hardware_parity[name] = {
                    "passed": bool(np.allclose(expected, hardware_outputs[name], atol=options.atol, rtol=options.rtol)),
                    "max_abs_error": float(difference.max()) if difference.size else 0.0,
                }
            result["stages"]["run"] = {"latency_ms": latency_ms, "parity": hardware_parity}
            if not all(value["passed"] for value in hardware_parity.values()):
                raise AssertionError("Hardware/reference parity failed.")

        if options.stage == "profile":
            clean_qpc(Path(assets["qpc_dir"]))
            profile_results = {}
            for name in ("compile", "run", "decode"):
                profile_results[name] = _execute([str(perf_scripts[name])], env=_compiler_environment(options))
            result["stages"]["profile"] = profile_results

        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        result_path.write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
    return run_dir, result
