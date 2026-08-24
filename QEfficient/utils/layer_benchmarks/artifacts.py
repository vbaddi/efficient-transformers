# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from QEfficient.compile.mdp_generator import generate_mdp_partition_config
from QEfficient.utils import constants


def parse_device_ids(value: str) -> tuple[int, ...]:
    text = value.strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError(f"device group must look like [0] or [0,1], got {value!r}.")
    inner = text[1:-1].strip()
    device_ids = tuple(int(part.strip()) for part in inner.split(",") if part.strip())
    if not device_ids:
        raise ValueError("device group cannot be empty.")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError(f"device group contains duplicate IDs: {value!r}.")
    return device_ids


def write_specializations(path: Path, values: Mapping[str, int]) -> Path:
    payload = {"specializations": [{key: str(value) for key, value in values.items()}]}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def write_custom_io(path: Path, mapping: Mapping[str, str]) -> Path:
    content = "".join(f" - IOName: {name}\n   Precision: {precision}\n\n" for name, precision in mapping.items())
    path.write_text(content)
    return path


def build_compile_command(
    *,
    onnx_path: Path,
    qpc_dir: Path,
    specializations_path: Path,
    custom_io_path: Path | None,
    mdp_path: Path | None,
    hw_version: str,
    num_cores: int,
    extra_args: Sequence[str],
    enable_mxfp6: bool,
    allow_mxint8_mdp_io: bool,
    retained_state: bool,
) -> list[str]:
    command = [
        constants.COMPILER[0],
        "-aic-hw",
        f"-aic-hw-version={hw_version}",
        f"-m={onnx_path}",
        "-convert-to-fp16",
        f"-aic-num-cores={num_cores}",
        f"-network-specialization-config={specializations_path}",
        f"-aic-binary-dir={qpc_dir}",
        "-user-tiled",
    ]
    if retained_state:
        command.insert(4, "-retained-state")
    if enable_mxfp6:
        command.append("-mxfp6-matmul")
    if allow_mxint8_mdp_io:
        command.append("-allow-mxint8-mdp-io")
    if custom_io_path is not None:
        command.append(f"-custom-IO-list-file={custom_io_path}")
    if mdp_path is not None:
        command.append(f"-mdp-load-partition-config={mdp_path}")
    command.extend(extra_args)
    return command


def write_script(path: Path, command: Sequence[str], compiler_lib_dir: Path | None = None) -> Path:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    if compiler_lib_dir is not None:
        lines.extend(
            [
                f"export AIC_COMPILER_LIB_DIR={shlex.quote(str(compiler_lib_dir))}",
                f"export QAIC_COMPILER_LIB={shlex.quote(str(compiler_lib_dir / 'libQAicCompiler.so'))}",
            ]
        )
    lines.append(shlex.join(str(part) for part in command))
    path.write_text("\n".join(lines) + "\n")
    os.chmod(path, 0o755)
    return path


def dump_runner_io(
    io_dir: Path,
    inputs: Mapping[str, np.ndarray],
    outputs: Mapping[str, np.ndarray],
    retained_names: set[str],
) -> Path:
    data_dir = io_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for direction, values in (("in", inputs), ("out", outputs)):
        for name, value in values.items():
            if name in retained_names:
                continue
            path = data_dir / f"{name}.raw"
            value.tofile(path)
            entries.append(
                {
                    "path": f"data/{name}.raw",
                    "io-direction": direction,
                    "elem-size": value.itemsize,
                    "map-to": name,
                    "dims": list(value.shape),
                }
            )
    json_path = io_dir / "aic_batch_io.json"
    json_path.write_text(json.dumps({"IO-files": [entries]}, indent=2) + "\n")
    return json_path


def emit_compile_assets(
    run_dir: Path,
    *,
    onnx_path: Path,
    inputs: Mapping[str, np.ndarray],
    output_names: Sequence[str],
    specializations: Mapping[str, int],
    custom_io: Mapping[str, str],
    device_ids: Sequence[int],
    hw_version: str,
    num_cores: int,
    extra_args: Sequence[str],
    compiler_lib_dir: Path | None,
    enable_mxfp6: bool,
    allow_mxint8_mdp_io: bool,
    retained_state: bool,
) -> dict[str, object]:
    np.savez(run_dir / "inputs.npz", **inputs)
    input_metadata = {
        "input_names": list(inputs),
        "output_names": list(output_names),
        "shapes": {name: list(value.shape) for name, value in inputs.items()},
        "dtypes": {name: str(value.dtype) for name, value in inputs.items()},
    }
    (run_dir / "inputs.json").write_text(json.dumps(input_metadata, indent=2) + "\n")
    specializations_path = write_specializations(run_dir / "specializations.json", specializations)
    custom_io_path = write_custom_io(run_dir / "custom_io.yaml", custom_io) if custom_io else None
    mdp_path = None
    if len(device_ids) > 1:
        mdp_path = run_dir / "mdp_ts_config.json"
        mdp_path.write_text(json.dumps(generate_mdp_partition_config(len(device_ids), num_cores), indent=2) + "\n")
    qpc_dir = run_dir / "qpc"
    command = build_compile_command(
        onnx_path=onnx_path,
        qpc_dir=qpc_dir,
        specializations_path=specializations_path,
        custom_io_path=custom_io_path,
        mdp_path=mdp_path,
        hw_version=hw_version,
        num_cores=num_cores,
        extra_args=extra_args,
        enable_mxfp6=enable_mxfp6,
        allow_mxint8_mdp_io=allow_mxint8_mdp_io,
        retained_state=retained_state,
    )
    script = write_script(run_dir / "compile.sh", command, compiler_lib_dir)
    return {
        "compile_command": command,
        "compile_script": str(script),
        "qpc_dir": str(qpc_dir),
        "specializations": str(specializations_path),
        "custom_io": str(custom_io_path) if custom_io_path else None,
        "mdp": str(mdp_path) if mdp_path else None,
    }


def emit_perf_scripts(
    run_dir: Path,
    compile_command: Sequence[str],
    io_json: Path,
    *,
    iterations: int,
    profile_start_iteration: int,
    samples: int,
    stats_level: int,
    compiler_lib_dir: Path | None,
) -> dict[str, Path]:
    perf_dir = run_dir / "perf_dump"
    stats_dir = perf_dir / "raw_device_stats"
    opstats_dir = perf_dir / "opstats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    opstats_dir.mkdir(parents=True, exist_ok=True)
    qpc_dir = run_dir / "qpc"
    profiled_compile = list(compile_command) + [
        f"-stats-level={stats_level}",
        "-ddr-stats",
        "-aic-pmu-recipe=KernelUtil",
    ]
    runner_command = [
        "/opt/qti-aic/exec/qaic-runner",
        "-t",
        str(qpc_dir),
        "-n",
        str(iterations),
        "--aic-profiling-type",
        "raw_device_stats",
        "--aic-profiling-start-iter",
        str(profile_start_iteration),
        "--aic-profiling-num-samples",
        str(samples),
        "--aic-profiling-out-dir",
        str(stats_dir),
        "--aic-batch-json-input",
        str(io_json),
    ]
    opstats_command = [
        "/opt/qti-aic/exec/qaic-opstats",
        "--qpc",
        str(qpc_dir / "programqpc.bin"),
        "--input-dir",
        str(stats_dir),
        "--output-dir",
        str(opstats_dir),
        "--summary",
        "--trace",
    ]
    scripts = {
        "compile": write_script(perf_dir / "compile_perf.sh", profiled_compile, compiler_lib_dir),
        "run": write_script(perf_dir / "run_perf.sh", runner_command),
        "decode": write_script(perf_dir / "decode_perf.sh", opstats_command),
    }
    return scripts


def clean_qpc(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
