# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from QEfficient.generation.cloud_infer import QAICInferenceSession


def _artifact_feed(artifact_dir: Path, session: QAICInferenceSession) -> dict[str, np.ndarray]:
    io_manifest = json.loads((artifact_dir / "io" / "aic_batch_io.json").read_text())
    user_inputs = {
        entry["map-to"] for batch in io_manifest["IO-files"] for entry in batch if entry["io-direction"] == "in"
    }
    with np.load(artifact_dir / "inputs.npz") as values:
        return {
            name: values[name] for name in values.files if name in user_inputs and name in session.binding_index_map
        }


def compare_qpc_artifacts(
    baseline: Path,
    candidate: Path,
    *,
    device_ids: Sequence[int],
    warmup: int,
    iterations: int,
    samples: int,
) -> dict[str, Any]:
    """Measure two prepared artifact directories with one runtime methodology."""
    measurements: dict[str, list[float]] = {"baseline": [], "candidate": []}
    paths = {"baseline": baseline, "candidate": candidate}
    for _ in range(samples):
        for label in ("baseline", "candidate"):
            session = QAICInferenceSession(paths[label] / "qpc", device_ids=list(device_ids))
            feed = _artifact_feed(paths[label], session)
            for _ in range(warmup):
                session.run(feed)
            started = time.perf_counter()
            for _ in range(iterations):
                session.run(feed)
            measurements[label].append((time.perf_counter() - started) * 1000.0 / iterations)
            session.deactivate()
    baseline_ms = statistics.median(measurements["baseline"])
    candidate_ms = statistics.median(measurements["candidate"])
    return {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "device_ids": list(device_ids),
        "warmup": warmup,
        "iterations": iterations,
        "samples": samples,
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "delta_ms": candidate_ms - baseline_ms,
        "delta_percent": (candidate_ms / baseline_ms - 1.0) * 100.0,
        "measurements_ms": measurements,
    }
