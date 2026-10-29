# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Opt-in timing for the weight-free export pipeline.

Set ``QEFF_EXPORT_PROFILE=1`` to print and save stage timings.  Set it to
``cprofile`` to also write a Python profile for the Dynamo export call.
Profiling is disabled by default and has no effect on the exported graph.
"""

from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_PROFILE_ENV = "QEFF_EXPORT_PROFILE"
_PREFIX = "[export-profile]"


def enabled() -> bool:
    return os.environ.get(_PROFILE_ENV, "").strip().lower() not in {"", "0", "false", "off", "no"}


def cprofile_enabled() -> bool:
    return os.environ.get(_PROFILE_ENV, "").strip().lower() == "cprofile"


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


class ExportProfiler:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._depth = 0

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not enabled():
            yield
            return
        depth = self._depth
        self._depth += 1
        print(f"{_PREFIX} {'  ' * depth}> {name}", flush=True)
        start = time.perf_counter()
        ok = False
        try:
            yield
            ok = True
        finally:
            elapsed = time.perf_counter() - start
            self._depth -= 1
            record = {
                "stage": name,
                "seconds": round(elapsed, 3),
                "peak_rss_mb": round(_peak_rss_mb(), 1),
                "ok": ok,
            }
            self.records.append(record)
            status = "" if ok else " FAILED"
            print(
                f"{_PREFIX} {'  ' * depth}< {name}: {elapsed:.2f}s (peak RSS {record['peak_rss_mb']:.0f} MB){status}",
                flush=True,
            )

    def save(self, path: Path) -> None:
        if not enabled():
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"stages": self.records}, indent=2))
        print(f"{_PREFIX} profile written to {path}", flush=True)


PROFILER = ExportProfiler()


@contextmanager
def maybe_cprofile(output_dir: Path) -> Iterator[None]:
    if not cprofile_enabled():
        yield
        return
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(output_dir / "export_cprofile.prof"))
        report = io.StringIO()
        pstats.Stats(profiler, stream=report).sort_stats("cumulative").print_stats(60)
        (output_dir / "export_cprofile_top.txt").write_text(report.getvalue())
