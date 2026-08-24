# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

from QEfficient.utils.layer_benchmarks.contracts import LayerBenchmark


def built_in_benchmarks() -> dict[str, LayerBenchmark]:
    from QEfficient.utils.layer_benchmarks.gqa import GQABenchmark
    from QEfficient.utils.layer_benchmarks.mla import MLABenchmark
    from QEfficient.utils.layer_benchmarks.moe import MoEBenchmark

    benchmarks: list[LayerBenchmark] = [GQABenchmark(), MLABenchmark(), MoEBenchmark()]
    registry: dict[str, LayerBenchmark] = {}
    for benchmark in benchmarks:
        if benchmark.name in registry:
            raise ValueError(f"Duplicate layer benchmark name: {benchmark.name!r}.")
        registry[benchmark.name] = benchmark
    return registry
