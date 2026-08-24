# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

STAGES = ("prepare", "compile", "run", "profile")


@dataclass(frozen=True)
class RuntimeOptions:
    stage: str
    artifact_root: Path
    artifact_dir: Optional[Path]
    seed: int
    atol: float
    rtol: float
    hw_version: str
    num_cores: int
    device_ids: tuple[int, ...]
    warmup: int
    iterations: int
    perf_iterations: int
    profile_start_iteration: int
    profile_samples: int
    perf_stats_level: int
    enable_mxfp6: bool
    allow_mxint8_mdp_io: bool
    extra_compiler_args: tuple[str, ...] = ()
    compiler_lib_dir: Optional[Path] = None


@dataclass
class BenchmarkCase:
    module: nn.Module
    reference: Callable[..., Any]
    inputs: "OrderedDict[str, torch.Tensor]"
    output_names: Sequence[str]
    dynamic_axes: Mapping[str, Mapping[int, str]]
    specializations: Mapping[str, int]
    custom_io: Mapping[str, str] = field(default_factory=dict)
    retained_names: set[str] = field(default_factory=set)

    @property
    def input_names(self) -> list[str]:
        return list(self.inputs)


@dataclass(frozen=True)
class Comparison:
    passed: bool
    max_abs_error: float
    mean_abs_error: float


class LayerBenchmark(ABC):
    name: str
    description: str

    @abstractmethod
    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add benchmark-specific command-line options."""

    @abstractmethod
    def resolved_config(self, args: argparse.Namespace) -> Mapping[str, Any]:
        """Return the validated, JSON-serializable benchmark configuration."""

    @abstractmethod
    def build_case(self, config: Mapping[str, Any], seed: int) -> BenchmarkCase:
        """Construct one deterministic candidate/reference benchmark case."""


def as_output_tuple(outputs: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    if isinstance(outputs, (list, tuple)) and all(isinstance(value, torch.Tensor) for value in outputs):
        return tuple(outputs)
    raise TypeError("Layer benchmarks must return a tensor or a sequence of tensors.")


def compare_outputs(
    expected: Sequence[torch.Tensor],
    actual: Sequence[torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> list[Comparison]:
    if len(expected) != len(actual):
        raise ValueError(f"Output count mismatch: reference={len(expected)}, candidate={len(actual)}.")
    comparisons = []
    for reference, candidate in zip(expected, actual):
        difference = (reference.float() - candidate.float()).abs()
        comparisons.append(
            Comparison(
                passed=bool(torch.allclose(reference, candidate, atol=atol, rtol=rtol)),
                max_abs_error=float(difference.max().item()) if difference.numel() else 0.0,
                mean_abs_error=float(difference.mean().item()) if difference.numel() else 0.0,
            )
        )
    return comparisons


def numpy_outputs(names: Sequence[str], outputs: Sequence[torch.Tensor]) -> dict[str, np.ndarray]:
    return {name: value.detach().cpu().numpy() for name, value in zip(names, outputs)}
