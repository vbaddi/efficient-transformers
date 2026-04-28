#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

WEIGHT_SPEC_VERSION = 1


@dataclass
class TiedWeightAlias:
    alias: str
    canonical: str


@dataclass
class WeightSpecInput:
    name: str
    fqn: str
    kind: str
    shape: List[int]
    dtype: str


@dataclass
class WeightSpec:
    model_name: str
    model_id: str
    checkpoint_files: List[str]
    inputs: List[WeightSpecInput]
    tied_weights: List[TiedWeightAlias] = field(default_factory=list)
    version: int = WEIGHT_SPEC_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def save_weight_spec(path: Path, spec: WeightSpec) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(spec.to_dict(), handle, indent=2, sort_keys=True)
    return path


def load_weight_spec(path: Path) -> WeightSpec:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return WeightSpec(
        model_name=data["model_name"],
        model_id=data["model_id"],
        checkpoint_files=list(data["checkpoint_files"]),
        inputs=[WeightSpecInput(**entry) for entry in data["inputs"]],
        tied_weights=[TiedWeightAlias(**entry) for entry in data.get("tied_weights", [])],
        version=data.get("version", WEIGHT_SPEC_VERSION),
    )


def resolve_weight_spec_path(onnx_path: Path) -> Path:
    return onnx_path.with_name("weight_spec.json")
