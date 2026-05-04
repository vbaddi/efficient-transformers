#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

WEIGHT_SPEC_VERSION = 1


@dataclass
class TiedWeightAlias:
    alias: str
    canonical: str


@dataclass
class WeightSpecLocation:
    type: str
    file: str
    key: str


@dataclass
class WeightSpecInput:
    name: str
    fqn: str
    kind: str
    shape: List[int]
    dtype: str
    location: Optional[WeightSpecLocation] = None


@dataclass
class WeightSpec:
    model_name: str
    model_id: str
    checkpoint_files: List[str] = field(default_factory=list)
    checkpoint_base_dir: Optional[str] = None
    inputs: List[WeightSpecInput] = field(default_factory=list)
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
        checkpoint_base_dir=data.get("checkpoint_base_dir"),
        inputs=[
            WeightSpecInput(
                name=entry["name"],
                fqn=entry["fqn"],
                kind=entry["kind"],
                shape=list(entry["shape"]),
                dtype=entry["dtype"],
                location=(WeightSpecLocation(**entry["location"]) if entry.get("location") is not None else None),
            )
            for entry in data["inputs"]
        ],
        tied_weights=[TiedWeightAlias(**entry) for entry in data.get("tied_weights", [])],
        version=data.get("version", WEIGHT_SPEC_VERSION),
    )


def resolve_weight_spec_path(onnx_path: Path) -> Path:
    return onnx_path.with_name("weight_spec.json")
