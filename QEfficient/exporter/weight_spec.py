#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

WEIGHT_SPEC_VERSION = 3


@dataclass
class TiedWeightAlias:
    alias: str
    canonical: str


@dataclass
class CheckpointFile:
    path: str
    format: str


@dataclass
class WeightSpecLocation:
    file: Union[int, str]
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
    checkpoint_files: List[CheckpointFile] = field(default_factory=list)
    weights_root: Optional[str] = None
    inputs: List[WeightSpecInput] = field(default_factory=list)
    tied_weights: List[TiedWeightAlias] = field(default_factory=list)
    version: int = WEIGHT_SPEC_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def save_weight_spec(path: Path, spec: WeightSpec) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(spec.to_dict(), handle, indent=2, sort_keys=True)
    return path


def _load_checkpoint_files(raw: list) -> List[CheckpointFile]:
    if not raw:
        return []
    # Backward compat: old format stored plain strings
    if isinstance(raw[0], str):
        return [CheckpointFile(path=entry, format="safetensors") for entry in raw]
    return [CheckpointFile(**entry) for entry in raw]


def _load_location(raw: Optional[dict]) -> Optional[WeightSpecLocation]:
    if raw is None:
        return None
    # Backward compat: old format had a redundant "type" field on the location
    return WeightSpecLocation(file=raw["file"], key=raw["key"])


def load_weight_spec(path: Path) -> WeightSpec:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return WeightSpec(
        model_name=data["model_name"],
        model_id=data["model_id"],
        checkpoint_files=_load_checkpoint_files(data.get("checkpoint_files", [])),
        weights_root=data.get("weights_root", data.get("checkpoint_base_dir")),
        inputs=[
            WeightSpecInput(
                name=entry["name"],
                fqn=entry["fqn"],
                kind=entry["kind"],
                shape=list(entry["shape"]),
                dtype=entry["dtype"],
                location=_load_location(entry.get("location")),
            )
            for entry in data["inputs"]
        ],
        tied_weights=[TiedWeightAlias(**entry) for entry in data.get("tied_weights", [])],
        version=data.get("version", WEIGHT_SPEC_VERSION),
    )


def resolve_weight_spec_path(onnx_path: Path) -> Path:
    return onnx_path.with_name("weight_spec.json")
