# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

WEIGHT_SPEC_VERSION = 5
QUANTIZED_WEIGHT_SPEC_VERSION = 6


@dataclass
class TiedWeightAlias:
    """Alias entry for checkpoint weights shared by multiple model parameters."""

    alias: str
    canonical: str


@dataclass
class ExternalDataFile:
    """External checkpoint file referenced by a weight spec."""

    path: str
    format: str


CheckpointFile = ExternalDataFile


@dataclass
class WeightSpecLocation:
    """Location of one external weight tensor inside a checkpoint file."""

    file: Union[int, str]
    key: str


@dataclass
class WeightSpecInput:
    """Mapping from an ONNX input name to its external checkpoint tensor."""

    name: str
    location: WeightSpecLocation  # required: every spec entry must point to a file
    quantization: Optional[Dict[str, Any]] = None


@dataclass
class WeightSpec:
    """Serializable weight-free export metadata.

    The spec records the checkpoint files shipped beside the ONNX model and
    maps promoted ONNX weight inputs back to tensor keys in those files.
    """

    model_name: str
    model_id: str
    files: List[ExternalDataFile] = field(default_factory=list)
    inputs: List[WeightSpecInput] = field(default_factory=list)
    version: int = WEIGHT_SPEC_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the weight spec."""
        data = asdict(self)
        data["model_id"] = str(data["model_id"])
        for input_data in data["inputs"]:
            if input_data.get("quantization") is None:
                input_data.pop("quantization", None)
        if any(item.quantization is not None for item in self.inputs):
            if self.version < QUANTIZED_WEIGHT_SPEC_VERSION:
                data["version"] = QUANTIZED_WEIGHT_SPEC_VERSION
        return data


def save_weight_spec(path: Path, spec: WeightSpec) -> Path:
    """Write a weight spec JSON file.

    Parameters
    ----------
    path : Path
        Destination path for the JSON file.
    spec : WeightSpec
        Weight spec instance to serialize.

    Returns
    -------
    Path
        The destination path.
    """
    with path.open("w", encoding="utf-8") as handle:
        json.dump(spec.to_dict(), handle, indent=2, sort_keys=True)
    return path


def _load_files(raw: list) -> List[ExternalDataFile]:
    """Deserialize external file entries, including the legacy string format."""
    if not raw:
        return []
    # Backward compat: old format stored plain strings
    if isinstance(raw[0], str):
        return [ExternalDataFile(path=entry, format="safetensors") for entry in raw]
    return [ExternalDataFile(**entry) for entry in raw]


def _load_location(raw: dict) -> WeightSpecLocation:
    """Deserialize a weight location entry from the spec JSON payload."""
    # Backward compat: old format had a redundant "type" field on the location
    return WeightSpecLocation(file=raw["file"], key=raw["key"])


def load_weight_spec(path: Path) -> WeightSpec:
    """Load a weight spec JSON file.

    Parameters
    ----------
    path : Path
        Path to ``weight_spec.json``.

    Returns
    -------
    WeightSpec
        Parsed weight spec with backward-compatible fields normalized.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    version = data.get("version", WEIGHT_SPEC_VERSION)
    if version not in {WEIGHT_SPEC_VERSION, QUANTIZED_WEIGHT_SPEC_VERSION}:
        raise ValueError(f"Unsupported weight spec version: {version}")
    inputs = []
    for entry in data["inputs"]:
        if entry.get("location") is None:  # backward compat: skip old buffer-only entries
            continue
        quantization = entry.get("quantization")
        if quantization is not None:
            required = {
                "format",
                "block_size",
                "axis",
                "layout",
                "dequantized_axis_size",
                "logical_dtype",
            }
            missing = sorted(required - set(quantization))
            if missing:
                raise ValueError(f"Quantized weight spec entry {entry['name']!r} is missing {missing}")
            if quantization["format"] != "mxfp6_e2m3" or quantization["layout"] != "inline_e8m0_fp6_lsb_v1":
                raise ValueError(f"Unknown quantized weight spec schema for {entry['name']!r}")
            if version < QUANTIZED_WEIGHT_SPEC_VERSION:
                raise ValueError("Quantized weight metadata requires weight spec version 6")
        inputs.append(
            WeightSpecInput(
                name=entry["name"],
                location=_load_location(entry["location"]),
                quantization=quantization,
            )
        )

    return WeightSpec(
        model_name=data["model_name"],
        model_id=data["model_id"],
        files=_load_files(data.get("files", data.get("checkpoint_files", []))),
        inputs=inputs,
        version=version,
    )


def resolve_weight_spec_path(onnx_path: Path) -> Path:
    """Return the sidecar weight spec path for an ONNX export path."""
    return onnx_path.with_name("weight_spec.json")
