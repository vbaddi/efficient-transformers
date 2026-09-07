# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""MXFP6 E2M3 checkpoint codec and manifest helpers.

The byte layout in this module is a QEfficient proposal for the compiler
handoff.  It is deliberately versioned and kept separate from the compiler's
existing runtime quantization flags.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import onnx_ir as ir
import torch

MXFP6_FORMAT = "mxfp6_e2m3"
MXFP6_LAYOUT = "inline_e8m0_fp6_lsb_v1"
MXFP6_MANIFEST_VERSION = 1
MXFP6_CONVERTER_VERSION = 1
MXFP6_BLOCK_SIZE = 32
MXFP6_PAYLOAD_BYTES = 24
MXFP6_STORAGE_BYTES = 25
MXFP6_DOMAIN = "com.qualcomm.qeff"
MXFP6_OP_TYPE = "MXDequantize"
MXFP6_OPSET = 1
MXFP6_OP_FORMAT = "MXFP6_E2M3"

_QWEN_PROJECTION_RE = re.compile(
    r"^(?P<prefix>(?:base_model\.)?model\.layers\.\d+\.)"
    r"(?P<projection>(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj)))\.weight$"
)


def _e2m3_value(exponent: int, mantissa: int) -> float:
    """Return the positive E2M3 value represented by exponent/mantissa."""
    if exponent:
        return math.ldexp(1.0 + math.ldexp(mantissa, -3), exponent - 1)
    return math.ldexp(mantissa, -3)


_E2M3_VALUES = tuple(_e2m3_value(exponent, mantissa) for exponent in range(4) for mantissa in range(8))
_E2M3_MANTISSAS = tuple(mantissa for _exponent in range(4) for mantissa in range(8))


def _scale_exp_scalar(block: Sequence[float]) -> int:
    max_abs = max(abs(float(value)) for value in block)
    if not math.isfinite(max_abs):
        raise ValueError("MXFP6 input contains NaN or Inf")
    _, exponent = math.frexp(max_abs)
    scale_exp = exponent - 2
    if not -127 <= scale_exp <= 127:
        raise ValueError(f"MXFP6 scale exponent {scale_exp} is not encodable as E8M0")
    return scale_exp


def quantize_block_scalar(block: Sequence[float]) -> tuple[list[int], int]:
    """Reference scalar oracle matching the supplied compiler E2M3 implementation.

    The return value contains unpacked six-bit codes (including the sign bit in
    bit five) and the signed E8M0 scale exponent.  This function intentionally
    does not call the production/vectorized encoder.
    """
    if len(block) != MXFP6_BLOCK_SIZE:
        raise ValueError(f"MXFP6 blocks must contain {MXFP6_BLOCK_SIZE} values, got {len(block)}")

    scale_exp = _scale_exp_scalar(block)
    codes: list[int] = []
    for value in block:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("MXFP6 input contains NaN or Inf")
        scaled = math.ldexp(value, -scale_exp)
        sign = 0x20 if math.copysign(1.0, scaled) < 0 else 0
        magnitude = abs(scaled)
        best_error = math.inf
        result = sign
        for code, representable in enumerate(_E2M3_VALUES):
            mantissa = _E2M3_MANTISSAS[code]
            error = abs(magnitude - representable)
            if error < best_error or (error == best_error and mantissa % 2 == 0):
                best_error = error
                result = sign | code
        codes.append(result)
    return codes, scale_exp


def _validate_scale_exponents(scale_exp: torch.Tensor) -> None:
    if bool(((scale_exp < -127) | (scale_exp > 127)).any()):
        bad = scale_exp[((scale_exp < -127) | (scale_exp > 127))][0].item()
        raise ValueError(f"MXFP6 scale exponent {bad} is not encodable as E8M0")


def _quantize_scaled_values(scaled: torch.Tensor) -> torch.Tensor:
    """Quantize bounded chunks of scaled magnitudes without a large distance matrix."""
    values = torch.tensor(_E2M3_VALUES, dtype=torch.float32, device=scaled.device)
    best_error = torch.full_like(scaled, float("inf"))
    best_code = torch.zeros_like(scaled, dtype=torch.uint8)
    for code in range(len(_E2M3_VALUES)):
        error = (scaled.abs() - values[code]).abs()
        even_mantissa = int(_E2M3_MANTISSAS[code]) % 2 == 0
        update = error < best_error
        if even_mantissa:
            update |= error == best_error
        best_error = torch.where(update, error, best_error)
        best_code = torch.where(update, torch.tensor(code, dtype=torch.uint8, device=scaled.device), best_code)
    return best_code


def quantize_tensor_vectorized(
    tensor: torch.Tensor,
    *,
    row_chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a two-dimensional tensor in bounded row chunks.

    Returns ``(codes, scale_exp)`` where codes has shape ``[O, ceil(K/32), 32]``
    and scale_exp has shape ``[O, ceil(K/32)]``.  The padded payload is always
    positive zero, while signed zeros in the source are retained.
    """
    if tensor.ndim != 2:
        raise ValueError(f"MXFP6 expects a rank-2 projection weight, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"MXFP6 expects a floating-point tensor, got {tensor.dtype}")
    rows, width = tensor.shape
    block_count = (width + MXFP6_BLOCK_SIZE - 1) // MXFP6_BLOCK_SIZE
    padded_width = block_count * MXFP6_BLOCK_SIZE
    codes = torch.empty((rows, block_count, MXFP6_BLOCK_SIZE), dtype=torch.uint8)
    scales = torch.empty((rows, block_count), dtype=torch.int16)

    for start in range(0, rows, row_chunk_size):
        chunk = tensor[start : start + row_chunk_size].to(torch.float32)
        if padded_width != width:
            padded = torch.zeros((chunk.shape[0], padded_width), dtype=torch.float32)
            padded[:, :width] = chunk
            chunk = padded
        blocks = chunk.reshape(chunk.shape[0], block_count, MXFP6_BLOCK_SIZE)
        if not bool(torch.isfinite(blocks).all()):
            raise ValueError(f"MXFP6 input contains NaN or Inf in rows [{start}, {start + chunk.shape[0]})")
        max_abs = blocks.abs().amax(dim=-1)
        _, exponent = torch.frexp(max_abs)
        scale_exp = exponent.to(torch.int16) - 2
        _validate_scale_exponents(scale_exp)
        scaled = torch.ldexp(blocks, -scale_exp.to(torch.int32).unsqueeze(-1))
        chunk_codes = _quantize_scaled_values(scaled)
        chunk_codes |= torch.signbit(blocks).to(torch.uint8) * 0x20
        codes[start : start + chunk.shape[0]] = chunk_codes
        scales[start : start + chunk.shape[0]] = scale_exp
    return codes, scales


def pack_codes(codes: torch.Tensor, scale_exp: torch.Tensor) -> torch.Tensor:
    """Pack six-bit codes and signed scale exponents into flattened UINT8 rows."""
    if codes.ndim != 3 or tuple(codes.shape[-1:]) != (MXFP6_BLOCK_SIZE,):
        raise ValueError(f"Expected codes [O, blocks, 32], got {tuple(codes.shape)}")
    if tuple(scale_exp.shape) != tuple(codes.shape[:-1]):
        raise ValueError("Scale shape must match the first two code dimensions")
    if bool((codes > 0x3F).any()):
        raise ValueError("MXFP6 codes must fit in six bits")
    _validate_scale_exponents(scale_exp)
    grouped = (codes & 0x3F).reshape(*codes.shape[:-1], 8, 4)
    c0, c1, c2, c3 = (grouped[..., i] for i in range(4))
    payload = torch.stack(
        [
            c0 | (c1 << 6),
            (c1 >> 2) | (c2 << 4),
            (c2 >> 4) | (c3 << 2),
        ],
        dim=-1,
    ).reshape(*codes.shape[:-1], MXFP6_PAYLOAD_BYTES)
    scale_bytes = (scale_exp.to(torch.int16) + 127).to(torch.uint8).unsqueeze(-1)
    return torch.cat((scale_bytes, payload), dim=-1).reshape(codes.shape[0], -1)


def unpack_codes(packed: torch.Tensor, logical_width: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`pack_codes`, returning six-bit codes and signed scales."""
    if packed.ndim != 2 or packed.shape[-1] % MXFP6_STORAGE_BYTES:
        raise ValueError(f"Expected packed UINT8 [O, 25*blocks], got {tuple(packed.shape)}")
    if packed.dtype != torch.uint8:
        raise TypeError(f"Expected packed UINT8, got {packed.dtype}")
    block_count = packed.shape[-1] // MXFP6_STORAGE_BYTES
    rows = packed.shape[0]
    blocks = packed.reshape(rows, block_count, MXFP6_STORAGE_BYTES)
    scale_exp = blocks[..., 0].to(torch.int16) - 127
    payload = blocks[..., 1:].reshape(rows, block_count, 8, 3)
    b0, b1, b2 = (payload[..., i].to(torch.int64) for i in range(3))
    codes = torch.stack(
        [
            b0 & 0x3F,
            ((b0 >> 6) | ((b1 & 0x0F) << 2)) & 0x3F,
            ((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3F,
            (b2 >> 2) & 0x3F,
        ],
        dim=-1,
    ).to(torch.uint8)
    codes = codes.reshape(rows, block_count, MXFP6_BLOCK_SIZE)
    if logical_width is not None:
        codes = codes[..., :logical_width]
    return codes, scale_exp


def dequantize_packed_tensor(
    packed: torch.Tensor,
    logical_shape: Sequence[int],
    *,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode flattened MXFP6 storage to the original logical shape."""
    codes, scale_exp = unpack_codes(packed, logical_width=int(logical_shape[1]))
    code_values = torch.tensor(_E2M3_VALUES, dtype=torch.float32, device=packed.device)
    magnitudes = code_values[(codes & 0x1F).to(torch.int64).reshape(-1)].reshape(codes.shape)
    values = torch.where((codes & 0x20) != 0, -magnitudes, magnitudes)
    values = torch.ldexp(values, scale_exp.to(torch.int32).unsqueeze(-1))
    values = values.reshape(values.shape[0], -1)[..., : int(logical_shape[1])]
    return values.reshape(tuple(logical_shape)).to(output_dtype)


def encode_tensor(tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """Encode a projection tensor into its flattened UINT8 storage shape."""
    codes, scales = quantize_tensor_vectorized(tensor, **kwargs)
    return pack_codes(codes, scales)


def selected_qwen_projection_keys(weight_map: Mapping[str, str], model_config: Any = None) -> list[str]:
    """Select only the seven dense Qwen decoder projection weights.

    Selection is based on the Qwen module naming contract and the config model
    type, never on tensor rank alone.
    """
    model_type = getattr(model_config, "model_type", None) if model_config is not None else None
    if model_type is None and isinstance(model_config, Mapping):
        model_type = model_config.get("model_type")
    normalized_model_type = str(model_type).lower() if model_type is not None else ""
    if model_type is not None and (
        not (normalized_model_type.startswith("qwen2") or normalized_model_type.startswith("qwen3"))
        or "moe" in normalized_model_type
    ):
        raise ValueError(f"MXFP6 Qwen conversion requires a Qwen dense config, got model_type={model_type!r}")
    selected = sorted(key for key in weight_map if _QWEN_PROJECTION_RE.fullmatch(key))
    if not selected:
        raise ValueError("No Qwen dense decoder projection weights matched the MXFP6 selection contract")
    expected_layer_count = getattr(model_config, "num_hidden_layers", None) if model_config is not None else None
    if expected_layer_count is None and isinstance(model_config, Mapping):
        expected_layer_count = model_config.get("num_hidden_layers")
    if expected_layer_count is not None:
        expected_layers = int(expected_layer_count)
        selected_layers = {
            int(_QWEN_PROJECTION_RE.fullmatch(key).group(0).split(".layers.")[1].split(".")[0]) for key in selected
        }
        if selected_layers != set(range(expected_layers)):
            raise ValueError(
                f"MXFP6 selection found layers {sorted(selected_layers)}, expected all {expected_layers} decoder layers"
            )
    return selected


def logical_dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.float16: "FLOAT16",
        torch.bfloat16: "BFLOAT16",
        torch.float32: "FLOAT",
    }.get(dtype, str(dtype).replace("torch.", "").upper())


def dtype_from_name(name: str) -> torch.dtype:
    values = {"FLOAT16": torch.float16, "BFLOAT16": torch.bfloat16, "FLOAT": torch.float32, "FLOAT32": torch.float32}
    try:
        return values[name.upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported MXFP6 graph dtype: {name!r}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mxfp6_manifest(directory: Path) -> Optional[dict[str, Any]]:
    path = Path(directory) / "qeff_mx_manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid MXFP6 manifest: {path}") from exc
    validate_mxfp6_manifest(manifest)
    return manifest


def validate_mxfp6_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the authoritative checkpoint-side MX manifest."""
    required = {"version", "format", "layout", "block_size", "axis", "tensors", "complete"}
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"MXFP6 manifest is missing required fields: {missing}")
    if manifest["version"] != MXFP6_MANIFEST_VERSION:
        raise ValueError(f"Unsupported MXFP6 manifest version: {manifest['version']}")
    if manifest["format"] != MXFP6_FORMAT or manifest["layout"] != MXFP6_LAYOUT:
        raise ValueError("Unknown MXFP6 format or storage layout; refusing to reinterpret packed bytes")
    if manifest["block_size"] != MXFP6_BLOCK_SIZE or manifest["axis"] != -1:
        raise ValueError("MXFP6 manifest must use block_size=32 and axis=-1")
    if not manifest["complete"]:
        raise ValueError("MXFP6 checkpoint manifest is not complete")
    for key, entry in manifest["tensors"].items():
        if entry.get("format") != MXFP6_FORMAT or entry.get("layout") != MXFP6_LAYOUT:
            raise ValueError(f"Unknown quantized schema for tensor {key!r}")
        if entry.get("physical_dtype") != "UINT8":
            raise ValueError(f"MXFP6 tensor {key!r} must have UINT8 physical storage")


def manifest_tensor_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    validate_mxfp6_manifest(manifest)
    return dict(manifest["tensors"])


def manifest_identity(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_mxfp6_checkpoint(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    target_dtype: torch.dtype = torch.bfloat16,
    row_chunk_size: int = 128,
) -> Path:
    """Prepare a dense Qwen safetensors checkpoint for weight-free MXFP6 export.

    This is intentionally a separate offline operation.  Exporting the same
    prepared directory never requantizes its tensors.
    """
    source = Path(source_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"MXFP6 source checkpoint directory does not exist: {source}")
    if source == output:
        raise ValueError("MXFP6 source and destination directories must differ")
    if output.exists() and any(output.iterdir()):
        existing = load_mxfp6_manifest(output)
        if existing is None:
            raise FileExistsError(
                f"Refusing to overwrite non-empty destination {output}; choose a new output directory"
            )
        expected = {
            "format": MXFP6_FORMAT,
            "layout": MXFP6_LAYOUT,
            "block_size": MXFP6_BLOCK_SIZE,
            "axis": -1,
            "target_dtype": logical_dtype_name(target_dtype),
        }
        if any(existing.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing MXFP6 checkpoint manifest does not match the requested conversion settings")

    from QEfficient.base.checkpoint_transforms import CheckpointTransformPipeline
    from QEfficient.exporter.weight_free.checkpoint_transforms import Mxfp6CheckpointTransform
    from QEfficient.utils.checkpoint_utils import read_weight_map

    config: Mapping[str, Any] = {}
    config_path = source / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    weight_map = read_weight_map(source)
    selected = selected_qwen_projection_keys(weight_map, config)
    pipeline = CheckpointTransformPipeline([Mxfp6CheckpointTransform])
    pipeline.apply(
        source,
        output,
        target_dtype=target_dtype,
        model_config=config,
        selected_keys=selected,
        row_chunk_size=row_chunk_size,
    )
    manifest = load_mxfp6_manifest(output)
    if manifest is None:
        raise RuntimeError(f"MXFP6 preparation completed without a manifest: {output}")
    return output


def mxfp6_prepared_cache_dir(source_dir: str | Path, target_dtype: torch.dtype) -> Path:
    """Return an input-keyed prepared-checkpoint location for convenience APIs."""
    source = Path(source_dir).expanduser().resolve()
    config_path = source / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    from QEfficient.utils.checkpoint_utils import read_weight_map

    weight_map = read_weight_map(source)
    selected = selected_qwen_projection_keys(weight_map, config)
    source_identity = [
        (path.name, path.stat().st_size, sha256_file(path))
        for path in sorted({source / shard for shard in weight_map.values()})
    ]
    cache_identity = {
        "source": source_identity,
        "selected_keys": selected,
        "target_dtype": logical_dtype_name(target_dtype),
        "format": MXFP6_FORMAT,
        "layout": MXFP6_LAYOUT,
        "block_size": MXFP6_BLOCK_SIZE,
        "axis": -1,
        "converter_version": MXFP6_CONVERTER_VERSION,
    }
    digest = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return source.parent / f"{source.name}-qeff-mxfp6-{logical_dtype_name(target_dtype).lower()}-{digest}"


def rewrite_mxfp6_weight_inputs(onnx_program, weight_spec: Any, manifest: Mapping[str, Any]) -> int:
    """Rewrite promoted MXFP6 inputs to ``com.qualcomm.qeff::MXDequantize`` nodes."""
    tensor_map = manifest_tensor_map(manifest)
    graph = onnx_program.model.graph
    graph.opset_imports[MXFP6_DOMAIN] = MXFP6_OPSET
    if hasattr(onnx_program.model, "opset_imports"):
        onnx_program.model.opset_imports[MXFP6_DOMAIN] = MXFP6_OPSET

    rewritten = 0
    for spec_input in weight_spec.inputs:
        quantization = spec_input.quantization
        if quantization is None:
            continue
        key = spec_input.location.key
        entry = tensor_map.get(key)
        if entry is None:
            raise ValueError(f"Weight spec marks {key!r} as quantized but the MX manifest has no such tensor")
        graph_input = next((value for value in graph.inputs if value.name == spec_input.name), None)
        if graph_input is None:
            raise ValueError(f"MXFP6 graph input {spec_input.name!r} is missing after initializer promotion")

        consumers = []
        logical_type = None
        logical_shape = None
        graph_nodes = list(graph.all_nodes())
        for node in graph_nodes:
            for index, value in enumerate(node.inputs):
                if value is not None and value.name == spec_input.name:
                    consumers.append((node, index))
                    if logical_type is None:
                        logical_type = value.type
                        logical_shape = value.shape
        if not consumers or logical_type is None or logical_shape is None:
            raise ValueError(f"MXFP6 promoted input {spec_input.name!r} has no logical graph consumers")
        logical_dims = tuple(int(dim) for dim in logical_shape)
        expected_shape = tuple(entry["logical_shape"])
        if logical_dims != expected_shape:
            raise ValueError(
                f"MXFP6 logical shape mismatch for {key!r}: graph={logical_dims}, manifest={expected_shape}"
            )
        expected_dtype = {
            "FLOAT": ir.DataType.FLOAT,
            "FLOAT16": ir.DataType.FLOAT16,
            "BFLOAT16": ir.DataType.BFLOAT16,
        }.get(entry["logical_dtype"])
        if expected_dtype is None or logical_type.dtype != expected_dtype:
            raise ValueError(
                f"MXFP6 logical dtype mismatch for {key!r}: graph={logical_type.dtype}, "
                f"manifest={entry['logical_dtype']}"
            )

        physical_shape = tuple(entry["physical_shape"])
        graph_input.shape = ir.Shape(physical_shape)
        graph_input.type = ir.TensorType(ir.DataType.UINT8)
        output = ir.Value(name=f"{spec_input.name}__dequantized", shape=ir.Shape(logical_dims), type=logical_type)
        node = ir.Node(
            MXFP6_DOMAIN,
            MXFP6_OP_TYPE,
            inputs=[graph_input],
            attributes=[
                ir.AttrString("format", MXFP6_OP_FORMAT),
                ir.AttrInt64("block_size", MXFP6_BLOCK_SIZE),
                ir.AttrInt64("axis", -1),
                ir.AttrInt64("axis_size", int(entry["axis_length"])),
                ir.AttrString("output_dtype", entry["logical_dtype"]),
            ],
            outputs=[output],
            name=f"{spec_input.name}__mxfp6_dequantize",
        )
        graph.insert_before(consumers[0][0], node)
        for consumer, index in consumers:
            consumer.replace_input_with(index, output)
        rewritten += 1

    graph.sort()
    return rewritten
