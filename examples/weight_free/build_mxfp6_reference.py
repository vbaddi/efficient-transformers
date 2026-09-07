"""Build the small, seeded Qwen MXFP6 compiler-reference bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import onnx
import torch
from safetensors import safe_open
from transformers import Qwen2Config, Qwen2ForCausalLM

from QEfficient.exporter.weight_free import prepare_mxfp6_checkpoint
from QEfficient.exporter.weight_free.mxfp6 import dequantize_packed_tensor
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    error = (actual.float() - expected.float()).abs()
    return {"max_abs": float(error.max()), "rmse": float(torch.sqrt((error * error).mean()))}


def _flatten_cache(cache) -> list[torch.Tensor]:
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    if hasattr(cache, "layers"):
        tensors = []
        for layer in cache.layers:
            tensors.extend(
                value for value in (getattr(layer, "keys", None), getattr(layer, "values", None)) if value is not None
            )
        return tensors
    if isinstance(cache, (tuple, list)):
        tensors = []
        for value in cache:
            tensors.extend(_flatten_cache(value))
        return tensors
    return [cache] if isinstance(cache, torch.Tensor) else []


def _rewrite_for_cpu_reference(model_path: Path, manifest: dict, output_path: Path) -> int:
    """Replace MXDequantize with logical float inputs in a validation-only graph."""
    model = onnx.load(model_path, load_external_data=False)
    logical_dtype = {
        "BFLOAT16": onnx.TensorProto.BFLOAT16,
        "FLOAT16": onnx.TensorProto.FLOAT16,
        "FLOAT": onnx.TensorProto.FLOAT,
    }
    replaced = 0
    existing_inputs = {value.name for value in model.graph.input}
    for node in model.graph.node:
        if node.domain != "com.qualcomm.qeff" or node.op_type != "MXDequantize":
            continue
        packed_name = node.input[0]
        entry = manifest["tensors"][packed_name]
        logical_name = f"{packed_name}__validation_logical"
        if logical_name not in existing_inputs:
            model.graph.input.append(
                onnx.helper.make_tensor_value_info(
                    logical_name,
                    logical_dtype[entry["logical_dtype"]],
                    entry["logical_shape"],
                )
            )
            existing_inputs.add(logical_name)
        node.input[0] = logical_name
        node.domain = ""
        node.op_type = "Identity"
        node.ClearField("attribute")
        replaced += 1
    opset_imports = [opset for opset in model.opset_import if opset.domain != "com.qualcomm.qeff"]
    del model.opset_import[:]
    model.opset_import.extend(opset_imports)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return replaced


def build_bundle(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty reference bundle: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Qwen2Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    torch.manual_seed(7)
    original_model = Qwen2ForCausalLM(config).eval().to(torch.bfloat16)

    with tempfile.TemporaryDirectory(prefix="qeff-mxfp6-reference-") as temporary:
        temporary_dir = Path(temporary)
        source_dir = temporary_dir / "source"
        prepared_dir = temporary_dir / "prepared"
        export_dir = temporary_dir / "export"
        original_model.save_pretrained(source_dir, safe_serialization=True)
        prepare_mxfp6_checkpoint(source_dir, prepared_dir, target_dtype=torch.bfloat16)

        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(str(prepared_dir), weight_free=True)
        onnx_path = Path(
            qeff_model.export(
                str(export_dir),
                use_onnx_subfunctions=True,
                offload_pt_weights=False,
            )
        )
        spec_path = onnx_path.with_name("weight_spec.json")
        exported_checkpoint = onnx_path.parent / "checkpoint"
        shutil.copy2(onnx_path, output_dir / "model.onnx")
        shutil.copy2(spec_path, output_dir / "weight_spec.json")
        shutil.copytree(exported_checkpoint, output_dir / "checkpoint")

        manifest = json.loads((output_dir / "checkpoint/qeff_mx_manifest.json").read_text(encoding="utf-8"))
        spec = json.loads((output_dir / "weight_spec.json").read_text(encoding="utf-8"))
        onnx_model = onnx.load(output_dir / "model.onnx", load_external_data=False)
        metadata = {
            prop.key: json.loads(prop.value) for prop in onnx_model.metadata_props if prop.key == "com.qti.aisw.extdata"
        }
        assert metadata["com.qti.aisw.extdata"] == spec
        onnx.checker.check_model(onnx_model)

        with safe_open(str(source_dir / "model.safetensors"), framework="pt") as handle:
            source_tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        decoded_tensors = {}
        tensor_errors = {}
        for key in manifest["selected_keys"]:
            with safe_open(str(output_dir / "checkpoint/model.safetensors"), framework="pt") as handle:
                packed = handle.get_tensor(key)
            decoded = dequantize_packed_tensor(
                packed,
                manifest["tensors"][key]["logical_shape"],
                output_dtype=torch.float32,
            )
            decoded_tensors[key] = decoded
            tensor_errors[key] = _metrics(decoded, source_tensors[key])

        quantized_model = copy.deepcopy(original_model)
        parameters = dict(quantized_model.named_parameters())
        for key, decoded in decoded_tensors.items():
            parameters[key].data.copy_(decoded.to(torch.bfloat16))

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int64)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            original_prefill = original_model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True
            )
            quantized_prefill = quantized_model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True
            )
        prefill_metrics = {"logits": _metrics(quantized_prefill.logits, original_prefill.logits)}
        prefill_metrics["kv"] = _metrics(
            torch.cat([tensor.reshape(-1) for tensor in _flatten_cache(quantized_prefill.past_key_values)]),
            torch.cat([tensor.reshape(-1) for tensor in _flatten_cache(original_prefill.past_key_values)]),
        )

        original_cache = original_prefill.past_key_values
        quantized_cache = quantized_prefill.past_key_values
        decode_metrics = []
        for step, token in enumerate((9, 10, 11), start=1):
            next_ids = torch.tensor([[token]], dtype=torch.int64)
            next_mask = torch.ones((1, input_ids.shape[1] + step), dtype=torch.int64)
            with torch.no_grad():
                original_decode = original_model(
                    input_ids=next_ids,
                    attention_mask=next_mask,
                    past_key_values=original_cache,
                    use_cache=True,
                    return_dict=True,
                )
                quantized_decode = quantized_model(
                    input_ids=next_ids,
                    attention_mask=next_mask,
                    past_key_values=quantized_cache,
                    use_cache=True,
                    return_dict=True,
                )
            decode_metrics.append(
                {
                    "logits": _metrics(quantized_decode.logits, original_decode.logits),
                    "kv": _metrics(
                        torch.cat([tensor.reshape(-1) for tensor in _flatten_cache(quantized_decode.past_key_values)]),
                        torch.cat([tensor.reshape(-1) for tensor in _flatten_cache(original_decode.past_key_values)]),
                    ),
                }
            )
            original_cache = original_decode.past_key_values
            quantized_cache = quantized_decode.past_key_values

        first_key = manifest["selected_keys"][0]
        original_weight = source_tensors[first_key].float()
        decoded_weight = decoded_tensors[first_key]
        generator = torch.Generator().manual_seed(11)
        linear_input = torch.randn((3, original_weight.shape[1]), generator=generator)
        linear_metrics = _metrics(linear_input @ decoded_weight.t(), linear_input @ original_weight.t())

        validation_graph = temporary_dir / "validation_graph.onnx"
        replaced = _rewrite_for_cpu_reference(output_dir / "model.onnx", manifest, validation_graph)
        validation = {
            "selected_tensor_count": len(manifest["selected_keys"]),
            "preserved_tensor_count": len(source_tensors) - len(manifest["selected_keys"]),
            "mxfp6_node_count": sum(
                node.domain == "com.qualcomm.qeff" and node.op_type == "MXDequantize" for node in onnx_model.graph.node
            ),
            "cpu_reference_graph_replaced_nodes": replaced,
            "tensor_errors": tensor_errors,
            "linear_output_error": linear_metrics,
            "prefill_error": prefill_metrics,
            "decode_error": decode_metrics,
            "note": (
                "The validation graph replaces MXDequantize with logical inputs; stock ORT execution "
                "of the proposed custom op was not claimed."
            ),
        }
        (output_dir / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")

    artifact_paths = [output_dir / "model.onnx", output_dir / "weight_spec.json", output_dir / "validation.json"]
    artifact_paths.extend(
        sorted(path for path in (output_dir / "checkpoint").iterdir() if not path.name.startswith("."))
    )
    hashes = "\n".join(f"- `{path.relative_to(output_dir)}`: `{_sha256(path)}`" for path in artifact_paths)
    readme = f"""# Seeded Qwen MXFP6 compiler-reference bundle

This is a deterministic, randomly initialized two-layer `Qwen2ForCausalLM`
fixture (`torch.manual_seed(7)`), not a pretrained accuracy benchmark. It was
exported through QEfficient's weight-free Dynamo path with
`use_onnx_subfunctions=True`.

The checkpoint contains the seven dense projections per decoder layer
(`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
as UINT8 `inline_e8m0_fp6_lsb_v1` rows `[O, 25*ceil(K/32)]`. Other weights and
tied aliases remain on their ordinary paths. The graph uses proposed
`com.qualcomm.qeff::MXDequantize` nodes, custom opset 1, and the
`format="MXFP6_E2M3"`, `axis=-1`, `block_size=32`, `axis_size=K`, and
`output_dtype` set to the logical weight dtype.
Each block stores its E8M0 scale byte, including the all-zero-block scale
exponent `-2`.

Reproduce from the repository root:

```bash
python examples/weight_free/build_mxfp6_reference.py /path/to/reference_bundle
```

`validation.json` reports tensor reconstruction, a linear-layer reference,
and BF16 PyTorch prefill/decode logits and KV-cache differences. Its graph
check replaces each MXDequantize with a logical floating-point input for
structural CPU validation. Stock ONNX Runtime execution of the proposed
custom operator is not claimed. Compilation and accelerator execution were
not attempted; the ABI remains proposed pending compiler agreement.

SHA-256:

{hashes}
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(build_bundle(args.output_dir))


if __name__ == "__main__":
    main()
