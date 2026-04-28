#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#

import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx_ir as ir
import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from QEfficient.exporter.weight_spec import (
    TiedWeightAlias,
    WeightSpec,
    WeightSpecInput,
    load_weight_spec,
    resolve_weight_spec_path,
    save_weight_spec,
)
from QEfficient.utils.export_utils import (
    _cleanup_onnx_subfunctions,
    _setup_onnx_subfunctions,
    get_decoder_layer_classes_for_export,
)
from QEfficient.utils.logging_utils import logger
from QEfficient.utils.torch_patches import (
    temporarily_disable_nested_compile_regions,
    temporarily_enable_nested_compile_regions,
)


def _to_meta(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return torch.empty_like(value, device="meta")
    if isinstance(value, tuple):
        return tuple(_to_meta(item) for item in value)
    if isinstance(value, list):
        return [_to_meta(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_meta(item) for key, item in value.items()}
    return value


def _resolve_checkpoint_dir(model_id_or_path: str) -> Path:
    candidate = Path(model_id_or_path).expanduser()
    if candidate.exists():
        return candidate

    snapshot_dir = snapshot_download(
        repo_id=model_id_or_path,
        allow_patterns=["*.safetensors", "*.json"],
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.txt", "*.pdf", "*.msgpack", "*.h5", "*.pth"],
        resume_download=True,
    )
    return Path(snapshot_dir)


def _resolve_checkpoint_files(model_id_or_path: str) -> List[str]:
    checkpoint_dir = _resolve_checkpoint_dir(model_id_or_path)
    checkpoint_files = sorted(str(path.resolve()) for path in checkpoint_dir.glob("*.safetensors"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No safetensors checkpoint files found for {model_id_or_path}")
    return checkpoint_files


def _module_name_map(model: nn.Module) -> Dict[int, str]:
    return {id(module): name for name, module in model.named_modules()}


def _collect_tied_weights(model: nn.Module) -> List[TiedWeightAlias]:
    if not getattr(model.config, "tie_word_embeddings", False):
        return []

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        return []

    module_names = _module_name_map(model)
    canonical_name = module_names.get(id(input_embeddings))
    alias_name = module_names.get(id(output_embeddings))
    if not canonical_name or not alias_name or canonical_name == alias_name:
        return []

    return [TiedWeightAlias(alias=f"{alias_name}.weight", canonical=f"{canonical_name}.weight")]


def _build_meta_qeff_model(qeff_model):
    model_ref = qeff_model.hash_params.get("pretrained_model_name_or_path")
    if not model_ref:
        raise ValueError(
            "Weight-free export requires checkpoint metadata. "
            "Pass `pretrained_model_name_or_path=...` when constructing the QEff model manually."
        )

    if getattr(qeff_model.model.config, "quantization_config", None) is not None:
        raise NotImplementedError("Weight-free export is not implemented yet for quantized causal LM checkpoints.")

    config = copy.deepcopy(qeff_model.model.config)
    with init_empty_weights():
        meta_model = qeff_model._hf_auto_class.from_config(config, attn_implementation="eager")

    meta_qeff_model = qeff_model.__class__(
        meta_model,
        continuous_batching=getattr(qeff_model, "continuous_batching", False),
        qaic_config=copy.deepcopy(getattr(qeff_model.model, "qaic_config", None)),
        max_seq_len_cached=getattr(qeff_model.model.config, "max_seq_len_cached", None),
        pretrained_model_name_or_path=model_ref,
    )
    meta_qeff_model.hash_params.update(copy.deepcopy(qeff_model.hash_params))
    meta_qeff_model.model.eval()
    return meta_qeff_model


def _kind_map_for_model(model: nn.Module) -> Dict[str, Tuple[str, str]]:
    mapping = {name: ("parameter", name) for name, _ in model.named_parameters()}
    mapping.update({name: ("buffer", name) for name, _ in model.named_buffers()})
    return mapping


def _promote_initializers_and_build_spec(onnx_program, model_ref: str, model_name: str, qeff_model) -> WeightSpec:
    model_ir = onnx_program.model
    kind_map = _kind_map_for_model(qeff_model.model)
    promoted_inputs: List[WeightSpecInput] = []

    for name, init_value in list(model_ir.graph.initializers.items()):
        if name not in kind_map:
            continue

        model_ir.graph.inputs.append(
            ir.Value(
                name=name,
                shape=init_value.shape,
                type=ir.TensorType(init_value.dtype),
            )
        )
        del model_ir.graph.initializers[name]
        kind, fqn = kind_map[name]
        promoted_inputs.append(
            WeightSpecInput(
                name=name,
                fqn=fqn,
                kind=kind,
                shape=[int(dim) for dim in init_value.shape],
                dtype=str(init_value.dtype),
            )
        )

    return WeightSpec(
        model_name=model_name,
        model_id=model_ref,
        checkpoint_files=_resolve_checkpoint_files(model_ref),
        inputs=promoted_inputs,
        tied_weights=_collect_tied_weights(qeff_model.model),
    )


def export_weight_free_onnx(
    qeff_model,
    tmp_onnx_path: Path,
    example_inputs: Dict[str, torch.Tensor],
    input_names: List[str],
    output_names: List[str],
    dynamic_shapes: Dict[str, Any],
    export_kwargs: Dict[str, Any],
    onnx_transform_kwargs: Dict[str, Any],
):
    meta_qeff_model = _build_meta_qeff_model(qeff_model)
    cleanup_required = False

    if getattr(qeff_model, "_use_onnx_subfunctions", False):
        _, subfunc_kwargs = _setup_onnx_subfunctions(
            meta_qeff_model,
            (),
            {
                "use_dynamo": True,
                "onnx_transform_kwargs": copy.deepcopy(onnx_transform_kwargs),
                "output_names": list(output_names),
            },
        )
        onnx_transform_kwargs = subfunc_kwargs.get("onnx_transform_kwargs", onnx_transform_kwargs)
        cleanup_required = True

    decoder_layer_classes = get_decoder_layer_classes_for_export(meta_qeff_model.model)
    if getattr(meta_qeff_model, "_use_onnx_subfunctions", False) and decoder_layer_classes:
        export_context = temporarily_enable_nested_compile_regions(meta_qeff_model.model, decoder_layer_classes)
    else:
        export_context = temporarily_disable_nested_compile_regions(meta_qeff_model.model, decoder_layer_classes)

    meta_example_inputs = _to_meta(example_inputs)
    model_ref = meta_qeff_model.hash_params["pretrained_model_name_or_path"]

    with export_context:
        onnx_program = torch.onnx.export(
            meta_qeff_model.model,
            args=(),
            f=None,
            kwargs=meta_example_inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,
            dynamic_shapes=dynamic_shapes,
            **export_kwargs,
        )
        if onnx_program is None:
            raise RuntimeError("torch.onnx.export returned None for weight-free dynamo export")

        spec = _promote_initializers_and_build_spec(
            onnx_program=onnx_program,
            model_ref=model_ref,
            model_name=qeff_model.model_name,
            qeff_model=meta_qeff_model,
        )
        onnx_program.save(str(tmp_onnx_path))
        save_weight_spec(resolve_weight_spec_path(tmp_onnx_path), spec)

    def cleanup():
        if cleanup_required:
            _cleanup_onnx_subfunctions(meta_qeff_model)

    return meta_qeff_model, onnx_transform_kwargs, cleanup


def _spec_dtype_to_torch(dtype: str) -> torch.dtype:
    normalized = dtype.lower()
    if "bfloat16" in normalized:
        return torch.bfloat16
    if "float16" in normalized:
        return torch.float16
    if "float32" in normalized or normalized == "float":
        return torch.float32
    if "int64" in normalized:
        return torch.int64
    if "int32" in normalized:
        return torch.int32
    if "bool" in normalized:
        return torch.bool
    raise ValueError(f"Unsupported dtype in weight spec: {dtype}")


def _load_checkpoint_index(checkpoint_files: List[str]) -> Dict[str, str]:
    tensor_to_file = {}
    for checkpoint_file in checkpoint_files:
        handle = safe_open(checkpoint_file, framework="pt")
        for key in handle.keys():
            tensor_to_file[key] = checkpoint_file
    return tensor_to_file


def _load_checkpoint_tensor(checkpoint_index: Dict[str, str], key: str, dtype: str) -> np.ndarray:
    checkpoint_file = checkpoint_index.get(key)
    if checkpoint_file is None:
        raise KeyError(key)

    handle = safe_open(checkpoint_file, framework="pt")
    tensor = handle.get_tensor(key).detach().cpu()
    target_dtype = _spec_dtype_to_torch(dtype)
    if tensor.dtype != target_dtype:
        tensor = tensor.to(target_dtype)
    return tensor.numpy()


def _materialize_buffer_from_config(config, fqn: str, dtype: str) -> np.ndarray:
    target_dtype = _spec_dtype_to_torch(dtype)

    if fqn.endswith("inv_freq"):
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_type = rope_scaling.get("rope_type", "default") if rope_scaling else "default"
        inv_freq, _ = ROPE_INIT_FUNCTIONS[rope_type](config, device=torch.device("cpu"))
        return inv_freq.to(target_dtype).cpu().numpy()

    if fqn.endswith("cos_cached") or fqn.endswith("sin_cached"):
        from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaRotaryEmbedding

        rotary_embedding = QEffLlamaRotaryEmbedding(config=config, device=torch.device("cpu"))
        tensor = rotary_embedding.cos_cached if fqn.endswith("cos_cached") else rotary_embedding.sin_cached
        return tensor.to(target_dtype).cpu().numpy()

    raise KeyError(fqn)


def load_weight_free_ort_inputs(weight_spec_path: Path, runtime_inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    spec = load_weight_spec(weight_spec_path)
    config = AutoConfig.from_pretrained(spec.model_id)
    checkpoint_index = _load_checkpoint_index(spec.checkpoint_files)
    tied_weights = {entry.alias: entry.canonical for entry in spec.tied_weights}

    ort_inputs = dict(runtime_inputs)
    for spec_input in spec.inputs:
        if spec_input.name in ort_inputs:
            continue

        key = tied_weights.get(spec_input.fqn, spec_input.fqn)
        try:
            ort_inputs[spec_input.name] = _load_checkpoint_tensor(checkpoint_index, key, spec_input.dtype)
        except KeyError:
            if spec_input.kind != "buffer":
                raise
            ort_inputs[spec_input.name] = _materialize_buffer_from_config(config, spec_input.fqn, spec_input.dtype)

    return ort_inputs


def log_weight_free_export(onnx_path: Path) -> None:
    logger.info(f"Weight-free ONNX exported to {onnx_path} with spec {resolve_weight_spec_path(onnx_path)}")
