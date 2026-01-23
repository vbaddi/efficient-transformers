# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import logging
import os
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import numpy as np
import onnx
import torch
from onnx import ModelProto, TensorProto, external_data_helper, numpy_helper

from QEfficient.customop.ctx_scatter_gather import (
    CtxGather,
    CtxGather3D,
    CtxGatherBlockedKV,
    CtxGatherFunc,
    CtxGatherFunc3D,
    CtxGatherFuncBlockedKV,
    CtxScatter,
    CtxScatter3D,
    CtxScatterFunc,
    CtxScatterFunc3D,
)
from QEfficient.customop.ctx_scatter_gather_cb import (
    CtxGatherBlockedKVCB,
    CtxGatherCB,
    CtxGatherCB3D,
    CtxGatherFuncBlockedKVCB,
    CtxGatherFuncCB,
    CtxGatherFuncCB3D,
    CtxScatterCB,
    CtxScatterCB3D,
    CtxScatterFuncCB,
    CtxScatterFuncCB3D,
)
from QEfficient.customop.rms_norm import CustomRMSNorm, CustomRMSNormFunc
from QEfficient.utils.constants import FILE_CHUNK_SIZE_DEFAULT, ONNX_EXPORT_OPSET, SIZE_THRESHOLD_DEFAULT

logger = logging.getLogger(__name__)


class BaseOnnxTransform:
    """Base class for ONNX graph modifications. Should NOT be instantiated."""

    def __init__(self):
        raise TypeError("Transform classes are not to be instantiated. Use the `apply` method directly.")

    @classmethod
    def apply(cls, model: ModelProto, **kwargs) -> Tuple[ModelProto, bool]:
        raise NotImplementedError("Use subclasses for ONNX transform")


class FP16ClipTransform(BaseOnnxTransform):
    """Clip FP32 tensors to FP16 range to avoid overflow during conversion."""

    @classmethod
    def apply(cls, tensor: TensorProto, onnx_base_dir: str, fp16_max: float, fp16_min: float) -> bool:
        nptensor = numpy_helper.to_array(tensor, onnx_base_dir)
        if nptensor.dtype == np.float32 and (np.any(nptensor > fp16_max) or np.any(nptensor < fp16_min)):
            neg_inf_mask = np.isinf(nptensor) & (nptensor < 0)
            clipped_tensor = np.clip(nptensor, fp16_min, fp16_max)

            if neg_inf_mask.any():
                clipped_tensor = np.where(neg_inf_mask, np.float32("-inf"), clipped_tensor)

            tensor.CopyFrom(numpy_helper.from_array(clipped_tensor, tensor.name))
            return True
        return False


class SplitTensorsTransform(BaseOnnxTransform):
    """Split large tensors into external data files for efficient storage."""

    @classmethod
    def apply(
        cls, tensor: TensorProto, model_name: str, file_num: int, mapping: Dict[str, Tuple[TensorProto, str]]
    ) -> None:
        file_name = f"{model_name}_{file_num}.onnx.data"
        mapping[tensor.name] = (tensor, file_name)


class CustomOpTransform(BaseOnnxTransform):
    """Register custom ONNX ops and append their function prototypes to the model."""

    _custom_ops: Dict[str, Tuple[Any, Any]] = {
        "CustomRMSNormFunc": (CustomRMSNormFunc, CustomRMSNorm),
        "CtxScatterFunc": (CtxScatterFunc, CtxScatter),
        "CtxScatterFunc3D": (CtxScatterFunc3D, CtxScatter3D),
        "CtxGatherFunc": (CtxGatherFunc, CtxGather),
        "CtxGatherFunc3D": (CtxGatherFunc3D, CtxGather3D),
        "CtxScatterFuncCB3D": (CtxScatterFuncCB3D, CtxScatterCB3D),
        "CtxGatherFuncCB3D": (CtxGatherFuncCB3D, CtxGatherCB3D),
        "CtxGatherFuncBlockedKV": (CtxGatherFuncBlockedKV, CtxGatherBlockedKV),
        "CtxGatherFuncBlockedKVCB": (CtxGatherFuncBlockedKVCB, CtxGatherBlockedKVCB),
        "CtxScatterFuncCB": (CtxScatterFuncCB, CtxScatterCB),
        "CtxGatherFuncCB": (CtxGatherFuncCB, CtxGatherCB),
    }

    @classmethod
    def apply(cls, model: ModelProto) -> bool:
        op_applied = False
        for op_name, (func_class, _) in cls._custom_ops.items():
            if hasattr(func_class, "symbolic"):
                torch.onnx.register_custom_op_symbolic(f"::{op_name}", func_class.symbolic, ONNX_EXPORT_OPSET)

        existing = {f.name for f in model.functions}
        for _, onnxscript_func in cls._custom_ops.values():
            proto = onnxscript_func.to_function_proto()
            if proto.name not in existing:
                model.functions.append(proto)
                op_applied = True
        return op_applied


class RenameFunctionOutputsTransform(BaseOnnxTransform):
    """Rename outputs of decoder-related functions for better clarity."""

    @classmethod
    def apply(cls, model: ModelProto) -> bool:
        graph = model.graph
        op_type_to_func = {f.name: f for f in model.functions}
        decoder_patterns = ["DecoderLayer", "Block", "Layer"]
        renamed = False
        model_out_map = {v.name: i for i, v in enumerate(graph.output)}
        layer_idx = 0

        for node in graph.node:
            if any(p in node.name or p in node.op_type for p in decoder_patterns):
                func = op_type_to_func.get(node.op_type)
                if not func:
                    continue
                for i, out_name in enumerate(func.output):
                    if "_InternalRetainedState" in out_name:
                        renamed = True
                        orig = node.output[i]
                        new = (
                            f"past_key.{layer_idx}_RetainedState"
                            if "key" in out_name
                            else f"past_value.{layer_idx}_RetainedState"
                            if "value" in out_name
                            else orig
                        )
                        node.output[i] = new
                        if orig in model_out_map:
                            graph.output[model_out_map[orig]].name = new
                layer_idx += 1
        return renamed


class AdapterWeightsToInputsTransform(BaseOnnxTransform):
    @classmethod
    def apply(cls, model: onnx.ModelProto, *, adapter_name: str, **kwargs) -> Tuple[onnx.ModelProto, bool]:
        transformed = False
        removed_initializers = []

        # Find nodes with lora weights as inputs
        weight_suffix = f".{adapter_name}.weight"
        lora_weight_nodes = {
            inp: node for node in model.graph.node for inp in node.input if inp.endswith(weight_suffix)
        }

        for i, weight in enumerate(model.graph.initializer):
            if weight.name.endswith(weight_suffix):
                transformed = True

                # Create input/output for lora weights
                new_weight_name = weight.name[: -len(weight_suffix)] + ".weight"
                type_proto = onnx.helper.make_tensor_type_proto(weight.data_type, shape=list(weight.dims))
                inp = onnx.ValueInfoProto(name=new_weight_name, type=type_proto)
                out = onnx.ValueInfoProto(name=new_weight_name + "_RetainedState", type=type_proto)
                model.graph.input.append(inp)
                model.graph.output.append(out)

                # Create a node that connects input -> output
                node = onnx.helper.make_node("Identity", [inp.name], [out.name], new_weight_name + "_identity")
                model.graph.node.append(node)

                # Rename weight input
                lora_weight_node = lora_weight_nodes[weight.name]
                for j, inp in enumerate(lora_weight_node.input):
                    if inp == weight.name:
                        lora_weight_node.input[j] = new_weight_name

                # Remove weight initializers
                removed_initializers.append(i)

        if transformed:
            for i in sorted(removed_initializers, reverse=True):
                model.graph.initializer.pop(i)

        return model, transformed


class HierarchyPreservationTransform(BaseOnnxTransform):
    """
    Injects hierarchical node names into ONNX functions based on PyTorch module structure.

    This transform:
    1. Extracts actual submodule names from PyTorch model
    2. Uses pattern detection to identify functional regions (attention, MLP)
    3. Assigns hierarchical names to nodes within ONNX functions
    4. Ensures unique names by adding counters for duplicate operations

    Example:
        Before: Softmax_12
        After:  /model/layers.0/self_attn/Softmax
    """

    # Pattern definitions for submodule categorization
    ATTENTION_PATTERNS = [
        "attn",
        "attention",
        "self_attn",
        "self_attention",
        "cross_attn",
        "cross_attention",
        "mha",
        "multihead",
    ]

    MLP_PATTERNS = ["mlp", "ffn", "feed_forward", "feedforward", "ff", "dense", "intermediate"]

    NORM_PATTERNS = ["norm", "layernorm", "layer_norm", "rmsnorm", "rms_norm", "ln", "normalization"]

    ACTIVATION_OPS = {"Gelu", "Relu", "Silu", "Tanh", "Swish"}

    @classmethod
    def apply(
        cls, model: ModelProto, *, pytorch_model=None, function_module_types: Set[type] = None, **kwargs
    ) -> Tuple[ModelProto, bool]:
        """
        Apply hierarchy preservation to ONNX model.

        Args:
            model: ONNX model to transform
            pytorch_model: Original PyTorch model for reference
            function_module_types: Set of module types exported as functions

        Returns:
            Tuple of (transformed_model, was_transformed)
        """
        if not pytorch_model or not function_module_types:
            logger.debug("Skipping HierarchyPreservationTransform: missing pytorch_model or function_module_types")
            return model, False

        if not model.functions:
            logger.debug("Skipping HierarchyPreservationTransform: no functions in model")
            return model, False

        try:
            # Extract submodule structure
            submodule_structure = cls._extract_submodule_structure(pytorch_model, function_module_types)

            if not submodule_structure:
                logger.debug("No function module instances found")
                return model, False

            # Build weight mapping
            weight_to_module = cls._build_weight_mapping(model, submodule_structure)

            # Map function calls to instances
            call_to_instance = cls._map_calls_to_instances(model, submodule_structure)

            if not call_to_instance:
                logger.debug("No function calls mapped to instances")
                return model, False

            # Process each function
            func_to_calls = cls._group_calls_by_function(model)
            transformed = cls._process_functions(
                model, submodule_structure, weight_to_module, call_to_instance, func_to_calls
            )

            if transformed:
                logger.info("HierarchyPreservationTransform applied successfully")

            return model, transformed

        except Exception as e:
            logger.error(f"HierarchyPreservationTransform failed: {e}")
            return model, False

    @staticmethod
    def _extract_submodule_structure(pytorch_model, function_module_types: Set[type]) -> Dict[str, Dict[str, str]]:
        """Extract submodule structure from PyTorch model."""
        structure = {}

        for name, module in pytorch_model.named_modules():
            if type(module) in function_module_types and name:
                submodules = {
                    child_name: type(child_module).__name__ for child_name, child_module in module.named_children()
                }
                structure[name] = submodules

        return structure

    @classmethod
    def _categorize_submodules(cls, submodules: Dict[str, str]) -> Dict[str, str]:
        """Categorize submodules into functional types."""
        categories = {}

        for submodule_name, submodule_type in submodules.items():
            name_lower = submodule_name.lower()
            type_lower = submodule_type.lower()
            combined = f"{name_lower} {type_lower}"

            # Check attention
            if any(pattern in combined for pattern in cls.ATTENTION_PATTERNS):
                if "attention" not in categories:
                    categories["attention"] = submodule_name

            # Check MLP
            elif any(pattern in combined for pattern in cls.MLP_PATTERNS):
                if "mlp" not in categories:
                    categories["mlp"] = submodule_name

            # Check normalization
            elif any(pattern in combined for pattern in cls.NORM_PATTERNS):
                if "input" in name_lower or "pre" in name_lower or name_lower.endswith("1"):
                    categories["norm_pre_attn"] = submodule_name
                elif "post" in name_lower or name_lower.endswith("2"):
                    categories["norm_post_attn"] = submodule_name
                elif "norm" not in categories:
                    categories["norm"] = submodule_name

        return categories

    @staticmethod
    def _build_weight_mapping(model: ModelProto, submodule_structure: Dict) -> Dict[str, Tuple[str, str]]:
        """Build mapping from weight names to (instance, relative_module)."""
        weight_to_module = {}

        for init in model.graph.initializer:
            weight_name = init.name

            for instance in submodule_structure.keys():
                if weight_name.startswith(f"{instance}."):
                    relative_key = weight_name[len(instance) + 1 :]

                    if "." in relative_key:
                        relative_module = relative_key.rsplit(".", 1)[0]
                        weight_to_module[weight_name] = (instance, relative_module)
                        break

        return weight_to_module

    @staticmethod
    def _map_calls_to_instances(model: ModelProto, submodule_structure: Dict) -> Dict[str, str]:
        """Map function call nodes to module instances."""
        function_names = {f.name for f in model.functions}
        call_nodes = [node for node in model.graph.node if node.op_type in function_names]
        module_instances = list(submodule_structure.keys())

        return {
            call_node.name: module_instances[idx]
            for idx, call_node in enumerate(call_nodes)
            if idx < len(module_instances)
        }

    @staticmethod
    def _group_calls_by_function(model: ModelProto) -> Dict[str, List]:
        """Group call nodes by function name."""
        function_names = {f.name for f in model.functions}
        func_to_calls = defaultdict(list)

        for node in model.graph.node:
            if node.op_type in function_names:
                func_to_calls[node.op_type].append(node)

        return dict(func_to_calls)

    @classmethod
    def _process_functions(
        cls,
        model: ModelProto,
        submodule_structure: Dict,
        weight_to_module: Dict,
        call_to_instance: Dict,
        func_to_calls: Dict,
    ) -> bool:
        """Process each function to inject hierarchical names."""
        transformed = False

        for func in model.functions:
            calls = func_to_calls.get(func.name, [])
            if not calls:
                continue

            representative_instance = call_to_instance.get(calls[0].name)
            if not representative_instance:
                continue

            # Get submodule structure for this instance
            instance_submodules = submodule_structure.get(representative_instance, {})
            submodule_categories = cls._categorize_submodules(instance_submodules)

            # Build weight mapping for this instance
            instance_weight_to_module = {
                weight_name: rel_module
                for weight_name, (instance, rel_module) in weight_to_module.items()
                if instance == representative_instance
            }

            # Identify functional regions
            attention_region = cls._identify_attention_regions(func.node)
            mlp_region = cls._identify_mlp_regions(func.node, attention_region)

            # Process nodes
            if cls._assign_hierarchical_names(
                func,
                representative_instance,
                instance_weight_to_module,
                attention_region,
                mlp_region,
                submodule_categories,
            ):
                transformed = True

        return transformed

    @staticmethod
    def _identify_attention_regions(nodes: List, window_size: int = 50) -> Set[int]:
        """Identify node indices belonging to attention mechanisms."""
        attention_region = set()
        softmax_indices = [i for i, node in enumerate(nodes) if node.op_type == "Softmax"]

        for softmax_idx in softmax_indices:
            start = max(0, softmax_idx - window_size)
            end = min(len(nodes), softmax_idx + window_size)
            attention_region.update(range(start, end))

        return attention_region

    @classmethod
    def _identify_mlp_regions(cls, nodes: List, attention_region: Set[int], window_size: int = 30) -> Set[int]:
        """Identify node indices belonging to MLP layers."""
        mlp_region = set()
        activation_indices = [
            i for i, node in enumerate(nodes) if node.op_type in cls.ACTIVATION_OPS and i not in attention_region
        ]

        for act_idx in activation_indices:
            start = max(0, act_idx - window_size)
            end = min(len(nodes), act_idx + window_size)
            mlp_region.update(range(start, end))

        return mlp_region

    @classmethod
    def _assign_hierarchical_names(
        cls,
        func,
        representative_instance: str,
        instance_weight_to_module: Dict,
        attention_region: Set[int],
        mlp_region: Set[int],
        submodule_categories: Dict,
    ) -> bool:
        """Assign hierarchical names to nodes in function."""
        # Initialize tracking
        tensor_to_module = {inp: "" for inp in func.input}
        tensor_is_constant = {out: True for node in func.node if node.op_type == "Constant" for out in node.output}

        # First pass: determine module for each node
        node_modules = []
        for node_idx, node in enumerate(func.node):
            node_module = cls._infer_node_module(
                node,
                node_idx,
                instance_weight_to_module,
                tensor_to_module,
                tensor_is_constant,
                attention_region,
                mlp_region,
                submodule_categories,
            )
            node_modules.append(node_module)

            # Update tensor tracking
            for out in node.output:
                tensor_to_module[out] = node_module

        # Second pass: assign unique names
        op_counters = defaultdict(lambda: defaultdict(int))

        for node_idx, node in enumerate(func.node):
            node_module = node_modules[node_idx]

            # Build hierarchical path
            full_path = representative_instance
            if node_module:
                full_path = f"{full_path}.{node_module}"

            hier_path = cls._path_to_hierarchical(full_path)
            op_type = node.op_type

            # Count operations
            op_counters[hier_path][op_type] += 1
            count = op_counters[hier_path][op_type]

            # Check if counter needed
            total_same_ops = sum(
                1 for i, mod in enumerate(node_modules) if mod == node_module and func.node[i].op_type == op_type
            )

            # Assign name
            if total_same_ops > 1:
                node.name = f"/{hier_path}/{op_type}_{count - 1}"
            else:
                node.name = f"/{hier_path}/{op_type}"

        return True

    @staticmethod
    def _infer_node_module(
        node,
        node_idx: int,
        instance_weight_to_module: Dict,
        tensor_to_module: Dict,
        tensor_is_constant: Dict,
        attention_region: Set[int],
        mlp_region: Set[int],
        submodule_categories: Dict,
    ) -> str:
        """Infer which submodule a node belongs to."""
        # Strategy 1: Direct weight usage
        weight_modules = {instance_weight_to_module[inp] for inp in node.input if inp in instance_weight_to_module}

        if weight_modules:
            return HierarchyPreservationTransform._common_prefix(list(weight_modules)).rstrip(".")

        # Strategy 2: Pattern-based regions
        if node_idx in attention_region:
            return submodule_categories.get("attention", "self_attn")

        if node_idx in mlp_region:
            return submodule_categories.get("mlp", "mlp")

        # Strategy 3: Data flow inheritance
        data_modules = {
            tensor_to_module[inp]
            for inp in node.input
            if inp in tensor_to_module and not tensor_is_constant.get(inp, False) and tensor_to_module[inp]
        }

        if data_modules:
            return HierarchyPreservationTransform._common_prefix(list(data_modules)).rstrip(".")

        return ""

    @staticmethod
    def _common_prefix(strings: List[str]) -> str:
        """Find longest common prefix among strings."""
        if not strings:
            return ""

        min_s = min(strings)
        max_s = max(strings)

        if not min_s:
            return ""

        for i in range(len(min_s)):
            if min_s[i] != max_s[i]:
                return min_s[:i]

        return min_s

    @staticmethod
    def _path_to_hierarchical(dot_path: str) -> str:
        """Convert dot notation to hierarchical path format."""
        parts = dot_path.split(".")
        hier_parts = []

        for p in parts:
            if p.isdigit() and hier_parts:
                hier_parts[-1] = f"{hier_parts[-1]}.{p}"
            else:
                hier_parts.append(p)

        return "/".join(hier_parts)


class OnnxTransformPipeline(BaseOnnxTransform):
    """Pipeline to apply multiple ONNX transformations in sequence."""

    def __init__(self, transforms: List[Type[BaseOnnxTransform]]):
        if not transforms:
            warnings.warn("Transform list is empty. No transformations will be applied.")
        self.transforms = transforms

    def apply(
        self,
        model: ModelProto,
        *,
        model_name: str = "",
        onnx_base_dir: Optional[str] = None,
        file_chunk_size: int = FILE_CHUNK_SIZE_DEFAULT,
        size_threshold: int = SIZE_THRESHOLD_DEFAULT,
        **kwargs,
    ) -> Tuple[ModelProto, bool]:
        if not self.transforms:
            return model, False

        # Same logic as before, but replace `transforms` with `self.transforms`
        mapping: Dict[str, Tuple[TensorProto, str]] = {}
        requested = set(self.transforms)
        applied = {t: False for t in requested}
        f16_applied = False
        do_fp16 = FP16ClipTransform in requested
        do_split = SplitTensorsTransform in requested
        fp16_min, fp16_max = np.finfo(np.float16).min, np.finfo(np.float16).max
        file_num_tracker = {"num": 0, "size": 0}
        external_data_helper.load_external_data_for_model(model, onnx_base_dir)

        if do_fp16 or do_split:
            for tensor in external_data_helper._get_all_tensors(model):
                if do_fp16 and FP16ClipTransform.apply(tensor, onnx_base_dir, fp16_max, fp16_min):
                    f16_applied = True
                applied[FP16ClipTransform] = f16_applied

                if do_split and tensor.HasField("raw_data"):
                    tsize = len(tensor.raw_data)
                    if tsize > size_threshold:
                        if file_num_tracker["size"] + tsize > file_chunk_size:
                            file_num_tracker["num"] += 1
                            file_num_tracker["size"] = tsize
                        else:
                            file_num_tracker["size"] += tsize
                        applied[SplitTensorsTransform] = True
                        SplitTensorsTransform.apply(tensor, model_name, file_num_tracker["num"], mapping)

        def _set_external_data(tensor, file_name):
            external_data_helper.set_external_data(tensor, file_name)

        max_workers = min(32, (os.cpu_count() or 1) * 4)
        logger.info(f"Applying external data mapping with {max_workers} threads")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_set_external_data, tensor, file_name) for tensor, file_name in mapping.values()]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to set external data: {e}")

        # Non-looping transforms
        if CustomOpTransform in requested:
            applied[CustomOpTransform] = CustomOpTransform.apply(model)

        if RenameFunctionOutputsTransform in requested:
            applied[RenameFunctionOutputsTransform] = RenameFunctionOutputsTransform.apply(model)

        if AdapterWeightsToInputsTransform in requested:
            applied[AdapterWeightsToInputsTransform] = AdapterWeightsToInputsTransform.apply(model, **kwargs)

        if HierarchyPreservationTransform in requested:
            model, transformed = HierarchyPreservationTransform.apply(model, **kwargs)
            applied[HierarchyPreservationTransform] = transformed

        for t, done in applied.items():
            logger.info(f"Transform '{t.__name__}' applied={done}")

        return model, any(applied.values())
