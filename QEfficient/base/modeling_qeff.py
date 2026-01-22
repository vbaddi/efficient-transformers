# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import gc
import inspect
import logging
import shutil
import subprocess
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import onnx
import torch

from QEfficient.base.onnx_transforms import (
    BaseOnnxTransform,
    OnnxTransformPipeline,
)
from QEfficient.base.pytorch_transforms import PytorchTransform
from QEfficient.compile.qnn_compiler import compile as qnn_compile
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import (
    constants,
    create_json,
    create_model_params,
    dump_qconfig,
    generate_mdp_partition_config,
    hash_dict_params,
    load_json,
)
from QEfficient.utils.export_utils import export_wrapper

logger = logging.getLogger(__name__)


class QEFFBaseModel(ABC):
    """
    Base class for all the model classes (i.e. LLMs, SD, quantized etc.).
    Provides certain utility methods to be used by child classes.

    Class variables:
    :_pytorch_transforms: Pytorch transformations to be applied after initialization.
    :_onnx_transforms: ONNX transformations to be applied after ONNX export.
    """

    _pytorch_transforms: List[PytorchTransform]
    _onnx_transforms = [BaseOnnxTransform]

    @classmethod
    def _transform_names(cls) -> List[str]:
        return [x.__name__ for x in cls._pytorch_transforms + cls._onnx_transforms]

    def __init__(self, model: torch.nn.Module, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.hash_params = create_model_params(self, **kwargs)
        self.onnx_path: Optional[str] = None
        self.qpc_path: Optional[str] = None
        self.qpc_session: Optional[QAICInferenceSession] = None
        self.model_architecture = (
            (arch := getattr(self.model.config, "architectures", None)) and len(arch) > 0 and arch[0]
        ) or None

        # Flag for checking if weights are offloaded
        self._is_weights_offloaded: bool = False

        # Apply the transformations
        any_transformed = False
        for transform in self._pytorch_transforms:
            self.model, transformed = transform.apply(self.model)
            any_transformed = any_transformed or transformed

        if not any_transformed:
            warnings.warn(f"No transforms applied to model: {self.model_name}. It may be an unsupported model!")
        else:
            logger.info(f"Pytorch transforms applied to model: {self.model_name}")

    def _offload_model_weights(self, offload_pt_weights: bool) -> bool:
        """Clear PyTorch model weights to reduce memory usage after ONNX export."""
        if offload_pt_weights and not self._is_weights_offloaded:
            try:
                for param in self.model.parameters():
                    if param.storage():
                        param.storage().resize_(0)
                for buffer in self.model.buffers():
                    if buffer.storage():
                        buffer.storage().resize_(0)

                meta_model = self.model.to("meta")
                del self.model
                gc.collect()

                self.model = meta_model
                self._is_weights_offloaded = True
                return True
            except Exception as e:
                logger.warning(f"Weight clearing failed, continuing: {e}")
                return False
        return False

    def _model_offloaded_check(self) -> None:
        """
        Check if the model is in meta state or weights are offloaded.

        Raises:
            RuntimeError: If model is in meta state or if weights are offloaded
        """
        if self._is_weights_offloaded or any(param.is_meta for param in self.model.parameters()):
            error_msg = (
                "Cannot re-export model: weights have been offloaded to save memory. "
                "To re-export, please create a new model instance using from_pretrained() method."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @property
    def model_name(self) -> str:
        """
        Get the model class name without QEff/QEFF prefix.

        This property extracts the underlying model's class name and removes
        any QEff or QEFF prefix that may have been added during wrapping.

        Returns:
            str: Model class name (e.g., "CLIPTextModel" instead of "QEffCLIPTextModel")
        """
        mname = self.model.__class__.__name__
        if mname.startswith("QEff") or mname.startswith("QEFF"):
            mname = mname[4:]
        return mname

    @property
    @abstractmethod
    def get_model_config(self) -> Dict:
        """
        Get the model configuration as a dictionary.

        This is an abstract property that must be implemented by all subclasses.
        Typically returns: self.model.config.__dict__

        Returns:
            Dict: The configuration dictionary of the underlying model
        """
        pass

    @abstractmethod
    def export(self, export_dir: Optional[str] = None) -> Path:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.

        Args:
            :export_dir (str): Specify the export directory. The export_dir will be suffixed with a hash corresponding to current model.

        Returns:
            :Path: Path of the generated ``ONNX`` file.
        """

    @abstractmethod
    def compile(self, *args, **kwargs) -> Path:
        """
        Compile the exported onnx to run on AI100.
        If the model has not been exported yet, this method will handle the export process.

        Args:
            :onnx_path (str): Onnx file to compile
            :compile_dir (str): Directory path to compile the qpc. A suffix is added to the directory path to avoid reusing same qpc for different parameters.
            :num_devices (int): Number of devices to compile for. ``Defaults to 1``.
            :num_cores (int): Number of cores to utilize in each device ``Defaults to 16``.
            :mxfp6_matmul (bool): Use MXFP6 to compress weights for MatMul nodes to run faster on device. ``Defaults to False``.
            :mxint8_kv_cache (bool): Use MXINT8 to compress KV-cache on device to access and update KV-cache faster. ``Defaults to False``.
            :compiler_options: Pass any compiler option as input.

                Following flag can be passed in compiler_options to enable QNN Compilation path.
                    :enable_qnn (bool): Enables QNN Compilation. ``Defaults to False. if not passed.``
                    :qnn_config (str): Path of QNN Config parameters file. ``Defaults to None. if not passed``

                for QAIC compilation path, any flag that is supported by ``qaic-exec`` can be passed. Params are converted to flags as below:

                    - aic_num_cores=16 -> -aic-num-cores=16
                    - convert_to_fp16=True -> -convert-to-fp16
                    - aic_hw_version=ai100 -> -aic-hw-version=ai100
                    - aic_hw_version=ai200 -> -aic-hw-version=ai200

        ``QEFFAutoModelForCausalLM`` Args:

            :full_batch_size (int): Full batch size to allocate cache lines.
            :batch_size (int): Batch size to compile for. ``Defaults to 1``.
            :prefill_seq_len (int): Prefill sequence length to compile for. Prompt will be chunked according to this length.
            :ctx_len (int): Context length to allocate space for KV-cache tensors.

        Returns:
            :str: Path of the compiled ``qpc`` package.
        """

    # ────────────────────────────────────────────────
    # Helper functions for hierarchy naming
    # ────────────────────────────────────────────────

    # def common_prefix(self, strings: List[str]) -> str:
    #     """Find common prefix of a list of strings."""
    #     if not strings:
    #         return ''
    #     min_s = min(strings)
    #     max_s = max(strings)
    #     if not min_s:
    #         return ''
    #     for i in range(len(min_s)):
    #         if min_s[i] != max_s[i]:
    #             return min_s[:i]
    #     return min_s

    # def path_to_hierarchical(self, dot_path: str) -> str:
    #     """
    #     Convert PyTorch module path to hierarchical format.

    #     Examples:
    #         'layers.0.attn.q_proj' → 'layers.0/attn/q_proj'
    #         'model.layers.1.mlp' → 'model/layers.1/mlp'

    #     Keeps numeric indices attached to their parent (layers.0, not layers/0)
    #     """
    #     parts = dot_path.split('.')
    #     hier_parts = []

    #     for p in parts:
    #         # If this part is a digit and we have a previous part, attach it
    #         if p.isdigit() and hier_parts:
    #             hier_parts[-1] += '.' + p
    #         else:
    #             hier_parts.append(p)

    #     return '/'.join(hier_parts)

    # def process_onnx_with_hierarchy(
    #     self,
    #     onnx_model,
    #     function_module_types: Set[type],
    #     model: torch.nn.Module
    # ) -> None:
    #     """
    #     Post-process ONNX model to inject hierarchical node names.
    #     Fixed to prevent incorrect module propagation.
    #     """
    #     print("\n  Post-processing ONNX with hierarchy...")

    #     # Find all module instances
    #     module_instances = [
    #         name for name, mod in model.named_modules()
    #         if type(mod) in function_module_types and name
    #     ]

    #     print(f"    Found {len(module_instances)} module instances:")
    #     for inst in module_instances:
    #         print(f"      - {inst}")

    #     if not onnx_model.functions:
    #         print("    ⚠️  No functions found in ONNX model!")
    #         return onnx_model

    #     print(f"\n    Original functions: {[f.name for f in onnx_model.functions]}")

    #     # Build weight-to-module mapping
    #     print(f"\n    Building weight-to-module mapping...")
    #     weight_to_module = {}

    #     for init in onnx_model.graph.initializer:
    #         weight_name = init.name

    #         for instance in module_instances:
    #             if weight_name.startswith(instance + "."):
    #                 relative_key = weight_name[len(instance) + 1:]

    #                 if '.' in relative_key:
    #                     relative_module = relative_key.rsplit('.', 1)[0]
    #                     weight_to_module[weight_name] = (instance, relative_module)
    #                     break

    #     print(f"    Mapped {len(weight_to_module)} weights")

    #     # Show what modules we have
    #     unique_modules = set()
    #     for instance, rel_module in weight_to_module.values():
    #         unique_modules.add(rel_module)
    #     print(f"    Unique submodules found: {sorted(unique_modules)}")

    #     # Analyze calls
    #     call_nodes_in_order = []
    #     for node in onnx_model.graph.node:
    #         if node.op_type in [f.name for f in onnx_model.functions]:
    #             call_nodes_in_order.append(node)

    #     call_to_instance = {}
    #     for idx, call_node in enumerate(call_nodes_in_order):
    #         if idx < len(module_instances):
    #             call_to_instance[call_node.name] = module_instances[idx]

    #     func_to_calls = {}
    #     for call_node in call_nodes_in_order:
    #         func_name = call_node.op_type
    #         if func_name not in func_to_calls:
    #             func_to_calls[func_name] = []
    #         func_to_calls[func_name].append(call_node)

    #     # Process each function
    #     for func in onnx_model.functions:
    #         print(f"\n    ═══════════════════════════════════════════")
    #         print(f"    Processing: {func.name}")
    #         print(f"    ═══════════════════════════════════════════")

    #         calls_for_this_func = func_to_calls.get(func.name, [])

    #         if not calls_for_this_func:
    #             print(f"      Skipping - no calls")
    #             continue

    #         representative_call = calls_for_this_func[0]
    #         representative_instance = call_to_instance.get(representative_call.name)

    #         if not representative_instance:
    #             print(f"      Skipping - no instance")
    #             continue

    #         print(f"      Instance: {representative_instance}")

    #         # Build weight mapping for this instance
    #         instance_weight_to_module = {}
    #         for weight_name, (instance, rel_module) in weight_to_module.items():
    #             if instance == representative_instance:
    #                 instance_weight_to_module[weight_name] = rel_module

    #         print(f"      Weights: {len(instance_weight_to_module)}")

    #         # Track tensors - but DON'T propagate across module boundaries
    #         tensor_to_module = {}
    #         tensor_is_constant = {}
    #         last_weight_module = None  # Track the last module that used weights

    #         for inp in func.input:
    #             tensor_to_module[inp] = ''

    #         for node in func.node:
    #             if node.op_type == 'Constant':
    #                 for out in node.output:
    #                     tensor_is_constant[out] = True

    #         # Process nodes
    #         for node_idx, node in enumerate(func.node):
    #             # Check if node uses weights
    #             weight_modules = set()
    #             for inp in node.input:
    #                 if inp in instance_weight_to_module:
    #                     weight_modules.add(instance_weight_to_module[inp])

    #             # Check data flow
    #             data_modules = set()
    #             for inp in node.input:
    #                 if inp in tensor_to_module and not tensor_is_constant.get(inp, False):
    #                     if tensor_to_module[inp]:  # Only add non-empty
    #                         data_modules.add(tensor_to_module[inp])

    #             # Determine module with better logic
    #             if weight_modules:
    #                 # Node uses weights - this is authoritative
    #                 node_module = self.common_prefix(list(weight_modules)).rstrip('.')
    #                 last_weight_module = node_module
    #             elif data_modules:
    #                 # Node doesn't use weights - inherit from inputs
    #                 # But only if all inputs agree on the module
    #                 if len(data_modules) == 1:
    #                     node_module = list(data_modules)[0]
    #                 else:
    #                     # Multiple different modules - use common prefix
    #                     node_module = self.common_prefix(list(data_modules)).rstrip('.')
    #             else:
    #                 # No information - use last known weight module if recent
    #                 # This helps operations like Softmax that follow attention MatMuls
    #                 if last_weight_module and node_idx > 0:
    #                     # Check if previous node was in the same module
    #                     prev_node = func.node[node_idx - 1]
    #                     if prev_node.name and last_weight_module in prev_node.name:
    #                         node_module = last_weight_module
    #                     else:
    #                         node_module = ''
    #                 else:
    #                     node_module = ''

    #             # Build path
    #             full_path = representative_instance
    #             if node_module:
    #                 full_path += '.' + node_module

    #             hier_path = self.path_to_hierarchical(full_path)
    #             new_node_name = f"/{hier_path}/{node.op_type}"

    #             # DEBUG for Softmax
    #             if node.op_type == 'Softmax':
    #                 print(f"\n      🔍 SOFTMAX (node {node_idx}):")
    #                 print(f"         Inputs: {list(node.input)}")
    #                 print(f"         Weight modules: {weight_modules}")
    #                 print(f"         Data modules: {data_modules}")
    #                 print(f"         Last weight module: {last_weight_module}")
    #                 print(f"         Determined: '{node_module}'")
    #                 print(f"         New name: {new_node_name}")

    #                 # Check previous few nodes
    #                 print(f"         Previous 3 nodes:")
    #                 for i in range(max(0, node_idx - 3), node_idx):
    #                     prev = func.node[i]
    #                     print(f"           {i}: {prev.op_type} -> {prev.name}")

    #             # Update name
    #             node.name = new_node_name

    #             # Propagate - but only if we have a definitive module
    #             if weight_modules:
    #                 # Definitive - propagate
    #                 for out in node.output:
    #                     tensor_to_module[out] = node_module
    #             elif node_module:
    #                 # Inherited - propagate but mark as less certain
    #                 for out in node.output:
    #                     tensor_to_module[out] = node_module
    #             else:
    #                 # No module - don't propagate
    #                 for out in node.output:
    #                     tensor_to_module[out] = ''

    #             if node_idx < 3:
    #                 print(f"        {new_node_name}")

    #         print(f"      ✓ Done")

    #     return onnx_model

    # def process_onnx_with_hierarchy(
    #     self,
    #     onnx_model,
    #     function_module_types: Set[type],
    #     model: torch.nn.Module
    # ) -> None:
    #     """
    #     Post-process ONNX model to inject hierarchical node names.
    #     Uses operation patterns to identify attention vs MLP vs LayerNorm.
    #     """
    #     print("\n  Post-processing ONNX with hierarchy...")

    #     # Find all module instances
    #     module_instances = [
    #         name for name, mod in model.named_modules()
    #         if type(mod) in function_module_types and name
    #     ]

    #     print(f"    Found {len(module_instances)} module instances:")
    #     for inst in module_instances:
    #         print(f"      - {inst}")

    #     if not onnx_model.functions:
    #         print("    ⚠️  No functions found in ONNX model!")
    #         return onnx_model

    #     print(f"\n    Original functions: {[f.name for f in onnx_model.functions]}")

    #     # Build weight-to-module mapping
    #     weight_to_module = {}
    #     for init in onnx_model.graph.initializer:
    #         weight_name = init.name
    #         for instance in module_instances:
    #             if weight_name.startswith(instance + "."):
    #                 relative_key = weight_name[len(instance) + 1:]
    #                 if '.' in relative_key:
    #                     relative_module = relative_key.rsplit('.', 1)[0]
    #                     weight_to_module[weight_name] = (instance, relative_module)
    #                     break

    #     print(f"    Mapped {len(weight_to_module)} weights")

    #     # Analyze calls
    #     call_nodes_in_order = []
    #     for node in onnx_model.graph.node:
    #         if node.op_type in [f.name for f in onnx_model.functions]:
    #             call_nodes_in_order.append(node)

    #     call_to_instance = {}
    #     for idx, call_node in enumerate(call_nodes_in_order):
    #         if idx < len(module_instances):
    #             call_to_instance[call_node.name] = module_instances[idx]

    #     func_to_calls = {}
    #     for call_node in call_nodes_in_order:
    #         func_name = call_node.op_type
    #         if func_name not in func_to_calls:
    #             func_to_calls[func_name] = []
    #         func_to_calls[func_name].append(call_node)

    #     # Process each function
    #     for func in onnx_model.functions:
    #         print(f"\n    Processing: {func.name}")

    #         calls_for_this_func = func_to_calls.get(func.name, [])
    #         if not calls_for_this_func:
    #             continue

    #         representative_call = calls_for_this_func[0]
    #         representative_instance = call_to_instance.get(representative_call.name)
    #         if not representative_instance:
    #             continue

    #         print(f"      Instance: {representative_instance}")

    #         # Build weight mapping
    #         instance_weight_to_module = {}
    #         for weight_name, (instance, rel_module) in weight_to_module.items():
    #             if instance == representative_instance:
    #                 instance_weight_to_module[weight_name] = rel_module

    #         # Identify attention region by finding Softmax
    #         softmax_indices = [i for i, n in enumerate(func.node) if n.op_type == 'Softmax']
    #         print(f"      Found Softmax at indices: {softmax_indices}")

    #         # Define regions based on patterns
    #         # Typical transformer layer structure:
    #         # 1. input_layernorm (0 to ~10)
    #         # 2. self_attn (around softmax, ~10 to ~150)
    #         # 3. residual add
    #         # 4. post_attention_layernorm
    #         # 5. mlp
    #         # 6. final residual add

    #         attention_region = set()
    #         if softmax_indices:
    #             # Attention region: 50 nodes before softmax to 50 nodes after
    #             for softmax_idx in softmax_indices:
    #                 for i in range(max(0, softmax_idx - 50), min(len(func.node), softmax_idx + 50)):
    #                     attention_region.add(i)

    #         print(f"      Attention region: {len(attention_region)} nodes")

    #         # Track tensors
    #         tensor_to_module = {}
    #         tensor_is_constant = {}

    #         for inp in func.input:
    #             tensor_to_module[inp] = ''

    #         for node in func.node:
    #             if node.op_type == 'Constant':
    #                 for out in node.output:
    #                     tensor_is_constant[out] = True

    #         # Process nodes
    #         for node_idx, node in enumerate(func.node):
    #             # Check if node uses weights
    #             weight_modules = set()
    #             for inp in node.input:
    #                 if inp in instance_weight_to_module:
    #                     weight_modules.add(instance_weight_to_module[inp])

    #             # Determine module
    #             if weight_modules:
    #                 # Has weights - use weight module
    #                 node_module = self.common_prefix(list(weight_modules)).rstrip('.')
    #             elif node_idx in attention_region:
    #                 # In attention region but no weights - it's self_attn
    #                 node_module = 'self_attn'
    #             else:
    #                 # Check data flow
    #                 data_modules = set()
    #                 for inp in node.input:
    #                     if inp in tensor_to_module and not tensor_is_constant.get(inp, False):
    #                         if tensor_to_module[inp]:
    #                             data_modules.add(tensor_to_module[inp])

    #                 if data_modules:
    #                     node_module = self.common_prefix(list(data_modules)).rstrip('.')
    #                 else:
    #                     # Check if it's likely MLP (after attention, has GELU nearby)
    #                     # Look for GELU in nearby nodes
    #                     has_gelu_nearby = any(
    #                         func.node[i].op_type in ['Gelu', 'Relu', 'Silu']
    #                         for i in range(max(0, node_idx - 10), min(len(func.node), node_idx + 10))
    #                     )
    #                     if has_gelu_nearby and node_idx > max(softmax_indices) if softmax_indices else False:
    #                         node_module = 'mlp'
    #                     else:
    #                         node_module = ''

    #             # Build path
    #             full_path = representative_instance
    #             if node_module:
    #                 full_path += '.' + node_module

    #             hier_path = self.path_to_hierarchical(full_path)
    #             new_node_name = f"/{hier_path}/{node.op_type}"

    #             # DEBUG for key operations
    #             if node.op_type in ['Softmax', 'Gelu', 'Silu']:
    #                 print(f"      {node.op_type} at {node_idx}: {new_node_name}")

    #             # Update name
    #             node.name = new_node_name

    #             # Propagate
    #             for out in node.output:
    #                 tensor_to_module[out] = node_module

    #             if node_idx < 3:
    #                 print(f"        {new_node_name}")

    #         print(f"      ✓ Done")

    #     return onnx_model

    # Add these methods to your QEFFBaseModel class

    def common_prefix(self, strings: List[str]) -> str:
        """
        Find the longest common prefix among a list of strings.

        Args:
            strings: List of strings to compare

        Returns:
            Common prefix string, empty if no common prefix exists
        """
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

    def path_to_hierarchical(self, dot_path: str) -> str:
        """
        Convert PyTorch dot notation to hierarchical path format.
        Keeps numeric indices attached to their parent module.

        Args:
            dot_path: Module path in dot notation

        Returns:
            Hierarchical path with slashes
        """
        parts = dot_path.split(".")
        hier_parts = []

        for p in parts:
            if p.isdigit() and hier_parts:
                hier_parts[-1] += "." + p
            else:
                hier_parts.append(p)

        return "/".join(hier_parts)

    def extract_submodule_structure(self, pytorch_model, function_module_types: Set[type]) -> Dict[str, Dict[str, str]]:
        """
        Extract the actual submodule structure from PyTorch model.

        Args:
            pytorch_model: Original PyTorch model
            function_module_types: Set of module types exported as functions

        Returns:
            Dictionary mapping instance_path -> {submodule_name: submodule_type}

        Example:
            {
                'model.layers.0': {
                    'self_attn': 'LlamaAttention',
                    'mlp': 'LlamaMLP',
                    'input_layernorm': 'LlamaRMSNorm',
                    'post_attention_layernorm': 'LlamaRMSNorm'
                },
                'model.layers.1': {...}
            }
        """
        submodule_structure = {}

        for name, module in pytorch_model.named_modules():
            if type(module) in function_module_types and name:
                # This is a function module instance (e.g., model.layers.0)
                submodules = {}

                for child_name, child_module in module.named_children():
                    submodules[child_name] = type(child_module).__name__

                submodule_structure[name] = submodules

        return submodule_structure

    def categorize_submodules(self, submodules: Dict[str, str]) -> Dict[str, str]:
        """
        Categorize submodules into functional types (attention, mlp, norm, etc.).

        Args:
            submodules: Dictionary of {submodule_name: submodule_type}

        Returns:
            Dictionary mapping category -> actual_submodule_name

        Example:
            Input: {'self_attn': 'LlamaAttention', 'mlp': 'LlamaMLP'}
            Output: {'attention': 'self_attn', 'mlp': 'mlp'}
        """
        categories = {}

        # Attention patterns
        attention_patterns = [
            "attn",
            "attention",
            "self_attn",
            "self_attention",
            "cross_attn",
            "cross_attention",
            "mha",
            "multihead",
        ]

        # MLP/FFN patterns
        mlp_patterns = ["mlp", "ffn", "feed_forward", "feedforward", "ff", "dense", "intermediate"]

        # Normalization patterns
        norm_patterns = ["norm", "layernorm", "layer_norm", "rmsnorm", "rms_norm", "ln", "normalization"]

        for submodule_name, submodule_type in submodules.items():
            name_lower = submodule_name.lower()
            type_lower = submodule_type.lower()
            combined = name_lower + " " + type_lower

            # Check attention
            if any(pattern in combined for pattern in attention_patterns):
                if "attention" not in categories:  # Use first match
                    categories["attention"] = submodule_name

            # Check MLP
            elif any(pattern in combined for pattern in mlp_patterns):
                if "mlp" not in categories:
                    categories["mlp"] = submodule_name

            # Check normalization (can have multiple)
            elif any(pattern in combined for pattern in norm_patterns):
                # Distinguish pre/post norms
                if "input" in name_lower or "pre" in name_lower or name_lower.endswith("1"):
                    categories["norm_pre_attn"] = submodule_name
                elif "post" in name_lower or name_lower.endswith("2"):
                    categories["norm_post_attn"] = submodule_name
                else:
                    # Generic norm
                    if "norm" not in categories:
                        categories["norm"] = submodule_name

        return categories

    def identify_attention_regions(self, nodes: List, window_size: int = 50) -> Set[int]:
        """
        Identify node indices that belong to attention mechanisms.
        Uses Softmax as anchor point.
        """
        attention_region = set()
        softmax_indices = [i for i, node in enumerate(nodes) if node.op_type == "Softmax"]

        for softmax_idx in softmax_indices:
            start = max(0, softmax_idx - window_size)
            end = min(len(nodes), softmax_idx + window_size)
            attention_region.update(range(start, end))

        return attention_region

    def identify_mlp_regions(self, nodes: List, attention_region: Set[int], window_size: int = 30) -> Set[int]:
        """
        Identify node indices that belong to MLP/FFN layers.
        Uses activation functions as indicators.
        """
        mlp_region = set()
        activation_ops = {"Gelu", "Relu", "Silu", "Tanh", "Swish"}

        activation_indices = [
            i for i, node in enumerate(nodes) if node.op_type in activation_ops and i not in attention_region
        ]

        for act_idx in activation_indices:
            start = max(0, act_idx - window_size)
            end = min(len(nodes), act_idx + window_size)
            mlp_region.update(range(start, end))

        return mlp_region

    def infer_node_module_with_structure(
        self,
        node,
        node_idx: int,
        instance_weight_to_module: Dict[str, str],
        tensor_to_module: Dict[str, str],
        tensor_is_constant: Dict[str, bool],
        attention_region: Set[int],
        mlp_region: Set[int],
        submodule_categories: Dict[str, str],
    ) -> str:
        """
        Infer which submodule a node belongs to using actual PyTorch structure.

        Args:
            node: ONNX node to analyze
            node_idx: Index of node in function
            instance_weight_to_module: Weight to module mapping
            tensor_to_module: Tensor ownership tracking
            tensor_is_constant: Constant tensor flags
            attention_region: Set of attention node indices
            mlp_region: Set of MLP node indices
            submodule_categories: Mapping of category -> actual submodule name

        Returns:
            Actual submodule name from PyTorch model
        """
        # Strategy 1: Direct weight usage (most reliable)
        weight_modules = {instance_weight_to_module[inp] for inp in node.input if inp in instance_weight_to_module}

        if weight_modules:
            return self.common_prefix(list(weight_modules)).rstrip(".")

        # Strategy 2: Pattern-based with actual names
        if node_idx in attention_region:
            # Use actual attention submodule name
            return submodule_categories.get("attention", "self_attn")

        if node_idx in mlp_region:
            # Use actual MLP submodule name
            return submodule_categories.get("mlp", "mlp")

        # Strategy 3: Inherit from data flow
        data_modules = {
            tensor_to_module[inp]
            for inp in node.input
            if inp in tensor_to_module and not tensor_is_constant.get(inp, False) and tensor_to_module[inp]
        }

        if data_modules:
            return self.common_prefix(list(data_modules)).rstrip(".")

        return ""

    def process_onnx_with_hierarchy(self, onnx_model, function_module_types: Set[type], pytorch_model):
        """
        Post-process ONNX model to inject hierarchical node names using actual PyTorch structure.

        This hybrid approach:
        1. Extracts actual submodule names from PyTorch model
        2. Uses pattern detection to identify functional regions
        3. Maps regions to actual submodule names

        Args:
            onnx_model: Loaded ONNX model (onnx.ModelProto)
            function_module_types: Set of PyTorch module types exported as functions
            pytorch_model: Original PyTorch model for reference

        Returns:
            Modified ONNX model with hierarchical node names using actual PyTorch names
        """
        # Extract actual submodule structure from PyTorch
        submodule_structure = self.extract_submodule_structure(pytorch_model, function_module_types)

        if not submodule_structure:
            return onnx_model

        if not onnx_model.functions:
            return onnx_model

        # Build weight-to-module mapping
        weight_to_module = {}
        for init in onnx_model.graph.initializer:
            weight_name = init.name
            for instance in submodule_structure.keys():
                if weight_name.startswith(instance + "."):
                    relative_key = weight_name[len(instance) + 1 :]
                    if "." in relative_key:
                        relative_module = relative_key.rsplit(".", 1)[0]
                        weight_to_module[weight_name] = (instance, relative_module)
                        break

        # Map function calls to module instances
        call_nodes_in_order = [
            node for node in onnx_model.graph.node if node.op_type in [f.name for f in onnx_model.functions]
        ]

        module_instances = list(submodule_structure.keys())
        call_to_instance = {
            call_node.name: module_instances[idx]
            for idx, call_node in enumerate(call_nodes_in_order)
            if idx < len(module_instances)
        }

        # Group calls by function name
        func_to_calls = defaultdict(list)
        for call_node in call_nodes_in_order:
            func_to_calls[call_node.op_type].append(call_node)

        # Process each function
        for func in onnx_model.functions:
            calls_for_this_func = func_to_calls.get(func.name, [])
            if not calls_for_this_func:
                continue

            representative_call = calls_for_this_func[0]
            representative_instance = call_to_instance.get(representative_call.name)
            if not representative_instance:
                continue

            # Get actual submodule structure for this instance
            instance_submodules = submodule_structure.get(representative_instance, {})
            submodule_categories = self.categorize_submodules(instance_submodules)

            # Build weight mapping for this instance
            instance_weight_to_module = {
                weight_name: rel_module
                for weight_name, (instance, rel_module) in weight_to_module.items()
                if instance == representative_instance
            }

            # Identify functional regions
            attention_region = self.identify_attention_regions(func.node)
            mlp_region = self.identify_mlp_regions(func.node, attention_region)

            # Initialize tensor tracking
            tensor_to_module = {inp: "" for inp in func.input}
            tensor_is_constant = {out: True for node in func.node if node.op_type == "Constant" for out in node.output}

            # Process each node
            for node_idx, node in enumerate(func.node):
                # Infer module using actual PyTorch structure
                node_module = self.infer_node_module_with_structure(
                    node,
                    node_idx,
                    instance_weight_to_module,
                    tensor_to_module,
                    tensor_is_constant,
                    attention_region,
                    mlp_region,
                    submodule_categories,
                )

                # Build full hierarchical path
                full_path = representative_instance
                if node_module:
                    full_path += "." + node_module

                hier_path = self.path_to_hierarchical(full_path)
                node.name = f"/{hier_path}/{node.op_type}"

                # Propagate module info
                for out in node.output:
                    tensor_to_module[out] = node_module

        return onnx_model

    @export_wrapper
    def _export(
        self,
        example_inputs: Dict[str, torch.Tensor],
        output_names: List[str],
        dynamic_axes: Dict[str, Dict[int, str]],
        onnx_transform_kwargs: Optional[Dict[str, any]] = None,
        export_dir: Optional[str] = None,
        offload_pt_weights: bool = True,
        prefill_only: Optional[bool] = False,
        **export_kwargs,
    ) -> str:
        """
        Export the PyTorch model to ONNX and apply ONNX transforms

        This method:
        1. Exports PyTorch model to ONNX using torch.onnx.export
        2. Clears PyTorch weights after export
        3. Applies ONNX transforms with reduced memory footprint

        Args:
            :example_inputs (dict): Sample inputs to trace the model.
            :output_names (list): names to assign to the output nodes of the graph, in order.
            :dynamic_axes (dict): Same as dynamic_axes parameter to be passed to `torch.onnx.export`.
            :export_kwargs (dict): Additional arguments to be passed to `torch.onnx.export`.
            :onnx_transform_kwargs (dict): Additional arguments to be passed to `Transform.apply` for this class.
            :export_dir (str): Specify the export directory. The export_dir will be suffixed with a hash corresponding to current model.
            :offload_pt_weights (bool): If True, offload PyTorch model weights to meta device
            after successful export to reduce memory usage. Set to False if you need to
            keep weights for further operations. Defaults to True.
            Note:
            Once weights are offloaded, the model cannot be re-exported. Create a new
            instance using from_pretrained() for re-export.

        """
        # TODO: Hack for retain_full_kv, handle this outside
        export_kwargs.pop("retain_full_kv", None)
        onnx_path = export_dir / f"{self.model_name}.onnx"

        # Return early if ONNX already exists
        if onnx_path.is_file():
            self.onnx_path = onnx_path
            return onnx_path

        # check if the model is in meta state or weights are offloaded
        self._model_offloaded_check()

        # Setup temporary paths
        tmp_onnx_dir = export_dir / "onnx_tmp"
        tmp_onnx_path = tmp_onnx_dir / f"{self.model_name}.onnx"
        tmp_onnx_dir.mkdir(parents=True, exist_ok=True)

        # Create input_names from example_inputs
        input_names = []
        for param in inspect.signature(self.model.forward).parameters:
            if param in example_inputs:
                if param == "past_key_values":
                    for i in range(len(example_inputs["past_key_values"])):
                        if len(example_inputs["past_key_values"][0]) == 2:
                            input_names.extend([f"past_key.{i}", f"past_value.{i}"])
                        elif len(example_inputs["past_key_values"][0]) == 4:
                            input_names.extend(
                                [
                                    f"past_key_self.{i}",
                                    f"past_value_self.{i}",
                                    f"past_key_cross.{i}",
                                    f"past_value_cross.{i}",
                                ]
                            )
                        else:
                            raise ValueError(
                                f"Unknown shape of past_key_values! Expected length of past_key_values for each layer to be either 2 or 4 but got {len(example_inputs['past_key_values'][0])}"
                            )
                else:
                    input_names.append(param)

        try:
            torch.onnx.export(
                self.model,
                (example_inputs,),
                str(tmp_onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=constants.ONNX_EXPORT_OPSET,
                **export_kwargs,
            )
            logger.info("PyTorch export successful")
            _ = self._offload_model_weights(offload_pt_weights)
            model = onnx.load(tmp_onnx_path, load_external_data=False)
            # model = self.process_onnx_with_hierarchy(model, export_kwargs["export_modules_as_functions"], self.model)
            if "export_modules_as_functions" in export_kwargs and export_kwargs["export_modules_as_functions"]:
                model = self.process_onnx_with_hierarchy(
                    model, export_kwargs["export_modules_as_functions"], self.model
                )

            # Clear temporary references
            transform_kwargs = {
                "onnx_base_dir": str(tmp_onnx_dir),
                "model_name": self.model_name,
            }
            if onnx_transform_kwargs is not None:
                transform_kwargs.update(onnx_transform_kwargs)

            onnx_transforms = OnnxTransformPipeline(transforms=self._onnx_transforms)
            model, transformed = onnx_transforms.apply(model, **transform_kwargs)

            # Add metadata to the model
            model.metadata_props.append(
                onnx.StringStringEntryProto(key="qeff_transforms", value=",".join(self._transform_names()))
            )
            logger.info("ONNX transforms applied")

            onnx.save(model, onnx_path)
            del model
            gc.collect()
            logger.info("Transformed ONNX saved")

        except Exception as e:
            logger.error(f"ONNX export or transforms failed: {e}")
            raise e

        finally:
            shutil.rmtree(tmp_onnx_dir, ignore_errors=True)

        self.onnx_path = onnx_path
        return onnx_path

    def get_onnx_path(
        self,
        prefill_only: Optional[bool] = False,
        enable_chunking: Optional[bool] = False,
        specializations: Optional[List[Dict[str, int]]] = None,
        offload_pt_weights: Optional[bool] = True,
        use_onnx_subfunctions: Optional[bool] = False,
        retain_full_kv: Optional[bool] = False,
    ):
        kwargs = {
            "offload_pt_weights": offload_pt_weights,
            "use_onnx_subfunctions": use_onnx_subfunctions,
            "retain_full_kv": retain_full_kv,
        }

        if prefill_only:
            kwargs.update(
                {
                    "prefill_only": prefill_only,
                    "prefill_seq_len": specializations[0].get("seq_len"),
                    "enable_chunking": enable_chunking,
                }
            )

        self.export(**kwargs)
        return self.onnx_path

    @dump_qconfig
    def _compile(
        self,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        mxint8_kv_cache: bool = False,
        specializations: Optional[List[Dict[str, int]]] = None,
        custom_io: Optional[Dict[str, str]] = None,
        mdp_ts_num_devices: int = 1,
        num_speculative_tokens: Optional[int] = None,
        enable_qnn: Optional[bool] = False,
        qnn_config: Optional[str] = None,
        use_onnx_subfunctions: bool = False,
        prefill_only: Optional[str] = None,
        offload_pt_weights: Optional[bool] = True,
        enable_chunking: Optional[bool] = False,
        retain_full_kv: Optional[bool] = None,
        **compiler_options,
    ) -> str:
        """
        Interface for qaic-exec compiler

        Args:
            :onnx_path (str): Onnx file to compile
            :compile_dir (str): Directory path to compile the qpc. A suffix is added to the directory path to avoid reusing same qpc for different parameters.
            :mxint8_kv_cache (bool, optional): Whether to use ``mxint8`` compression for KV cache. ``Defaults to False``.
            :specializations (list): List of specializations to compile for
            :custom_io (dict): Custom IO to specify the input and outputs in different formats than default
            :mdp_ts_num_devices (int): Number of devices to partition to use Multi-Device Partitioning with tensor-slicing.
            :num_speculative_tokens (int, optional): Number of speculative tokens to take as input for Speculative Decoding Target Language Model.
            :enable_qnn (bool): Enables QNN Compilation. ``Defaults to False.``
            :qnn_config (str): Path of QNN Config parameters file. Any extra parameters for QNN compilation can be passed via this file. ``Defaults to None.``
            :compiler_options: Pass any compiler option as input.
                Any flag that is supported by `qaic-exec` can be passed. Params are converted to flags as below:

                - aic_num_cores=16 -> -aic-num-cores=16
                - convert_to_fp16=True -> -convert-to-fp16
                - aic_hw_version=ai100 -> -aic-hw-version=ai100
                - aic_hw_version=ai200 -> -aic-hw-version=ai200

                For QNN Compilation path, when enable_qnn is set to True, any parameter passed in compiler_options will be ignored.
        """
        onnx_path = Path(
            onnx_path
            if onnx_path
            else self.onnx_path
            if self.onnx_path
            else self.get_onnx_path(
                prefill_only,
                enable_chunking,
                specializations,
                offload_pt_weights,
                use_onnx_subfunctions,
                retain_full_kv,
            )
        )
        compile_dir = Path(compile_dir or onnx_path.parent)
        qpc_path = compile_dir / "qpc"
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX file not found at: {onnx_path}")

        if enable_qnn:
            if compiler_options:
                logger.warning(
                    f"Extra arguments to QNN compilation are supported only via qnn_config file. Ignoring {compiler_options}"
                )

            self.qpc_path = qnn_compile(
                onnx_path=onnx_path,
                qpc_base_path=compile_dir,
                specializations=specializations,
                custom_io=custom_io,
                device_group=list(range(mdp_ts_num_devices)),
                num_cores=compiler_options.get("aic_num_cores", constants.DEFAULT_AIC_NUM_CORES),
                mxfp6=compiler_options.get("mxfp6_matmul", constants.DEFAULT_AIC_MXPF6_MATMUL),
                mxint8=mxint8_kv_cache,
                qnn_config=qnn_config,
            )

            return self.qpc_path

        command = (
            constants.COMPILER
            + [
                f"-aic-hw-version={compiler_options.pop('aic_hw_version', compiler_options.pop('aic-hw-version', constants.DEFAULT_AIC_HW_VERSION))}"
            ]
            + [f"-m={onnx_path}"]
        )

        # MDP partition config: prioritize dump over load
        mdp_dump_json_path = compiler_options.pop("mdp_dump_partition_config", None)
        mdp_ts_json_path = compiler_options.pop("mdp_load_partition_config", None)
        mdp_ts_json = None
        user_provided_load_config = False

        if mdp_dump_json_path:
            if mdp_ts_json_path:
                logger.warning(
                    "Loading and Dumping partition is not supported at the same time. Prioritizing dump config over load config!"
                )
            command.append(f"-mdp-dump-partition-config={mdp_dump_json_path}")
        elif mdp_ts_json_path:
            command.append(f"-mdp-load-partition-config={mdp_ts_json_path}")
            mdp_ts_json = load_json(str(mdp_ts_json_path))
            user_provided_load_config = True
        elif mdp_ts_num_devices > 1:
            # Generate mdp config only if neither dump nor load is provided and num_devices > 1
            mdp_ts_json = generate_mdp_partition_config(
                mdp_ts_num_devices, compiler_options.get("aic_num_cores", constants.DEFAULT_AIC_NUM_CORES)
            )

        for key, value in compiler_options.items():
            option = "-" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(option)
                continue
            command.append(f"{option}={value}")

        if use_onnx_subfunctions:
            logger.info("Using ONNX subfunctions for compilation.")
            command.append("-sub-functions")

        compile_hash_params = {
            "command": command,
            "specializations": specializations,
            "custom_io": custom_io,
            "mdp_ts_num_devices": mdp_ts_num_devices,
            "mdp_ts_json": mdp_ts_json,
            "num_speculative_tokens": num_speculative_tokens,
            "prefill_only": prefill_only,
        }
        compile_hash = hash_dict_params(compile_hash_params)

        compile_dir = qpc_path.with_name(qpc_path.name + "-" + compile_hash)
        qpc_path = compile_dir / "qpc"
        qpc_path.mkdir(parents=True, exist_ok=True)

        if qpc_path.is_dir():
            if (qpc_path / "programqpc.bin").is_file():
                self.qpc_path = qpc_path
                return qpc_path
            # Probably compilation failure last time, delete directory to start over
            shutil.rmtree(qpc_path)

        # Write the generated MDP partition config file (not if user provided it)
        if mdp_ts_json is not None and not user_provided_load_config:
            mdp_ts_json_path = compile_dir / f"mdp_ts_{mdp_ts_num_devices}.json"
            create_json(str(mdp_ts_json_path), mdp_ts_json)
            command.append(f"-mdp-load-partition-config={mdp_ts_json_path}")

        # Write specializations.json file
        if specializations is not None:
            specializations_json = compile_dir / "specializations.json"
            specializations_data = {
                "specializations": [{k: str(v) for k, v in spec.items()} for spec in specializations]
            }
            create_json(str(specializations_json), specializations_data)
            command.append(f"-network-specialization-config={specializations_json}")

        # Write custom_io.yaml file
        if custom_io is not None:
            custom_io_yaml = compile_dir / "custom_io.yaml"
            with open(custom_io_yaml, "w") as fp:
                for io_name, dtype in custom_io.items():
                    fp.write(f" - IOName: {io_name}\n   Precision: {dtype}\n\n")
            command.append(f"-custom-IO-list-file={custom_io_yaml}")

        command.append(f"-aic-binary-dir={qpc_path}")
        logger.info(f"Running compiler: {' '.join(command)}")

        try:
            subprocess.run(command, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "\n".join(
                    [
                        "Compilation failed!",
                        f"Compiler command: {e.cmd}",
                        f"Compiler exitcode: {e.returncode}",
                        "Compiler stderr:",
                        e.stderr.decode(),
                    ]
                )
            )
        # Dump JSON file with hashed parameters
        hashed_compile_params_path = compile_dir / "hashed_compile_params.json"
        create_json(hashed_compile_params_path, compile_hash_params)
        logger.info("Hashed parameters exported successfully.")

        self.qpc_path = qpc_path
        return qpc_path
