# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Runtime monkey patches for ONNX export compatibility."""

import importlib
import inspect
import re
import textwrap
from contextlib import contextmanager

import torch
import torch.onnx.utils as onnx_utils
from torch import _C
from torch._export import verifier as export_verifier
from torch._subclasses import functional_tensor as functional_tensor_mod
from torch.export import exported_program as exported_program_mod
from torch.onnx._internal.torchscript_exporter import utils as ts_utils

core_exporter_mod = importlib.import_module("torch.onnx._internal.exporter._core")

invoke_subgraph_mod = importlib.import_module("torch._higher_order_ops.invoke_subgraph")
hop_utils_mod = importlib.import_module("torch._higher_order_ops.utils")

# Store original references before patching
_original_setup_trace_module_map = onnx_utils._setup_trace_module_map
_original_get_module_attributes = getattr(onnx_utils, "_get_module_attributes", None)
_original_track_scope_attrs = getattr(_C, "_jit_pass_onnx_track_scope_attributes", None)
_original_ts_setup_trace_module_map = ts_utils._setup_trace_module_map
_original_ts_get_module_attributes = getattr(ts_utils, "_get_module_attributes", None)
_original_functional_tensor_dispatch = functional_tensor_mod.FunctionalTensorMode.__torch_dispatch__
_original_verify_exported_program_signature = export_verifier._verify_exported_program_signature
_original_exported_program_named_buffers = exported_program_mod.ExportedProgram.named_buffers
_original_invoke_subgraph_placeholder = invoke_subgraph_mod.invoke_subgraph_placeholder
_original_materialize_as_graph = hop_utils_mod.materialize_as_graph
_original_invoke_subgraph_gen_schema = invoke_subgraph_mod.InvokeSubgraphHOP.gen_schema
_original_translate_fx_graph = core_exporter_mod._translate_fx_graph
_original_convert_fx_arg_to_onnx_arg = core_exporter_mod._convert_fx_arg_to_onnx_arg

_PATCHES_ACTIVE = False
_MISSING_INSTANCE_ATTR = object()


def _setup_trace_module_map_patched(
    model,
    export_modules_as_functions,
):
    """Patched version of _setup_trace_module_map that fixes onnx_attrs type mismatch."""

    def __register_attribute_hook():
        attr_name = "_onnx_attrs"

        def _track_module_attributes_forward_pre_hook(module, input):
            setattr(module, attr_name, _get_module_attributes(module))

        def _track_module_attributes_forward_hook(module, input, output):
            tracing_state = _C._get_tracing_state()
            if not tracing_state:
                return
            graph = tracing_state.graph()
            onnx_attrs = {}
            if hasattr(module, attr_name):
                onnx_attrs = getattr(module, attr_name)
                delattr(module, attr_name)
            try:
                onnx_attrs = {}  # HACK: to reduce export time # TODO: study behaviour across models
                _C._jit_pass_onnx_track_scope_attributes(graph, onnx_attrs)
            except Exception:
                # Silently skip: scope-attribute tracking is best-effort and not required for export.
                pass

        for m in model.modules():
            m.register_forward_hook(_track_module_attributes_forward_hook)
            m.register_forward_pre_hook(_track_module_attributes_forward_pre_hook)

    def _unqualified_variable_name(qualified_name: str) -> str:
        name_atoms = qualified_name.split(".")
        for i, atom in reversed(list(enumerate(name_atoms))):
            if not atom.isnumeric():
                return ".".join(name_atoms[i:])
        return qualified_name

    trace_module_map = {
        _m: torch._C._jit_onnx_create_full_scope_name(torch.typename(type(_m)), _unqualified_variable_name(_n))
        for _n, _m in model.named_modules()
    }
    torch.jit._trace._trace_module_map = trace_module_map

    if isinstance(export_modules_as_functions, bool) and export_modules_as_functions:
        module_typenames = {torch.typename(type(module)) for module in trace_module_map}
    elif isinstance(export_modules_as_functions, set) and export_modules_as_functions:

        def _find_typename(v):
            if isinstance(v, type):
                return torch.typename(v)
            else:
                raise RuntimeError(
                    "Only type of the `nn.Module` should be passed in the set for argument `export_modules_as_functions`. "
                    f"Got `{type(v).__name__}`."
                )

        module_typenames = {_find_typename(v) for v in export_modules_as_functions}
    else:
        module_typenames = set()

    if module_typenames:
        __register_attribute_hook()

    return module_typenames


def _get_module_attributes(module):
    """Helper function to get module attributes safely."""
    import typing

    import torch.nn

    def _is_safe_value(value):
        if isinstance(value, (int, float, bool, str, torch.Tensor)) or value is None:
            return True
        if isinstance(value, (list, tuple)):
            return all(_is_safe_value(item) for item in value)
        return False

    annotations = typing.get_type_hints(type(module))
    base_m_annotations = typing.get_type_hints(torch.nn.Module)
    [annotations.pop(k, None) for k in base_m_annotations]

    attrs = {}
    for k in annotations:
        try:
            value = getattr(module, k)
            if _is_safe_value(value):
                attrs[k] = value
        except AttributeError:
            _C._jit_onnx_log(f"Skipping module attribute '{k}'")
            continue
    return attrs


def _track_scope_attributes_patched(graph, attrs):
    """Ensure scope attributes passed to ONNX are IValue-compatible."""
    safe_attrs = {}
    for key, value in attrs.items():
        if isinstance(value, (int, float, bool, str, torch.Tensor)) or value is None:
            safe_attrs[key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (int, float, bool, str, torch.Tensor)) or item is None for item in value
        ):
            safe_attrs[key] = value
    return _original_track_scope_attrs(graph, safe_attrs)


def _patch_function_from_source(owner, attr_name, rewrite_source):
    original = getattr(owner, attr_name)
    source = textwrap.dedent(inspect.getsource(original))
    patched_source = rewrite_source(source)
    if patched_source == source:
        return

    namespace = dict(inspect.getmodule(original).__dict__)
    namespace.update(original.__globals__)
    namespace.setdefault("torch", torch)
    exec("from __future__ import annotations\n" + patched_source, namespace)
    setattr(owner, attr_name, namespace[attr_name])


def _rewrite_functional_tensor_dispatch(source: str) -> str:
    if "skip\n                                continue" in source:
        return source
    old = """                            tracker_entry = m.tracer.tensor_tracker[unwrapped]
                            curr_node = tracker_entry.proxy.node
"""
    new = """                            try:
                                tracker_entry = m.tracer.tensor_tracker[unwrapped]
                            except KeyError:
                                # Lifted tensor constants can appear in nested HOP subgraph
                                # tracing without a tracker entry. In that case we can skip
                                # replay metadata sync for this tensor.
                                continue
                            curr_node = tracker_entry.proxy.node
"""
    if old in source:
        return source.replace(old, new, 1)
    old_legacy = """                            try:
                                tracker_entry = m.tracer.tensor_tracker[unwrapped]
                            except KeyError:
                                raise RuntimeError(
                                    f"cannot find {unwrapped} in tensor_tracker"
                                ) from None
                            curr_node = tracker_entry.proxy.node
"""
    if old_legacy in source:
        return source.replace(old_legacy, new, 1)
    # PyTorch >= 2.9: tensor_tracker no longer exists in this function;
    # the issue this patch addressed has been resolved upstream.
    return source


def _rewrite_verify_exported_program_signature(source: str) -> str:
    if "is_nested_lifted_constant" in source:
        return source
    pattern = re.compile(
        r"""(?P<indent>\s*)if \(\n"""
        r"""(?P=indent)\s*input_spec\.persistent is True\n"""
        r"""(?P=indent)\s*and buffer not in exported_program\.state_dict\n"""
        r"""(?P=indent)\s*\):\n"""
        r"""(?P=indent)\s*raise SpecViolationError\(\s*(?:\n(?P=indent)\s*)?f"Buffer \{buffer\} is not in the state dict\."\s*(?:\n(?P=indent)\s*)?\)\n""",
        re.MULTILINE,
    )
    match = pattern.search(source)
    if match is None:
        raise RuntimeError("Unable to patch _verify_exported_program_signature")
    indent = match.group("indent")
    replacement = f"""{indent}if (
{indent}    input_spec.persistent is True
{indent}    and buffer not in exported_program.state_dict
{indent}):
{indent}    # Nested invoke_subgraph tracing can materialize lifted tensor
{indent}    # constants as repeated_subgraph-local buffers (e.g.
{indent}    # repeated_subgraph0._tensor_constant0). These buffers are
{indent}    # owned by the subgraph module and are not persisted in the
{indent}    # top-level state_dict.
{indent}    is_nested_lifted_constant = (
{indent}        buffer.startswith("repeated_subgraph")
{indent}        and "._tensor_constant" in buffer
{indent}    )
{indent}    if not is_nested_lifted_constant:
{indent}        raise SpecViolationError(
{indent}            f"Buffer {{buffer}} is not in the state dict."
{indent}        )
"""
    return source[: match.start()] + replacement + source[match.end() :]


def _rewrite_exported_program_named_buffers(source: str) -> str:
    if "elif buffer_name in self.constants" in source:
        return source
    old = """        else:
            yield buffer_name, self.state_dict[buffer_name]
"""
    new = """        else:
            if buffer_name in self.state_dict:
                yield buffer_name, self.state_dict[buffer_name]
            elif buffer_name in self.constants:
                # Nested HOP-lifted tensor constants may be represented as
                # buffers in graph signature but stored in constants.
                yield buffer_name, self.constants[buffer_name]
"""
    if old not in source:
        raise RuntimeError("Unable to patch ExportedProgram.named_buffers")
    return source.replace(old, new, 1)


def _rewrite_invoke_subgraph_placeholder(source: str) -> str:
    if "_invoke_subgraph_placeholder_wrapper(func, args, kwargs)" in source:
        return source
    old = """        def _invoke_subgraph_placeholder_wrapper(func, args):
            return invoke_subgraph_placeholder(func, *args)
"""
    new = """        def _invoke_subgraph_placeholder_wrapper(func, args, kwargs):
            return invoke_subgraph_placeholder(func, *args, **kwargs)
"""
    if old not in source:
        raise RuntimeError("Unable to patch invoke_subgraph_placeholder wrapper signature")
    source = source.replace(old, new, 1)

    old = """            )(func, args)
"""
    new = """            )(func, args, kwargs)
"""
    if old not in source:
        raise RuntimeError("Unable to patch invoke_subgraph_placeholder wrapper callsite")
    return source.replace(old, new, 1)


def _rewrite_materialize_as_graph(source: str) -> str:
    if "contains_functional_tensor = any(" in source:
        return source
    old = """        with suspend_functionalization(), disable_functional_mode():
            fake_mode = None
"""
    new = """        contains_functional_tensor = any(
            isinstance(arg, FunctionalTensor) for arg in pytree.tree_leaves(args)
        )
        functional_mode_context = (
            contextlib.nullcontext() if contains_functional_tensor else disable_functional_mode()
        )
        with suspend_functionalization(), functional_mode_context:
            fake_mode = None
"""
    if old not in source:
        # PyTorch >= 2.9: materialize_as_graph has been rewritten and no longer
        # contains the pattern this patch targeted; no patching needed.
        return source
    return source.replace(old, new, 1)


def _rewrite_invoke_subgraph_gen_schema(source: str) -> str:
    if "gm: torch.fx.GraphModule | None = None" in source:
        return source
    pattern = re.compile(
        r"""(?P<indent>\s*)gm: torch\.fx\.GraphModule = materialize_as_graph\(\n"""
        r"""(?P=indent)\s*subgraph, operands, subgraph_decomp_table=subgraph_decomp_table\n"""
        r"""(?P=indent)\)\n""",
        re.MULTILINE,
    )
    match = pattern.search(source)
    if match is None:
        # PyTorch >= 2.9: gen_schema has been rewritten to already handle GraphModule
        # subgraphs directly; no patching needed.
        return source
    indent = match.group("indent")
    replacement = f"""{indent}gm: torch.fx.GraphModule | None = None
{indent}if isinstance(subgraph, torch.fx.GraphModule):
{indent}    gm = subgraph
{indent}elif isinstance(subgraph, FunctionalizeCtxWrapper):
{indent}    gm = subgraph.subgraph
{indent}if gm is None:
{indent}    gm = materialize_as_graph(
{indent}        subgraph, operands, subgraph_decomp_table=subgraph_decomp_table
{indent}    )
"""
    return source[: match.start()] + replacement + source[match.end() :]


def _rewrite_translate_fx_graph(source: str) -> str:
    if "if isinstance(node.target, str) and node.target not in owned_graphs:" in source:
        return source
    old = """            elif node.op == "get_attr":
                _handle_get_attr_node(
                    node,
                    owned_graphs=owned_graphs,
                    node_name_to_local_functions=node_name_to_local_functions,
                )
"""
    new = """            elif node.op == "get_attr":
                if isinstance(node.target, str) and node.target not in owned_graphs:
                    tensor_value = node.meta.get("val", None)
                    if tensor_value is None:
                        tensor_value = node.meta.get("example_value", None)
                    if not isinstance(tensor_value, torch.Tensor):
                        raise KeyError(node.target)
                    value = ir.Value(name=node.name)
                    value.const_value = TorchTensor(tensor_value, name=node.name)
                    _set_shape_type(value, tensor_value, complex_to_float=lower != "none")
                    model.graph.initializers[node.name] = value
                    node_name_to_values[node.name] = value
                else:
                    _handle_get_attr_node(
                        node,
                        owned_graphs=owned_graphs,
                        node_name_to_local_functions=node_name_to_local_functions,
                    )
"""
    if old not in source:
        raise RuntimeError("Unable to patch _translate_fx_graph get_attr handling")
    return source.replace(old, new, 1)


def _rewrite_convert_fx_arg_to_onnx_arg(source: str) -> str:
    if "if arg.name in node_name_to_values:" in source:
        return source
    old = """        if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
            return node_name_to_local_functions[arg.name]
"""
    new = """        if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
            if arg.name in node_name_to_values:
                return node_name_to_values[arg.name]
            return node_name_to_local_functions[arg.name]
"""
    if old not in source:
        raise RuntimeError("Unable to patch _convert_fx_arg_to_onnx_arg get_attr fallback")
    return source.replace(old, new, 1)


def apply_torch_patches():
    """Apply monkey patches for ONNX export."""
    global _PATCHES_ACTIVE
    if _PATCHES_ACTIVE:
        return

    onnx_utils._setup_trace_module_map = _setup_trace_module_map_patched
    if hasattr(onnx_utils, "_get_module_attributes"):
        onnx_utils._get_module_attributes = _get_module_attributes
    ts_utils._setup_trace_module_map = _setup_trace_module_map_patched
    if hasattr(ts_utils, "_get_module_attributes"):
        ts_utils._get_module_attributes = _get_module_attributes
    if _original_track_scope_attrs is not None:
        _C._jit_pass_onnx_track_scope_attributes = _track_scope_attributes_patched
    _patch_function_from_source(
        functional_tensor_mod.FunctionalTensorMode,
        "__torch_dispatch__",
        _rewrite_functional_tensor_dispatch,
    )
    _patch_function_from_source(
        export_verifier,
        "_verify_exported_program_signature",
        _rewrite_verify_exported_program_signature,
    )
    _patch_function_from_source(
        exported_program_mod.ExportedProgram,
        "named_buffers",
        _rewrite_exported_program_named_buffers,
    )
    _patch_function_from_source(
        invoke_subgraph_mod,
        "invoke_subgraph_placeholder",
        _rewrite_invoke_subgraph_placeholder,
    )
    _patch_function_from_source(
        hop_utils_mod,
        "materialize_as_graph",
        _rewrite_materialize_as_graph,
    )
    _patch_function_from_source(
        invoke_subgraph_mod.InvokeSubgraphHOP,
        "gen_schema",
        _rewrite_invoke_subgraph_gen_schema,
    )
    _patch_function_from_source(
        core_exporter_mod,
        "_translate_fx_graph",
        _rewrite_translate_fx_graph,
    )
    _patch_function_from_source(
        core_exporter_mod,
        "_convert_fx_arg_to_onnx_arg",
        _rewrite_convert_fx_arg_to_onnx_arg,
    )
    _PATCHES_ACTIVE = True


def undo_torch_patches():
    """Undo monkey patches and restore original functions."""
    global _PATCHES_ACTIVE
    if not _PATCHES_ACTIVE:
        return

    onnx_utils._setup_trace_module_map = _original_setup_trace_module_map
    if _original_get_module_attributes:
        onnx_utils._get_module_attributes = _original_get_module_attributes
    ts_utils._setup_trace_module_map = _original_ts_setup_trace_module_map
    if _original_ts_get_module_attributes:
        ts_utils._get_module_attributes = _original_ts_get_module_attributes
    if _original_track_scope_attrs is not None:
        _C._jit_pass_onnx_track_scope_attributes = _original_track_scope_attrs
    functional_tensor_mod.FunctionalTensorMode.__torch_dispatch__ = _original_functional_tensor_dispatch
    export_verifier._verify_exported_program_signature = _original_verify_exported_program_signature
    exported_program_mod.ExportedProgram.named_buffers = _original_exported_program_named_buffers
    invoke_subgraph_mod.invoke_subgraph_placeholder = _original_invoke_subgraph_placeholder
    hop_utils_mod.materialize_as_graph = _original_materialize_as_graph
    invoke_subgraph_mod.InvokeSubgraphHOP.gen_schema = _original_invoke_subgraph_gen_schema
    core_exporter_mod._translate_fx_graph = _original_translate_fx_graph
    core_exporter_mod._convert_fx_arg_to_onnx_arg = _original_convert_fx_arg_to_onnx_arg
    _PATCHES_ACTIVE = False


@contextmanager
def temporarily_enable_nested_compile_regions(model, target_classes=None):
    """
    Wrap selected module ``forward`` methods with ``nested_compile_region``
    during export so repeated block functions are materialized by dynamo.
    """

    target_classes = tuple(target_classes) if target_classes else None
    patched_modules = []

    try:
        for module in model.modules():
            if target_classes and not isinstance(module, target_classes):
                continue

            bound_forward = getattr(module, "forward", None)
            if bound_forward is None:
                continue

            wrapped_forward = getattr(bound_forward, "__func__", bound_forward)
            if getattr(wrapped_forward, "__qualname__", "") == "mark_compile_region.<locals>.wrap.<locals>.inner":
                continue

            previous_forward = module.__dict__.get("forward", _MISSING_INSTANCE_ATTR)
            nested_forward = torch.compiler.nested_compile_region(wrapped_forward)
            setattr(module, "forward", nested_forward.__get__(module, type(module)))
            patched_modules.append((module, previous_forward))

        yield
    finally:
        for module, previous_forward in reversed(patched_modules):
            if previous_forward is _MISSING_INSTANCE_ATTR:
                delattr(module, "forward")
            else:
                setattr(module, "forward", previous_forward)


@contextmanager
def temporarily_disable_nested_compile_regions(model, target_classes=None):
    """
    Replace nested_compile_region-wrapped ``forward`` methods with their original
    underlying functions for the duration of plain dynamo export.
    """

    target_classes = tuple(target_classes) if target_classes else None
    patched_modules = []

    try:
        for module in model.modules():
            if target_classes and not isinstance(module, target_classes):
                continue

            bound_forward = getattr(module, "forward", None)
            if bound_forward is None:
                continue

            wrapped_forward = getattr(bound_forward, "__func__", bound_forward)
            if getattr(wrapped_forward, "__qualname__", "") != "mark_compile_region.<locals>.wrap.<locals>.inner":
                continue

            closure = getattr(wrapped_forward, "__closure__", None) or ()
            original_forward = next(
                (cell.cell_contents for cell in closure if inspect.isfunction(cell.cell_contents)),
                None,
            )
            if original_forward is None:
                continue

            previous_forward = module.__dict__.get("forward", _MISSING_INSTANCE_ATTR)
            setattr(module, "forward", original_forward.__get__(module, type(module)))
            patched_modules.append((module, previous_forward))

        yield
    finally:
        for module, previous_forward in reversed(patched_modules):
            if previous_forward is _MISSING_INSTANCE_ATTR:
                delattr(module, "forward")
            else:
                setattr(module, "forward", previous_forward)
