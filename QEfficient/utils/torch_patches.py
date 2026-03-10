# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""Monkey patches for torch to fix ONNX export and export correctness issues."""

import os

import torch
import torch._export.verifier as _verifier_module
import torch.export.exported_program as _ep_module
import torch.onnx.utils as onnx_utils
from torch import _C
from torch.onnx._internal.torchscript_exporter import utils as ts_utils

# ---------------------------------------------------------------------------
# Store original references before patching
# ---------------------------------------------------------------------------
_original_setup_trace_module_map = onnx_utils._setup_trace_module_map
_original_get_module_attributes = getattr(onnx_utils, "_get_module_attributes", None)
_original_track_scope_attrs = getattr(_C, "_jit_pass_onnx_track_scope_attributes", None)
_original_ts_setup_trace_module_map = ts_utils._setup_trace_module_map
_original_ts_get_module_attributes = getattr(ts_utils, "_get_module_attributes", None)


# ---------------------------------------------------------------------------
# Patch 1: ONNX – _setup_trace_module_map
# ---------------------------------------------------------------------------
def _setup_trace_module_map_patched(model, export_modules_as_functions):
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
            # FIX: use empty dict to avoid type mismatch
            onnx_attrs = {}
            _C._jit_pass_onnx_track_scope_attributes(graph, onnx_attrs)

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
                    "Only type of the `nn.Module` should be passed in the set for argument "
                    f"`export_modules_as_functions`. Got `{type(v).__name__}`."
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


# ---------------------------------------------------------------------------
# Patch 2: functional_tensor.py – FunctionalTensorMode.__torch_dispatch__
#
# The only change vs upstream: when iterating over args to sync view-replay
# metadata, a KeyError on tensor_tracker now does `continue` instead of
# raising RuntimeError. This handles lifted tensor constants in nested HOP
# subgraph tracing that have no tracker entry.
# ---------------------------------------------------------------------------
def _patch_functional_tensor_mode_dispatch():
    """
    Patch FunctionalTensorMode.__torch_dispatch__ so that a missing
    tensor_tracker entry for a FunctionalTensor causes a `continue`
    (skip) rather than a RuntimeError.

    We do this by wrapping the proxy-mode tensor_tracker with a safe
    proxy that returns None for missing keys, then calling the original
    implementation.  The original code checks `tracker_entry.proxy.node`
    so returning None would crash; instead we temporarily replace the
    tracker with one that silently skips missing keys by catching KeyError
    inside the loop.

    The cleanest approach is to directly patch the module-level source of
    the bug: we replace __torch_dispatch__ with a version that has the
    `continue` fix inlined.
    """
    import warnings

    import torch.fx.traceback as fx_traceback
    import torch.utils._pytree as pytree
    from torch._ops import TorchBindOpOverload
    from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode
    from torch.utils._python_dispatch import (
        autograd_would_have_decomposed,
        return_and_correct_aliasing,
    )

    not_implemented_log = torch._logging.getArtifactLogger("torch._subclasses.functional_tensor", "not_implemented")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: N807
        if kwargs is None:
            kwargs = {}

        unrecognized_types = [
            t
            for t in types
            if not issubclass(t, torch._subclasses.FakeTensor) and t not in [torch.Tensor, FunctionalTensor]
        ]
        if unrecognized_types:
            not_implemented_log.debug("FunctionalTensor unrecognized subclass(es): %s", unrecognized_types)
            return NotImplemented

        def _can_decompose(func):
            if self.export and func is torch.ops.aten.dropout.default:
                return False
            from torch._decomp import _should_decompose_because_unsafe_op

            if _should_decompose_because_unsafe_op(func):
                return True
            alias_info_present = any(arg.alias_info for arg in func._schema.arguments)
            if alias_info_present or func._schema.is_mutable:
                return True
            if self.export:
                if self.pre_dispatch:
                    if func.namespace not in ["aten", "prim"] and func._can_decompose():
                        warnings.warn(
                            f"At pre-dispatch tracing, we assume that any custom op marked with "
                            f"CompositeImplicitAutograd and have functional schema are safe to not decompose. "
                            f"Found {func} to be one such op.",
                            stacklevel=2,
                        )
                    return False
                return True
            flat_args_kwargs, _ = pytree.tree_flatten((args, kwargs))
            return autograd_would_have_decomposed(func, flat_args_kwargs)

        if (
            func not in FunctionalTensor.metadata_fns
            and _can_decompose(func)
            and torch._C._dispatch_has_kernel(func.name())
        ):
            with self:
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return r

        def wrap(x):
            if isinstance(x, FunctionalTensor):
                raise AssertionError("x must not be a FunctionalTensor in wrap()")
            if isinstance(x, torch.Tensor) and torch._is_functional_tensor(x):
                return FunctionalTensor(x, self)
            return x

        def unwrap(x):
            return x.elem

        from torch._higher_order_ops.auto_functionalize import (
            can_auto_functionalize,
            do_auto_functionalize,
            do_auto_functionalize_v2,
        )

        if can_auto_functionalize(func) and not torch._C._dispatch_has_kernel_for_dispatch_key(
            func.name(), torch._C.DispatchKey.Functionalize
        ):
            import torch._export.config as export_config
            import torch._inductor.config as inductor_config

            if torch.compiler.is_exporting():
                if export_config.enable_auto_functionalized_v2_for_export:
                    return do_auto_functionalize_v2(self, func, args, kwargs)
                return do_auto_functionalize(self, func, args, kwargs)
            if inductor_config.enable_auto_functionalized_v2:
                return do_auto_functionalize_v2(self, func, args, kwargs)
            return do_auto_functionalize(self, func, args, kwargs)

        from torch._higher_order_ops.effects import handle_effects, has_effects

        if has_effects(func):
            if torch._C._dispatch_has_kernel_for_dispatch_key(func.name(), torch._C.DispatchKey.Functionalize):
                raise AssertionError(
                    f"func {func.name()} with effects should not have a kernel for Functionalize dispatch key"
                )
            return handle_effects(self._allow_token_discovery, self._tokens, func, args, kwargs)

        args_unwrapped, kwargs_unwrapped = pytree.tree_map_only(FunctionalTensor, unwrap, (args, kwargs))

        is_included = torch._C._dispatch_tls_is_dispatch_key_included(torch._C.DispatchKey.Functionalize)
        is_excluded = torch._C._dispatch_tls_is_dispatch_key_excluded(torch._C.DispatchKey.Functionalize)
        if not is_excluded and is_included:
            raise AssertionError("Functionalization should not already be enabled above this mode")

        include_to_set = torch._C._dispatch_tls_local_include_set() | torch._C.DispatchKeySet(
            torch._C.DispatchKey.Functionalize
        )
        exclude_to_set = (
            torch._C._dispatch_tls_local_exclude_set().remove(torch._C.DispatchKey.Functionalize)
            - FunctionalTensor._extra_dispatch_keys
        )

        if isinstance(func, TorchBindOpOverload):
            from torch._subclasses.functional_tensor import PythonFunctionalizeAPI

            ctx = PythonFunctionalizeAPI()
            fully_unwrapped_args = ctx.unwrap_tensors(args)
            fully_unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
            outs_unwrapped = func(*fully_unwrapped_args, **fully_unwrapped_kwargs)
            outs_wrapped = ctx.wrap_tensors(outs_unwrapped)
        else:
            with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
                try:
                    old_apply_views = torch._functionalize_enable_reapply_views(True)
                    if func in FunctionalTensor.metadata_fns:
                        outs_unwrapped = func(*args_unwrapped, **kwargs_unwrapped)
                        outs_wrapped = pytree.tree_map_only(torch.Tensor, wrap, outs_unwrapped)
                    else:
                        if m := torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY):
                            for a in pytree.tree_leaves([args, kwargs]):
                                if not isinstance(a, FunctionalTensor):
                                    continue
                                unwrapped = torch._from_functional_tensor(a.elem)
                                try:
                                    tracker_entry = m.tracer.tensor_tracker[unwrapped]
                                except KeyError:
                                    # FIX: Lifted tensor constants can appear in nested
                                    # HOP subgraph tracing without a tracker entry.
                                    # In that case we can skip replay metadata sync
                                    # for this tensor.
                                    continue
                                curr_node = tracker_entry.proxy.node
                                with fx_traceback.set_current_replay_node(curr_node):
                                    torch._sync(a)

                        outs_unwrapped = func._op_dk(
                            torch._C.DispatchKey.Functionalize,
                            *args_unwrapped,
                            **kwargs_unwrapped,
                        )
                        if self.export:
                            if func is torch.ops.aten.dropout.default:
                                torch._freeze_functional_tensor(outs_unwrapped)
                        outs_wrapped = pytree.tree_map_only(torch.Tensor, wrap, outs_unwrapped)
                finally:
                    torch._disable_functionalization()
                    torch._functionalize_enable_reapply_views(old_apply_views)

        is_included = torch._C._dispatch_tls_is_dispatch_key_included(torch._C.DispatchKey.Functionalize)
        is_excluded = torch._C._dispatch_tls_is_dispatch_key_excluded(torch._C.DispatchKey.Functionalize)
        if not is_excluded and is_included:
            raise AssertionError("Functionalization should not already be enabled above this mode after dispatch")

        if (
            not any(isinstance(x, FunctionalTensor) for x in pytree.tree_leaves(outs_wrapped))
            or func is torch.ops.aten.lift_fresh.default
        ):
            return outs_wrapped

        if torch.Tag.inplace_view in func.tags and func is not torch.ops.aten.set_.source_Tensor:
            with torch.utils._mode_utils.no_dispatch():
                func(*args, **kwargs)

        return return_and_correct_aliasing(func, args, kwargs, outs_wrapped)

    FunctionalTensorMode.__torch_dispatch__ = __torch_dispatch__


# ---------------------------------------------------------------------------
# Patch 3: verifier.py – _verify_exported_program_signature
#
# Allow persistent buffers whose name matches the nested repeated_subgraph
# lifted-constant pattern to be absent from state_dict.
# ---------------------------------------------------------------------------
def _patched_verify_exported_program_signature(exported_program):
    from torch._export.verifier import SpecViolationError
    from torch.export.graph_signature import (
        CustomObjArgument,
        InputKind,
        SymBoolArgument,
        SymFloatArgument,
        SymIntArgument,
        TensorArgument,
        TokenArgument,
    )

    gs = exported_program.graph_signature

    input_node_names = [node.name for node in exported_program.graph.nodes if node.op == "placeholder"]
    if len(input_node_names) != len(gs.input_specs):
        input_spec_names = [spec.arg.name for spec in gs.input_specs if hasattr(spec.arg, "name")]
        missing_in_specs = set(input_node_names) - set(input_spec_names)
        missing_in_graph = set(input_spec_names) - set(input_node_names)
        raise SpecViolationError(
            f"Number of graph inputs ({len(input_node_names)}) "
            f"does not match number of inputs in the graph signature ({len(gs.input_specs)})\n"
            f"Placeholders missing input_specs: {missing_in_specs}\n"
            f"Input_specs missing placeholders: {missing_in_graph}"
        )

    for input_spec, node in zip(gs.input_specs, input_node_names):
        if isinstance(
            input_spec.arg,
            (TensorArgument, SymIntArgument, SymFloatArgument, SymBoolArgument),
        ):
            if input_spec.arg.name != node:
                raise SpecViolationError(f"Input spec name {input_spec.arg.name} does not match node name {node}")

        if input_spec.kind == InputKind.USER_INPUT:
            continue
        elif input_spec.kind == InputKind.PARAMETER:
            if not isinstance(input_spec.arg, TensorArgument):
                raise SpecViolationError(
                    f"Parameter {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead."
                )
            if input_spec.target is None:
                raise SpecViolationError(f"InputSpec for {input_spec.name} has no target.")
            param = input_spec.target
            if param not in exported_program.state_dict:
                raise SpecViolationError(f"Parameter {param} is not in the state dict.")
            if not isinstance(exported_program.state_dict[param], torch.nn.Parameter):
                raise SpecViolationError(
                    f"State dict entry for parameter {param} is not an instance of torch.nn.Parameter."
                )
        elif input_spec.kind == InputKind.BUFFER:
            if not isinstance(input_spec.arg, TensorArgument):
                raise SpecViolationError(
                    f"Buffer {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead."
                )
            if input_spec.target is None:
                raise SpecViolationError(f"InputSpec for {input_spec.name} has no target.")
            buffer = input_spec.target
            if input_spec.persistent is None:
                raise SpecViolationError(f"Buffer {buffer} is missing a persistence flag")
            if input_spec.persistent is True and buffer not in exported_program.state_dict:
                # FIX: Nested invoke_subgraph tracing can materialize lifted tensor
                # constants as repeated_subgraph-local buffers (e.g.
                # repeated_subgraph0._tensor_constant0). These buffers are owned by
                # the subgraph module and are not persisted in the top-level state_dict.
                is_nested_lifted_constant = buffer.startswith("repeated_subgraph") and "._tensor_constant" in buffer
                if not is_nested_lifted_constant:
                    raise SpecViolationError(f"Buffer {buffer} is not in the state dict.")
            if input_spec.persistent is False and buffer in exported_program.state_dict:
                raise SpecViolationError(f"Non-persistent buffer {buffer} is in the state dict, it should not be.")
        elif input_spec.kind == InputKind.CONSTANT_TENSOR:
            if not isinstance(input_spec.arg, TensorArgument):
                raise SpecViolationError(
                    f"Constant tensor {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead."
                )
            if input_spec.target is None:
                raise SpecViolationError(f"InputSpec for {input_spec.name} has no target.")
            tensor_const = input_spec.target
            if tensor_const not in exported_program.constants:
                raise SpecViolationError(f"Constant tensor {tensor_const} is not in the constants dictionary.")
        elif input_spec.kind == InputKind.CUSTOM_OBJ:
            if not isinstance(input_spec.arg, CustomObjArgument):
                raise SpecViolationError(
                    f"Custom object {input_spec.name} is not a custom object argument. Found {input_spec.arg} instead."
                )
            if input_spec.target is None:
                raise SpecViolationError(f"InputSpec for {input_spec.name} has no target.")
            custom_obj = input_spec.target
            if custom_obj not in exported_program.constants:
                raise SpecViolationError(f"Custom object {custom_obj} is not in the constants dictionary.")
        elif input_spec.kind == InputKind.TOKEN:
            if not isinstance(input_spec.arg, TokenArgument):
                raise SpecViolationError(
                    f"Constant tensor {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead."
                )
        else:
            raise SpecViolationError(f"Unknown InputKind {input_spec.kind}.")

    output_node = list(exported_program.graph.nodes)[-1]
    if output_node.op != "output":
        raise AssertionError(f"last node must be output, got {output_node.op}")
    output_nodes = [arg.name if isinstance(arg, torch.fx.Node) else arg for arg in output_node.args[0]]

    if len(output_nodes) != len(gs.output_specs):
        output_spec_names = [spec.arg.name if hasattr(spec.arg, "name") else str(spec.arg) for spec in gs.output_specs]
        missing_out_specs = set(output_nodes) - set(output_spec_names)
        missing_out_graph = set(output_spec_names) - set(output_nodes)
        raise SpecViolationError(
            f"Number of output nodes {len(output_nodes)} is different "
            f"Than the number of outputs specified by the graph signature: "
            f"{len(gs.output_specs)}\n"
            f"Nodes missing output_specs: {missing_out_specs}\n"
            f"Output_specs missing nodes: {missing_out_graph}"
        )

    num_tokens = len(gs.output_tokens)
    end = len(gs.buffers_to_mutate) + len(gs.parameters_to_mutate) + len(gs.user_inputs_to_mutate) + num_tokens
    mutate_nodes = output_nodes[num_tokens:end]
    user_output_nodes = output_nodes[end : end + len(gs.user_outputs)]

    for mutation_node in mutate_nodes:
        if mutation_node in gs.buffers_to_mutate:
            if gs.buffers_to_mutate[mutation_node] not in gs.buffers:
                raise SpecViolationError(
                    f"Buffer output {mutation_node} does not point to a buffer that exists. \n"
                    f"Dict of buffers that are mutated, in order: {gs.buffers_to_mutate} \n"
                    f"Buffer nodes available: {gs.buffers} \n"
                )
        elif mutation_node in gs.parameters_to_mutate:
            if gs.parameters_to_mutate[mutation_node] not in gs.parameters:
                raise SpecViolationError(
                    f"Parameter output {mutation_node} does not point to a parameter that exists. \n"
                    f"Dict of parameters that are mutated, in order: {gs.parameters_to_mutate} \n"
                    f"Parameter nodes available: {gs.parameters} \n"
                )
        elif mutation_node in gs.user_inputs_to_mutate:
            if gs.user_inputs_to_mutate[mutation_node] not in gs.user_inputs:
                raise SpecViolationError(
                    f"User input output {mutation_node} does not point to a user input that exists. \n"
                    f"Dict of user inputs that are mutated, in order: {gs.user_inputs_to_mutate} \n"
                    f"User input nodes available: {gs.user_inputs} \n"
                )
        else:
            raise SpecViolationError(
                f"Mutation node {mutation_node} is neither a buffer nor a user input. "
                f"Buffers to mutate: {gs.buffers_to_mutate}, "
                f"User inputs to mutate: {gs.user_inputs_to_mutate}"
            )

    for user_output_node, user_output_name in zip(user_output_nodes, gs.user_outputs):
        if user_output_node != user_output_name:
            raise SpecViolationError(
                f"User output {user_output_node} is not in the correct "
                "order or is not found in the "
                f"exported program's user_output list: {gs.user_outputs}. "
            )


# ---------------------------------------------------------------------------
# Patch 4: exported_program.py – ExportedProgram.named_buffers
#
# Fall back to self.constants when a buffer is not in self.state_dict, to
# handle nested HOP-lifted tensor constants stored as buffers in the
# graph signature but not in state_dict.
# ---------------------------------------------------------------------------
def _patched_named_buffers(self):
    non_persistent_buffers = set(self.graph_signature.non_persistent_buffers)
    for buffer_name in self.graph_signature.buffers:
        if buffer_name in non_persistent_buffers:
            yield buffer_name, self.constants[buffer_name]
        else:
            if buffer_name in self.state_dict:
                yield buffer_name, self.state_dict[buffer_name]
            elif buffer_name in self.constants:
                # FIX: Nested HOP-lifted tensor constants may be represented as
                # buffers in graph signature but stored in constants.
                yield buffer_name, self.constants[buffer_name]


# ---------------------------------------------------------------------------
# Patch 5: invoke_subgraph.py – three targeted fixes
#
# (a) invoke_subgraph_placeholder: wrapper uses *args/**kwargs instead of
#     a positional `args` list, so kwargs are forwarded correctly.
# (b) py_functionalize_impl: HopInstance.create wrapped in try/except with
#     env-var-gated fallback; invoke_subgraph call also try/except with
#     env-var-gated direct subgraph fallback.
# (c) ProxyTorchDispatchMode impl: insert_deferred_runtime_asserts gated by
#     TORCH_INVOKE_SKIP_DEFERRED_ASSERTS env var.
# ---------------------------------------------------------------------------
def _patch_invoke_subgraph():
    import torch._higher_order_ops.invoke_subgraph as _isg_mod

    # ---- (a) invoke_subgraph_placeholder --------------------------------
    _original_placeholder = _isg_mod.invoke_subgraph_placeholder

    def invoke_subgraph_placeholder_patched(func, *args, **kwargs):
        if torch.compiler.is_dynamo_compiling():
            raise RuntimeError("invoke_subgraph should not be called directly in Dynamo")
        if torch.compiler.is_compiling():

            def _wrapper(func, *args, **kwargs):
                return invoke_subgraph_placeholder_patched(func, *args, **kwargs)

            from torch._higher_order_ops.utils import setup_compilation_env

            with setup_compilation_env() as backend:
                return torch.compile(
                    _wrapper,
                    backend=backend,
                    fullgraph=True,
                )(func, *args, **kwargs)
        return func(*args, **kwargs)

    _isg_mod.invoke_subgraph_placeholder = invoke_subgraph_placeholder_patched

    # Patch mark_compile_region's inner closure to use the patched version.
    # mark_compile_region captures invoke_subgraph_placeholder by name at
    # definition time, so we need to update the module-level name it resolves.
    # The function is already defined; patching the module attribute is enough
    # because inner() calls invoke_subgraph_placeholder via the module global.

    # ---- (b) py_functionalize_impl --------------------------------------
    # We need to replace the registered py_functionalize_impl.
    # The impl is registered via @invoke_subgraph.py_functionalize_impl,
    # which stores it in invoke_subgraph._dispatch_cache / py_kernels.
    # The cleanest way is to re-register it.

    invoke_subgraph = _isg_mod.invoke_subgraph
    FunctionalizeCtxWrapper = _isg_mod.FunctionalizeCtxWrapper
    get_invoke_subgraph_cache = _isg_mod.get_invoke_subgraph_cache

    @invoke_subgraph.py_functionalize_impl
    def _functionalize_impl(ctx, subgraph, identifier, *operands):
        from torch._higher_order_ops.auto_functionalize import (
            can_auto_functionalize,
            do_auto_functionalize_v2,
        )
        from torch._higher_order_ops.utils import HopInstance

        tokens_before = dict(ctx.mode._tokens)

        invoke_subgraph_cache = get_invoke_subgraph_cache()
        effects = None
        if invoke_subgraph_cache:
            effects = invoke_subgraph_cache.get_effects(identifier)

        if effects:
            if len(effects) != 1:
                raise AssertionError(f"Multiple effects within a subgraph NYI, got {len(effects)} effects")
            tokens = ctx.mode._tokens
            effects = next(iter(effects))
            token_input = tokens[effects]
            operands = (token_input, *operands)

            def wrap_subgraph(subgraph):
                def wrapped_subgraph(token, *args):
                    res = subgraph(*args)
                    return ctx.unwrap_tensors(ctx.mode._tokens[effects]), *res

                return wrapped_subgraph

            subgraph = wrap_subgraph(subgraph)

        unwrapped_operands = ctx.unwrap_tensors(operands)

        # FIX (b1): wrap HopInstance.create in try/except with env-var fallback
        hop_instance = None
        try:
            hop_instance = HopInstance.create(invoke_subgraph, subgraph, identifier, *operands)
        except Exception:
            if os.environ.get("TORCH_INVOKE_ALLOW_CREATE_FALLBACK", "0") != "1":
                raise

        disable_auto_functionalize = os.environ.get("TORCH_INVOKE_DISABLE_AUTO_FUNCTIONALIZE", "0") == "1"

        if hop_instance is not None and can_auto_functionalize(hop_instance) and not disable_auto_functionalize:
            if not isinstance(identifier, str):
                raise AssertionError(f"identifier must be a string for auto_functionalize, got {type(identifier)}")
            return do_auto_functionalize_v2(
                ctx.mode,
                hop_instance,
                (subgraph, "auto_functionalized_" + identifier, *operands),
                {},
            )

        with ctx.redispatch_to_next():
            functionalized_subgraph = FunctionalizeCtxWrapper(ctx, subgraph)
            # FIX (b2): wrap invoke_subgraph call with try/except + env-var fallback
            try:
                out = invoke_subgraph(functionalized_subgraph, identifier, *unwrapped_operands)
            except RuntimeError:
                if os.environ.get("TORCH_INVOKE_ALLOW_FUNCTIONALIZE_FALLBACK", "0") != "1":
                    raise
                if getattr(subgraph, "_boxed_call", False):
                    out = subgraph(list(unwrapped_operands))
                else:
                    out = subgraph(*unwrapped_operands)

        if effects:
            (new_token, *out) = out
            ctx.mode._tokens[effects] = new_token

        tokens_after = dict(ctx.mode._tokens)
        discovered_effects = set()
        for effect_type, token in tokens_after.items():
            if effect_type not in tokens_before or tokens_before[effect_type] is not token:
                discovered_effects.add(effect_type)

        if discovered_effects:
            if not ctx.mode._allow_token_discovery:
                raise AssertionError(
                    f"Number of tokens changed by {len(discovered_effects)} when tracing subgraph {subgraph}."
                )
            if invoke_subgraph_cache:
                invoke_subgraph_cache.add_effects(identifier, discovered_effects)

        return ctx.wrap_tensors(out)

    # ---- (c) ProxyTorchDispatchMode impl --------------------------------
    from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
    from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts

    @invoke_subgraph.py_impl(ProxyTorchDispatchMode)
    def _proxy_mode_impl(proxy_mode, subgraph, identifier, *operands):
        graph = None
        invoke_subgraph_cache = get_invoke_subgraph_cache()
        if invoke_subgraph_cache:
            graph = invoke_subgraph_cache.get_proxy_dispatch_entry(identifier)

        if graph is None:
            from torch._dynamo.utils import dynamo_timed

            with dynamo_timed("invoke_subgraph_proxy_tensor", log_pt2_compile_event=True):
                subgraph_decomp_table = _isg_mod._extract_nested_region_config(subgraph)
                with torch.fx.traceback._preserve_node_seq_nr():
                    graph = _isg_mod.reenter_make_fx(subgraph, subgraph_decomp_table=subgraph_decomp_table)(*operands)

                from torch._guards import detect_fake_mode

                fake_mode = detect_fake_mode(operands)

                # FIX (c): gate deferred runtime asserts with env var
                skip_deferred_asserts = os.environ.get("TORCH_INVOKE_SKIP_DEFERRED_ASSERTS", "0") == "1"
                if fake_mode is not None and fake_mode.shape_env is not None and not skip_deferred_asserts:
                    insert_deferred_runtime_asserts(
                        graph,
                        fake_mode.shape_env,
                        "invoke_subgraph_proxy_torch_dispatch_mode",
                        export=True,
                    )
                graph.recompile()

            if not isinstance(proxy_mode.tracer, torch.fx.Tracer):
                raise AssertionError(f"expected proxy_mode.tracer to be torch.fx.Tracer, got {type(proxy_mode.tracer)}")
            if invoke_subgraph_cache:
                invoke_subgraph_cache.add_proxy_dispatch_entry(identifier, graph)

        import torch.utils._pytree as pytree
        from torch.fx.experimental.proxy_tensor import track_tensor_tree

        node_args = (graph, identifier, *operands)

        def _unwrap_proxy(arg):
            if isinstance(arg, torch.fx.GraphModule):
                registered_before = False
                for _, submod in proxy_mode.tracer.root.named_modules():
                    if arg is submod:
                        registered_before = True
                if not registered_before:
                    qualname = proxy_mode.tracer.get_fresh_qualname("repeated_subgraph")
                    proxy_mode.tracer.root.register_module(qualname, arg)
            return proxy_mode.tracer.unwrap_proxy(arg)

        proxy_args = pytree.tree_map(_unwrap_proxy, node_args)
        out_proxy = proxy_mode.tracer.create_proxy("call_function", invoke_subgraph, proxy_args, {})
        example_out = invoke_subgraph(graph, identifier, *operands)
        return track_tensor_tree(example_out, out_proxy, constant=None, tracer=proxy_mode.tracer)


_original_ep_verify_signature = getattr(_ep_module, "_verify_exported_program_signature", None)
_original_verifier_verify_signature = getattr(_verifier_module, "_verify_exported_program_signature", None)
_original_named_buffers = _ep_module.ExportedProgram.named_buffers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def apply_torch_patches():
    """Apply all monkey patches."""
    # --- ONNX patches ---
    onnx_utils._setup_trace_module_map = _setup_trace_module_map_patched
    if hasattr(onnx_utils, "_get_module_attributes"):
        onnx_utils._get_module_attributes = _get_module_attributes
    ts_utils._setup_trace_module_map = _setup_trace_module_map_patched
    if hasattr(ts_utils, "_get_module_attributes"):
        ts_utils._get_module_attributes = _get_module_attributes
    if _original_track_scope_attrs is not None:
        _C._jit_pass_onnx_track_scope_attributes = _track_scope_attributes_patched

    # --- FunctionalTensorMode.__torch_dispatch__ ---
    _patch_functional_tensor_mode_dispatch()

    # --- verifier._verify_exported_program_signature ---
    if hasattr(_verifier_module, "_verify_exported_program_signature"):
        _verifier_module._verify_exported_program_signature = _patched_verify_exported_program_signature
    if hasattr(_ep_module, "_verify_exported_program_signature"):
        _ep_module._verify_exported_program_signature = _patched_verify_exported_program_signature

    # --- ExportedProgram.named_buffers ---
    _ep_module.ExportedProgram.named_buffers = _patched_named_buffers

    # --- invoke_subgraph patches ---
    _patch_invoke_subgraph()


def undo_torch_patches():
    """Undo all monkey patches and restore original functions."""
    # --- ONNX patches ---
    onnx_utils._setup_trace_module_map = _original_setup_trace_module_map
    if _original_get_module_attributes:
        onnx_utils._get_module_attributes = _original_get_module_attributes
    ts_utils._setup_trace_module_map = _original_ts_setup_trace_module_map
    if _original_ts_get_module_attributes:
        ts_utils._get_module_attributes = _original_ts_get_module_attributes
    if _original_track_scope_attrs is not None:
        _C._jit_pass_onnx_track_scope_attributes = _original_track_scope_attrs

    # --- verifier restore ---
    if _original_verifier_verify_signature is not None and hasattr(
        _verifier_module, "_verify_exported_program_signature"
    ):
        _verifier_module._verify_exported_program_signature = _original_verifier_verify_signature
    if _original_ep_verify_signature is not None and hasattr(_ep_module, "_verify_exported_program_signature"):
        _ep_module._verify_exported_program_signature = _original_ep_verify_signature

    # --- ExportedProgram.named_buffers restore ---
    _ep_module.ExportedProgram.named_buffers = _original_named_buffers

    # Note: FunctionalTensorMode.__torch_dispatch__ and invoke_subgraph
    # dispatch impls are not easily reversible without storing the originals
    # before patching. If undo is needed for these, store originals before
    # calling apply_torch_patches().
