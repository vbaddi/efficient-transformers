# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import ast
import copy
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import onnx
from onnx import helper

_NAME_SCOPES_KEY = "pkg.torch.onnx.name_scopes"
_CLASS_HIERARCHY_KEY = "pkg.torch.onnx.class_hierarchy"


def _get_metadata_value(node: onnx.NodeProto, key: str) -> str:
    for prop in node.metadata_props:
        if prop.key == key:
            return prop.value
    return ""


def _parse_scopes(value: str) -> List[str]:
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed if str(item)]
    if isinstance(parsed, str):
        return [item for item in parsed.split("/") if item]
    if parsed is not None:
        return [str(parsed)]
    return [item for item in value.split("/") if item]


def _get_scopes(node: onnx.NodeProto) -> Tuple[List[str], List[str]]:
    class_hierarchy = _get_metadata_value(node, _CLASS_HIERARCHY_KEY)
    name_scopes = _get_metadata_value(node, _NAME_SCOPES_KEY)
    return _parse_scopes(class_hierarchy), _parse_scopes(name_scopes)


def _get_instance_key_scopes(name_scopes: List[str]) -> List[str] | None:
    patterns = [
        r"(.*\.h\.\d+)",
        r"(.*\.layers\.\d+)",
        r"(.*\.layer\.\d+)",
        r"(.*\.blocks?\.\d+)",
    ]
    for idx, scope in enumerate(name_scopes):
        for pattern in patterns:
            match = re.search(pattern, scope)
            if match:
                base_scope = match.group(1)
                return name_scopes[:idx] + [base_scope]
    return None


def _make_function_name(target_classname: str, index: int) -> str:
    return target_classname if index == 0 else f"{target_classname}_{index}"


def _normalize_name(name: str) -> str:
    name = re.sub(r"\.h\.\d+", ".h.#", name)
    name = re.sub(r"\.\d+", ".#", name)
    return re.sub(r"\d+", "#", name)


def _parse_block_index(scope_path: str) -> int | None:
    patterns = [r"\.h\.(\d+)", r"\.layers\.(\d+)", r"\.layer\.(\d+)", r"\.blocks?\.(\d+)"]
    for pattern in patterns:
        match = re.search(pattern, scope_path)
        if match:
            return int(match.group(1))
    return None


def _is_past_name(name: str) -> bool:
    if name.endswith("_RetainedState"):
        return True
    if "_InternalRetainedState" in name:
        return True
    tokens = ("past_key", "past_value", "past_key_values", "key_cache", "value_cache")
    return any(token in name for token in tokens)


def _is_past_input_name(name: str) -> bool:
    if "_InternalRetainedState" in name or name.endswith("_RetainedState"):
        return False
    return "past_key" in name or "past_value" in name


def _is_shape_node(node: onnx.NodeProto) -> bool:
    return node.op_type in {
        "Shape",
        "Size",
    }


def _is_shape_op_type(op_type: str) -> bool:
    return op_type in {
        "Shape",
        "Size",
        "Unsqueeze",
        "Squeeze",
        "Concat",
        "Slice",
        "Range",
        "Cast",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "FloorDiv",
        "Mod",
        "Greater",
        "Less",
        "Equal",
        "Where",
        "Gather",
        "Reshape",
        "Expand",
    }


def _outputs_used_outside_group(node: onnx.NodeProto, consumers: Dict[str, set], group_index_set: set[int]) -> bool:
    for out_name in node.output:
        if not out_name:
            continue
        consumer_indices = consumers.get(out_name, set())
        if any(idx not in group_index_set for idx in consumer_indices):
            return True
    return False


def _outputs_used_outside_group_or_copy(
    node: onnx.NodeProto,
    consumers: Dict[str, set],
    group_index_set: set[int],
    copy_index_set: set[int],
) -> bool:
    for out_name in node.output:
        if not out_name:
            continue
        consumer_indices = consumers.get(out_name, set())
        if any(idx not in group_index_set and idx not in copy_index_set for idx in consumer_indices):
            return True
    return False


def _collect_consumers(nodes: Iterable[onnx.NodeProto]) -> dict:
    consumers = {}
    for idx, node in enumerate(nodes):
        for inp in node.input:
            if not inp:
                continue
            consumers.setdefault(inp, set()).add(idx)
    return consumers


def _collect_producers(nodes: Iterable[onnx.NodeProto]) -> dict:
    producers = {}
    for idx, node in enumerate(nodes):
        for out in node.output:
            if out:
                producers[out] = idx
    return producers


def _expand_group_indices(
    nodes: Sequence[onnx.NodeProto],
    indices: Sequence[int],
    producers: Dict[str, int],
    consumers: Dict[str, set],
    graph_inputs: set,
    graph_outputs: set,
    node_instances: Sequence[str | None],
    scope_path: str,
) -> set[int]:
    expanded = set(indices)
    changed = True
    while changed:
        changed = False
        for idx in list(expanded):
            node = nodes[idx]
            for inp in node.input:
                if not inp or inp in graph_inputs:
                    continue
                prod_idx = producers.get(inp)
                if prod_idx is None or prod_idx in expanded:
                    continue
                prod_scope = node_instances[prod_idx]
                if prod_scope != scope_path and prod_scope is not None:
                    continue
                prod_node = nodes[prod_idx]
                only_used_by_group = True
                for out in prod_node.output:
                    if not out:
                        continue
                    if out in graph_outputs:
                        only_used_by_group = False
                        break
                    consumer_indices = consumers.get(out, set())
                    if any(consumer not in expanded for consumer in consumer_indices):
                        only_used_by_group = False
                        break
                if not only_used_by_group:
                    continue
                expanded.add(prod_idx)
                changed = True
    return expanded


def _is_standard_domain(domain: str) -> bool:
    return domain in ("", "ai.onnx")


def _topo_sort_graph(graph: onnx.GraphProto) -> None:
    available = {inp.name for inp in graph.input}
    available.update(init.name for init in graph.initializer)
    nodes = list(graph.node)
    sorted_nodes = []
    remaining = nodes
    while remaining:
        progress = False
        next_remaining = []
        for node in remaining:
            if all((not inp) or inp in available for inp in node.input):
                sorted_nodes.append(node)
                for out in node.output:
                    if out:
                        available.add(out)
                progress = True
            else:
                next_remaining.append(node)
        if not progress:
            sorted_nodes.extend(next_remaining)
            break
        remaining = next_remaining
    del graph.node[:]
    graph.node.extend(sorted_nodes)


def _prune_unused_function_outputs(model: onnx.ModelProto) -> None:
    if not model.functions:
        return
    functions_by_key = {(fn.domain, fn.name): fn for fn in model.functions}
    calls_by_name: Dict[str, List[onnx.NodeProto]] = {}
    for node in model.graph.node:
        if (node.domain, node.op_type) in functions_by_key:
            calls_by_name.setdefault(node.op_type, []).append(node)

    consumers = _collect_consumers(model.graph.node)
    graph_outputs = {output.name for output in model.graph.output}

    for fn in model.functions:
        calls = calls_by_name.get(fn.name, [])
        if not calls:
            continue
        used_indices = set()
        for i, fn_out in enumerate(fn.output):
            if fn_out in graph_outputs:
                used_indices.add(i)
                continue
            for call in calls:
                if i >= len(call.output):
                    continue
                call_out = call.output[i]
                if call_out in graph_outputs or consumers.get(call_out):
                    used_indices.add(i)
                    break
        if len(used_indices) == len(fn.output):
            continue
        keep_indices = [i for i in range(len(fn.output)) if i in used_indices]
        fn.output[:] = [fn.output[i] for i in keep_indices]
        for call in calls:
            call.output[:] = [call.output[i] for i in keep_indices]


def _collect_shape_node_indices(model: onnx.ModelProto, max_scalar_size: int = 16) -> set[int]:
    shape_seed_ops = {"Shape", "Size"}
    shape_ops = {
        "Shape",
        "Size",
        "Unsqueeze",
        "Squeeze",
        "Concat",
        "Slice",
        "Range",
        "Cast",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "FloorDiv",
        "Mod",
        "Greater",
        "Less",
        "Equal",
        "Where",
        "Gather",
        "Reshape",
        "Expand",
    }

    init_sizes = {}
    for init in model.graph.initializer:
        size = 1
        for dim in init.dims:
            size *= dim if dim > 0 else 0
        init_sizes[init.name] = size

    shape_values = set()
    for node in model.graph.node:
        if node.op_type in shape_seed_ops:
            shape_values.update(out for out in node.output if out)

    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            if node.op_type not in shape_ops or not node.output:
                continue
            if all(
                inp in shape_values or init_sizes.get(inp, max_scalar_size + 1) <= max_scalar_size
                for inp in node.input
                if inp
            ):
                new_outputs = [out for out in node.output if out and out not in shape_values]
                if new_outputs:
                    shape_values.update(new_outputs)
                    changed = True

    consumers = _collect_consumers(model.graph.node)
    shape_node_indices = set()
    for idx, node in enumerate(model.graph.node):
        if not any(out in shape_values for out in node.output if out):
            continue
        all_shape_consumers = True
        for out in node.output:
            if not out:
                continue
            for consumer_idx in consumers.get(out, set()):
                consumer = model.graph.node[consumer_idx]
                if consumer.op_type not in shape_ops:
                    all_shape_consumers = False
                    break
            if not all_shape_consumers:
                break
        if all_shape_consumers:
            shape_node_indices.add(idx)
    return shape_node_indices


def _topo_sort_function(fn: onnx.FunctionProto) -> None:
    available = set(fn.input)
    nodes = list(fn.node)
    sorted_nodes = []
    remaining = nodes
    while remaining:
        progress = False
        next_remaining = []
        for node in remaining:
            if all((not inp) or inp in available for inp in node.input):
                sorted_nodes.append(node)
                for out in node.output:
                    if out:
                        available.add(out)
                progress = True
            else:
                next_remaining.append(node)
        if not progress:
            sorted_nodes.extend(next_remaining)
            break
        remaining = next_remaining
    del fn.node[:]
    fn.node.extend(sorted_nodes)


def _ensure_function_domain_import(model: onnx.ModelProto, function_domain: str) -> None:
    for opset in model.opset_import:
        if opset.domain == function_domain:
            return
    model.opset_import.append(helper.make_opsetid(function_domain, 1))


def _collect_used_functions(model: onnx.ModelProto) -> set[Tuple[str, str]]:
    used = set()
    queue = [(node.domain, node.op_type) for node in model.graph.node]
    functions_by_key = {(fn.domain, fn.name): fn for fn in model.functions}
    while queue:
        key = queue.pop()
        if key in used:
            continue
        used.add(key)
        fn = functions_by_key.get(key)
        if not fn:
            continue
        for fn_node in fn.node:
            queue.append((fn_node.domain, fn_node.op_type))
    return used


def _collect_value_types(model: onnx.ModelProto) -> Dict[str, int]:
    types: Dict[str, int] = {}
    for inp in model.graph.input:
        t = inp.type.tensor_type
        if t.HasField("elem_type"):
            types[inp.name] = t.elem_type
    for vi in model.graph.value_info:
        t = vi.type.tensor_type
        if t.HasField("elem_type"):
            types[vi.name] = t.elem_type
    for init in model.graph.initializer:
        types[init.name] = init.data_type
    return types


def _collect_value_info(model: onnx.ModelProto) -> Dict[str, onnx.ValueInfoProto]:
    infos: Dict[str, onnx.ValueInfoProto] = {}
    for inp in model.graph.input:
        infos[inp.name] = inp
    for out in model.graph.output:
        infos[out.name] = out
    for vi in model.graph.value_info:
        infos[vi.name] = vi
    return infos


def _ensure_initializer_value_info(model: onnx.ModelProto) -> None:
    existing = {vi.name for vi in model.graph.value_info}
    existing.update(inp.name for inp in model.graph.input)
    existing.update(out.name for out in model.graph.output)
    for init in model.graph.initializer:
        if init.name in existing:
            continue
        vi = helper.make_tensor_value_info(init.name, init.data_type, init.dims)
        model.graph.value_info.append(vi)
        existing.add(init.name)


def _collect_function_required_inputs(fn: onnx.FunctionProto) -> List[str]:
    produced = set()
    for node in fn.node:
        for out in node.output:
            if out:
                produced.add(out)
    required = []
    seen = set()
    for node in fn.node:
        for inp in node.input:
            if not inp or inp in produced or inp in seen:
                continue
            seen.add(inp)
            required.append(inp)
    return required


def _align_function_inputs(model: onnx.ModelProto, function_domain: str) -> None:
    if not model.functions:
        return
    for fn in model.functions:
        if fn.domain != function_domain:
            continue
        required_inputs = _collect_function_required_inputs(fn)
        old_inputs = list(fn.input)
        if required_inputs == old_inputs:
            continue
        fn.input[:] = required_inputs
        old_index = {name: idx for idx, name in enumerate(old_inputs)}
        for node in model.graph.node:
            if node.domain != function_domain or node.op_type != fn.name:
                continue
            new_inputs = []
            for inp_name in required_inputs:
                idx = old_index.get(inp_name)
                if idx is not None and idx < len(node.input):
                    new_inputs.append(node.input[idx])
                else:
                    new_inputs.append(inp_name)
            node.input[:] = new_inputs


def _swap_value_names(model: onnx.ModelProto, name_a: str, name_b: str) -> None:
    def swap(name: str) -> str:
        if name == name_a:
            return name_b
        if name == name_b:
            return name_a
        return name

    for inp in model.graph.input:
        if inp.name in {name_a, name_b}:
            inp.name = swap(inp.name)
    for out in model.graph.output:
        if out.name in {name_a, name_b}:
            out.name = swap(out.name)
    for vi in model.graph.value_info:
        if vi.name in {name_a, name_b}:
            vi.name = swap(vi.name)
    for init in model.graph.initializer:
        if init.name in {name_a, name_b}:
            init.name = swap(init.name)
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in {name_a, name_b}:
                node.input[i] = swap(inp)
        for i, out in enumerate(node.output):
            if out in {name_a, name_b}:
                node.output[i] = swap(out)


def fix_misaligned_gpt2_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
    inputs = {inp.name: inp.type.tensor_type for inp in model.graph.input}
    if "position_ids" not in inputs or "past_key.0" not in inputs:
        return model
    pos_type = inputs["position_ids"]
    pk_type = inputs["past_key.0"]
    pos_elem = pos_type.elem_type if pos_type.HasField("elem_type") else None
    pk_elem = pk_type.elem_type if pk_type.HasField("elem_type") else None
    pos_rank = len(pos_type.shape.dim)
    pk_rank = len(pk_type.shape.dim)
    if pos_elem == onnx.TensorProto.FLOAT and pk_elem == onnx.TensorProto.INT64 and pos_rank > pk_rank:
        _swap_value_names(model, "position_ids", "past_key.0")
    return model


def restore_symbolic_batch_dim(
    model: onnx.ModelProto,
    batch_dim_name: str = "batch_size",
    input_prefixes: Sequence[str] = ("input_ids", "position_ids", "attention_mask", "past_key.", "past_value."),
) -> onnx.ModelProto:
    for inp in model.graph.input:
        name = inp.name
        if not any(name == prefix or name.startswith(prefix) for prefix in input_prefixes):
            continue
        shape = inp.type.tensor_type.shape
        if not shape.dim:
            continue
        dim0 = shape.dim[0]
        if dim0.dim_param == batch_dim_name:
            continue
        if dim0.HasField("dim_value"):
            dim0.ClearField("dim_value")
        dim0.dim_param = batch_dim_name
    return model


def replace_invalid_index_constants(model: onnx.ModelProto) -> onnx.ModelProto:
    def fix_constant_node(node: onnx.NodeProto) -> None:
        if node.op_type != "Constant":
            return
        for attr in node.attribute:
            if attr.type != onnx.AttributeProto.TENSOR:
                continue
            tensor = attr.t
            if tensor.data_type == onnx.TensorProto.UNDEFINED:
                continue
            arr = onnx.numpy_helper.to_array(tensor)
            if arr.shape == () and np.issubdtype(arr.dtype, np.integer) and int(arr) == 2147483647:
                new_tensor = onnx.numpy_helper.from_array(np.array(0, dtype=arr.dtype))
                attr.t.CopyFrom(new_tensor)

    for node in model.graph.node:
        fix_constant_node(node)
    for fn in model.functions:
        for node in fn.node:
            fix_constant_node(node)
    return model


def replace_invalid_index_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    def fix_initializer(tensor: onnx.TensorProto) -> bool:
        if tensor.data_type == onnx.TensorProto.UNDEFINED:
            return False
        if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
            return False
        arr = onnx.numpy_helper.to_array(tensor)
        if arr.shape == () and np.issubdtype(arr.dtype, np.integer) and int(arr) == 2147483647:
            new_tensor = onnx.numpy_helper.from_array(np.array(0, dtype=arr.dtype), name=tensor.name)
            tensor.CopyFrom(new_tensor)
            return True
        return False

    for tensor in model.graph.initializer:
        fix_initializer(tensor)
    return model


def extract_functions_from_metadata(
    model: onnx.ModelProto,
    target_module_classnames: Sequence[str],
    function_domain: str = "qeff.subfunction",
    min_nodes: int = 5,
    preserve_graph: bool = False,
) -> onnx.ModelProto:
    if not target_module_classnames:
        return model

    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    _ensure_initializer_value_info(model)

    decoder_like = any(
        "DecoderLayer" in cls_name or cls_name.endswith("Block") for cls_name in target_module_classnames
    )

    class_scopes_list: List[List[str]] = []
    name_scopes_list: List[List[str]] = []
    for node in model.graph.node:
        class_scopes, name_scopes = _get_scopes(node)
        class_scopes_list.append(class_scopes)
        name_scopes_list.append(name_scopes)

    original_nodes_by_output = {}
    for node in model.graph.node:
        for out in node.output:
            if out and out not in original_nodes_by_output:
                original_nodes_by_output[out] = copy.deepcopy(node)

    node_instances: List[str | None] = []
    for class_scopes, name_scopes in zip(class_scopes_list, name_scopes_list):
        match_idx = next(
            (
                i
                for i, scope in enumerate(class_scopes)
                if any(cls_name in scope for cls_name in target_module_classnames)
            ),
            None,
        )
        if match_idx is None:
            match_idx = next(
                (
                    i
                    for i, scope in enumerate(name_scopes)
                    if any(cls_name in scope for cls_name in target_module_classnames)
                ),
                None,
            )
        if match_idx is None:
            instance_scopes = _get_instance_key_scopes(name_scopes)
            if instance_scopes is not None and decoder_like:
                node_instances.append("/".join(instance_scopes))
            else:
                node_instances.append(None)
            continue
        instance_scopes = _get_instance_key_scopes(name_scopes)
        if instance_scopes is not None:
            key_scopes = instance_scopes
        elif name_scopes and len(name_scopes) > match_idx:
            key_scopes = name_scopes[: match_idx + 1]
        else:
            key_scopes = class_scopes[: match_idx + 1]
        node_instances.append("/".join(key_scopes) if key_scopes else None)

    shape_node_indices = _collect_shape_node_indices(model)

    graph_outputs = {output.name for output in model.graph.output}
    graph_inputs = {inp.name for inp in model.graph.input}
    initializer_map = {init.name: init for init in model.graph.initializer}
    graph_inputs.update(initializer_map.keys())
    producers = _collect_producers(model.graph.node)
    consumers = _collect_consumers(model.graph.node)
    value_types = _collect_value_types(model)
    value_info_map = _collect_value_info(model)
    non_past_outputs = [output.name for output in model.graph.output if not _is_past_name(output.name)]
    values_reaching_outputs = set(non_past_outputs)
    stack = list(non_past_outputs)
    while stack:
        value = stack.pop()
        prod_idx = producers.get(value)
        if prod_idx is None:
            continue
        prod_node = model.graph.node[prod_idx]
        for inp in prod_node.input:
            if inp and inp not in values_reaching_outputs:
                values_reaching_outputs.add(inp)
                stack.append(inp)

    prune_group_nodes = False
    shared_outputs = set()
    if prune_group_nodes:
        for out_name, consumer_idxs in consumers.items():
            scopes = {node_instances[idx] for idx in consumer_idxs if idx < len(node_instances) and node_instances[idx]}
            if len(scopes) > 1:
                shared_outputs.add(out_name)

    groups: List[Dict[str, object]] = []
    for idx, node in enumerate(model.graph.node):
        if idx in shape_node_indices:
            continue
        class_scopes = class_scopes_list[idx]
        name_scopes = name_scopes_list[idx]
        matched_class = None
        instance_scopes = _get_instance_key_scopes(name_scopes)
        if class_scopes:
            match_idx = next(
                (
                    i
                    for i, scope in enumerate(class_scopes)
                    if any(cls_name in scope for cls_name in target_module_classnames)
                ),
                None,
            )
            if match_idx is not None:
                matched_class = next(
                    (cls_name for cls_name in target_module_classnames if cls_name in class_scopes[match_idx]),
                    None,
                )
                if name_scopes and len(name_scopes) > match_idx:
                    instance_scopes = _get_instance_key_scopes(name_scopes)
                    if instance_scopes is not None:
                        key_scopes = instance_scopes
                    else:
                        key_scopes = name_scopes[: match_idx + 1]
                else:
                    key_scopes = class_scopes[: match_idx + 1]
            else:
                match_idx = next(
                    (
                        i
                        for i, scope in enumerate(name_scopes)
                        if any(cls_name in scope for cls_name in target_module_classnames)
                    ),
                    None,
                )
                if match_idx is None:
                    continue
                matched_class = next(
                    (cls_name for cls_name in target_module_classnames if cls_name in name_scopes[match_idx]),
                    None,
                )
                instance_scopes = _get_instance_key_scopes(name_scopes)
                if instance_scopes is not None:
                    key_scopes = instance_scopes
                else:
                    key_scopes = name_scopes[: match_idx + 1] if name_scopes else []
        else:
            match_idx = next(
                (
                    i
                    for i, scope in enumerate(name_scopes)
                    if any(cls_name in scope for cls_name in target_module_classnames)
                ),
                None,
            )
            if match_idx is None:
                continue
            matched_class = next(
                (cls_name for cls_name in target_module_classnames if cls_name in name_scopes[match_idx]),
                None,
            )
            instance_scopes = _get_instance_key_scopes(name_scopes)
            if instance_scopes is not None:
                key_scopes = instance_scopes
            else:
                key_scopes = name_scopes[: match_idx + 1] if name_scopes else []
        if not key_scopes or matched_class is None:
            continue
        if prune_group_nodes and any(out in shared_outputs for out in node.output if out):
            continue
        scope_path = "/".join(key_scopes)
        groups.append({"class": matched_class, "scope": scope_path, "idx": idx})

    if not groups:
        return model

    grouped_indices: Dict[Tuple[str, str], List[int]] = {}
    for entry in groups:
        key = (entry["class"], entry["scope"])
        grouped_indices.setdefault(key, []).append(entry["idx"])

    scope_to_index: Dict[str, int] = {}
    for _, scope in grouped_indices.keys():
        block_index = _parse_block_index(scope)
        if block_index is not None:
            scope_to_index[scope] = block_index

    index_to_scope = {idx: scope for scope, idx in scope_to_index.items()}
    # max_block_index = max(index_to_scope.keys(), default=None)

    hidden_input_by_scope: Dict[str, str] = {}
    norm_tokens = (
        "ln_1",
        "input_layernorm",
        "input_layer_norm",
        "input_layernorm",
        "pre_attention_layernorm",
        "rms_norm",
        "layer_norm",
        "norm",
    )
    norm_op_types = {
        "LayerNormalization",
        "RMSNormalization",
        "RMSNorm",
        "CustomRMSNorm",
    }
    for scope in grouped_indices.keys():
        scope_path = scope[1]
        for idx, node in enumerate(model.graph.node):
            if not _is_standard_domain(node.domain):
                continue
            if node_instances[idx] != scope_path:
                continue
            name_scopes = name_scopes_list[idx]
            class_scopes = class_scopes_list[idx]
            if (
                any(token in scope_name for token in norm_tokens for scope_name in name_scopes)
                or any(token in scope_name for token in norm_tokens for scope_name in class_scopes)
                or node.op_type in norm_op_types
            ):
                if node.input:
                    hidden_input_by_scope[scope_path] = node.input[0]
                    break

    primary_output_by_scope: Dict[str, str] = {}
    for scope_path, block_index in scope_to_index.items():
        next_scope = index_to_scope.get(block_index + 1)
        if next_scope and next_scope in hidden_input_by_scope:
            primary_output_by_scope[scope_path] = hidden_input_by_scope[next_scope]

    signature_to_name: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], str] = {}
    signature_templates: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Dict[str, object]] = {}
    class_name_counts: Dict[str, int] = {}
    group_entries: List[Dict[str, object]] = []
    for (matched_class, scope_path), indices in grouped_indices.items():
        indices = sorted(indices)
        expanded_indices = _expand_group_indices(
            model.graph.node,
            indices,
            producers,
            consumers,
            graph_inputs,
            graph_outputs,
            node_instances,
            scope_path,
        )
        for idx, scope in enumerate(node_instances):
            if scope != scope_path:
                continue
            if idx in shape_node_indices:
                continue
            if prune_group_nodes and any(out in shared_outputs for out in model.graph.node[idx].output if out):
                continue
            expanded_indices.add(idx)
        if len(expanded_indices) < min_nodes:
            continue
        group_index_set = set(expanded_indices)
        group_nodes = [model.graph.node[i] for i in sorted(group_index_set)]
        outputs_set = set()
        outputs_order = []
        for node in group_nodes:
            for out_name in node.output:
                if out_name and out_name not in outputs_set:
                    outputs_set.add(out_name)
                    outputs_order.append(out_name)

        # Pull in nodes that are downstream of group outputs and upstream of group inputs (feedback).
        group_input_names = set()
        for node in group_nodes:
            for inp in node.input:
                if inp and inp not in outputs_set:
                    group_input_names.add(inp)

        forward_nodes = set()
        forward_values = set(outputs_set)
        queue = list(outputs_set)
        while queue:
            val = queue.pop()
            for consumer_idx in consumers.get(val, set()):
                if consumer_idx in group_index_set:
                    continue
                if consumer_idx not in forward_nodes:
                    forward_nodes.add(consumer_idx)
                consumer_node = model.graph.node[consumer_idx]
                for out in consumer_node.output:
                    if out and out not in forward_values:
                        forward_values.add(out)
                        queue.append(out)

        backward_nodes = set()
        seen_values = set(group_input_names)
        queue = list(group_input_names)
        while queue:
            val = queue.pop()
            prod_idx = producers.get(val)
            if prod_idx is None or prod_idx in group_index_set:
                continue
            if prod_idx not in backward_nodes:
                backward_nodes.add(prod_idx)
                prod_node = model.graph.node[prod_idx]
                for inp in prod_node.input:
                    if inp and inp not in seen_values:
                        seen_values.add(inp)
                        queue.append(inp)

        feedback_nodes = forward_nodes.intersection(backward_nodes)
        if feedback_nodes:
            group_index_set.update(feedback_nodes)
            group_nodes = [model.graph.node[i] for i in sorted(group_index_set)]
            outputs_set = set()
            outputs_order = []
            for node in group_nodes:
                for out_name in node.output:
                    if out_name and out_name not in outputs_set:
                        outputs_set.add(out_name)
                        outputs_order.append(out_name)

        primary_output = primary_output_by_scope.get(scope_path)
        if primary_output is None:
            for out_name in reversed(outputs_order):
                if not out_name or _is_past_name(out_name):
                    continue
                consumer_indices = consumers.get(out_name, set())
                if out_name in graph_outputs or any(idx not in group_index_set for idx in consumer_indices):
                    primary_output = out_name
                    break
        if primary_output is None:
            for out_name in outputs_order:
                if out_name in values_reaching_outputs and not _is_past_name(out_name):
                    primary_output = out_name
                    break

        if prune_group_nodes:
            # Drop nodes that only feed other scopes (shared setup values), so block outputs stay minimal.
            removable = set()
            for idx in list(group_index_set):
                node = model.graph.node[idx]
                outs = [out for out in node.output if out]
                if not outs:
                    continue
                if primary_output and primary_output in outs:
                    continue
                if any(_is_past_name(out) for out in outs):
                    continue
                if any(out in shared_outputs for out in outs):
                    removable.add(idx)
                    continue
                external_scope_use = False
                for out in outs:
                    for consumer_idx in consumers.get(out, set()):
                        consumer_scope = node_instances[consumer_idx] if consumer_idx < len(node_instances) else None
                        if consumer_scope != scope_path:
                            external_scope_use = True
                            break
                    if external_scope_use:
                        break
                if external_scope_use:
                    removable.add(idx)
                    continue
                internal_use = False
                for out in outs:
                    for consumer_idx in consumers.get(out, set()):
                        if consumer_idx in group_index_set:
                            internal_use = True
                            break
                    if internal_use:
                        break
                if internal_use:
                    continue
                if any(out in graph_outputs for out in outs):
                    continue
                removable.add(idx)
            if removable:
                group_index_set.difference_update(removable)
                group_nodes = [model.graph.node[i] for i in sorted(group_index_set)]
                outputs_set = set()
                outputs_order = []
                for node in group_nodes:
                    for out_name in node.output:
                        if out_name and out_name not in outputs_set:
                            outputs_set.add(out_name)
                            outputs_order.append(out_name)

        past_outputs = [name for name in outputs_order if _is_past_name(name)]

        hidden_input = hidden_input_by_scope.get(scope_path)
        boundary_inputs = []
        boundary_inputs_set = set()
        for node in group_nodes:
            for inp in node.input:
                if not inp or inp in outputs_set or inp in boundary_inputs_set:
                    continue
                boundary_inputs_set.add(inp)
                boundary_inputs.append(inp)

        function_index_set = set(group_index_set)
        function_nodes = group_nodes

        # No input normalization; keep function nodes intact for correctness-first behavior.

        function_outputs = set()
        for node in function_nodes:
            for out_name in node.output:
                if out_name:
                    function_outputs.add(out_name)

        function_inputs = []
        function_inputs_set = set()
        for node in function_nodes:
            for inp in node.input:
                if not inp or inp in function_outputs or inp in function_inputs_set:
                    continue
                function_inputs_set.add(inp)
                function_inputs.append(inp)

        boundary_outputs = []
        if past_outputs:
            boundary_outputs.extend(past_outputs)
        primary_output = primary_output_by_scope.get(scope_path)

        boundary_outputs_set = set(boundary_outputs)
        for out_name in outputs_order:
            if out_name in boundary_outputs_set:
                continue
            consumer_indices = consumers.get(out_name, set())
            used_outside = any(idx not in function_index_set for idx in consumer_indices)
            if primary_output and out_name == primary_output:
                boundary_outputs_set.add(out_name)
                boundary_outputs.append(out_name)
                continue
            if out_name in graph_outputs or used_outside:
                boundary_outputs_set.add(out_name)
                boundary_outputs.append(out_name)

        if not boundary_outputs:
            continue

        ordered_inputs = []
        if hidden_input and hidden_input in function_inputs_set:
            ordered_inputs.append(hidden_input)
        past_inputs = [name for name in function_inputs if _is_past_name(name)]
        past_inputs.sort(key=_normalize_name)
        for name in past_inputs:
            if name not in ordered_inputs:
                ordered_inputs.append(name)
        remaining_inputs = [name for name in function_inputs if name not in ordered_inputs]
        remaining_inputs.sort(key=_normalize_name)
        ordered_inputs.extend(remaining_inputs)
        const_inputs = []

        hidden_sig_input = hidden_input if hidden_input in ordered_inputs else None
        if hidden_sig_input is None:
            skip_hidden = {"input_ids", "position_ids", "attention_mask", "token_type_ids", "input_mask"}
            for name in ordered_inputs:
                if name in skip_hidden or _is_past_name(name) or name in initializer_map:
                    continue
                hidden_sig_input = name
                break
        normalized_inputs = tuple(
            (
                "hidden_state" if hidden_sig_input and name == hidden_sig_input else _normalize_name(name),
                value_types.get(name, 0),
            )
            for name in ordered_inputs
        )

        def _normalize_output_name(name: str) -> str:
            if primary_output and name == primary_output:
                return "hidden_state_out"
            return _normalize_name(name)

        normalized_outputs = tuple(
            (_normalize_output_name(name), value_types.get(name, 0)) for name in boundary_outputs
        )
        signature = (matched_class, normalized_inputs, normalized_outputs)
        if signature not in signature_to_name:
            class_index = class_name_counts.get(matched_class, 0)
            function_name = _make_function_name(matched_class, class_index)
            class_name_counts[matched_class] = class_index + 1
            signature_to_name[signature] = function_name
            signature_templates[signature] = {
                "nodes": function_nodes,
                "inputs": ordered_inputs,
                "outputs": boundary_outputs,
                "const_inputs": const_inputs,
            }

        group_entries.append(
            {
                "indices": sorted(group_index_set),
                "boundary_inputs": ordered_inputs,
                "boundary_outputs": boundary_outputs,
                "signature": signature,
                "scope": scope_path,
                "class": matched_class,
            }
        )

    # Keep all group entries so every layer instance is emitted as a function call,
    # even when signature variants exist (e.g. first/last block differences).

    group_entries.sort(key=lambda entry: max(entry["indices"]), reverse=True)
    pending_calls = []
    nodes_to_remove = set()
    for call_idx, entry in enumerate(group_entries):
        signature = entry["signature"]
        function_name = signature_to_name.get(signature)
        if not function_name:
            continue
        boundary_inputs = entry["boundary_inputs"]
        call_inputs = boundary_inputs
        boundary_outputs = entry["boundary_outputs"]
        scope_path = entry.get("scope")
        scope_index = scope_to_index.get(scope_path) if scope_path else None
        call_id = scope_index if scope_index is not None else call_idx
        if preserve_graph:
            call_outputs = [f"{function_name}_out_{call_id}_{i}" for i in range(len(boundary_outputs))]
        else:
            call_outputs = list(boundary_outputs)
        replacement_map = dict(zip(boundary_outputs, call_outputs))

        if not preserve_graph:
            existing_vi_names = {vi.name for vi in model.graph.value_info}
            graph_output_names = {output.name for output in model.graph.output}
            new_value_infos = []
            for old_name, new_name in replacement_map.items():
                if new_name in existing_vi_names or new_name in graph_output_names:
                    continue
                vi = value_info_map.get(old_name)
                if vi is not None:
                    vi_copy = copy.deepcopy(vi)
                    vi_copy.name = new_name
                    new_value_infos.append(vi_copy)
                    existing_vi_names.add(new_name)
                    continue
                elem_type = value_types.get(old_name)
                if elem_type:
                    vi_copy = helper.make_tensor_value_info(new_name, elem_type, None)
                    new_value_infos.append(vi_copy)
                    existing_vi_names.add(new_name)
            if new_value_infos:
                model.graph.value_info.extend(new_value_infos)

            group_node_ids = {id(model.graph.node[idx]) for idx in entry["indices"]}
            nodes_to_remove.update(group_node_ids)
            for node in model.graph.node:
                if id(node) in group_node_ids:
                    continue
                for input_idx, inp in enumerate(node.input):
                    if inp in replacement_map:
                        node.input[input_idx] = replacement_map[inp]

            for output in model.graph.output:
                if output.name in replacement_map:
                    output.name = replacement_map[output.name]
            pending_calls.append(
                {
                    "function_name": function_name,
                    "inputs": call_inputs,
                    "outputs": call_outputs,
                    "original_max_index": max(entry["indices"]),
                }
            )
        else:
            call_node = helper.make_node(function_name, call_inputs, call_outputs, domain=function_domain)
            model.graph.node.append(call_node)

    if nodes_to_remove:
        kept_nodes = [node for node in model.graph.node if id(node) not in nodes_to_remove]
        del model.graph.node[:]
        model.graph.node.extend(kept_nodes)

    if pending_calls:
        pending_calls.sort(key=lambda item: item["original_max_index"])
        for spec in pending_calls:
            producer_index = {}
            for idx, node in enumerate(model.graph.node):
                for out_name in node.output:
                    producer_index[out_name] = idx
            max_producer = max((producer_index.get(inp, -1) for inp in spec["inputs"] if inp), default=-1)
            insert_index = max_producer + 1
            call_node = helper.make_node(spec["function_name"], spec["inputs"], spec["outputs"], domain=function_domain)
            model.graph.node.insert(insert_index, call_node)

    if not preserve_graph and original_nodes_by_output:
        max_rounds = 5
        for _ in range(max_rounds):
            producers = {out for node in model.graph.node for out in node.output if out}
            missing_inputs = set()
            for node in model.graph.node:
                for inp in node.input:
                    if not inp:
                        continue
                    if inp in producers or inp in graph_inputs or inp in initializer_map:
                        continue
                    missing_inputs.add(inp)
            if not missing_inputs:
                break
            added = False
            for missing in sorted(missing_inputs):
                orig_node = original_nodes_by_output.get(missing)
                if orig_node is None:
                    continue
                model.graph.node.append(copy.deepcopy(orig_node))
                added = True
            if not added:
                break

    for signature, template in signature_templates.items():
        _ensure_function_domain_import(model, function_domain)
        func_nodes = []
        for const_name in template.get("const_inputs", []):
            init = initializer_map.get(const_name)
            if init is None:
                continue
            const_node = helper.make_node("Constant", inputs=[], outputs=[const_name], value=init)
            func_nodes.append(const_node)
        func_nodes.extend([copy.deepcopy(node) for node in template["nodes"]])
        function_proto = helper.make_function(
            function_domain,
            signature_to_name[signature],
            template["inputs"],
            template["outputs"],
            func_nodes,
            opset_imports=model.opset_import,
        )
        model.functions.append(function_proto)

    # Keep function inputs/outputs unchanged for correctness-first behavior.
    _align_function_inputs(model, function_domain)

    for fn in model.functions:
        _topo_sort_function(fn)
    _topo_sort_graph(model.graph)

    if model.functions:
        used = _collect_used_functions(model)
        kept = [fn for fn in model.functions if (fn.domain, fn.name) in used]
        del model.functions[:]
        model.functions.extend(kept)

    return model
