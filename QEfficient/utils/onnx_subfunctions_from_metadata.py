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
    tokens = ("past_key", "past_value", "past_key_values", "key_cache", "value_cache")
    return any(token in name for token in tokens)


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


def extract_functions_from_metadata(
    model: onnx.ModelProto,
    target_module_classnames: Sequence[str],
    function_domain: str = "qeff.subfunction",
    min_nodes: int = 5,
) -> onnx.ModelProto:
    if not target_module_classnames:
        return model

    decoder_like = any(
        "DecoderLayer" in cls_name or cls_name.endswith("Block") for cls_name in target_module_classnames
    )

    class_scopes_list: List[List[str]] = []
    name_scopes_list: List[List[str]] = []
    for node in model.graph.node:
        class_scopes, name_scopes = _get_scopes(node)
        class_scopes_list.append(class_scopes)
        name_scopes_list.append(name_scopes)

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

    groups: List[Dict[str, object]] = []
    for idx, node in enumerate(model.graph.node):
        class_scopes = class_scopes_list[idx]
        name_scopes = name_scopes_list[idx]
        matched_class = None
        instance_scopes = _get_instance_key_scopes(name_scopes)
        if instance_scopes is not None and decoder_like:
            matched_class = next(
                (
                    cls_name
                    for cls_name in target_module_classnames
                    if ("DecoderLayer" in cls_name or cls_name.endswith("Block"))
                ),
                target_module_classnames[0],
            )
            key_scopes = instance_scopes
            scope_path = "/".join(key_scopes)
            groups.append({"class": matched_class, "scope": scope_path, "idx": idx})
            continue
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
        scope_path = "/".join(key_scopes)
        groups.append({"class": matched_class, "scope": scope_path, "idx": idx})

    if not groups:
        return model

    graph_outputs = {output.name for output in model.graph.output}
    graph_inputs = {inp.name for inp in model.graph.input}
    graph_inputs.update(init.name for init in model.graph.initializer)
    producers = _collect_producers(model.graph.node)
    consumers = _collect_consumers(model.graph.node)
    value_types = _collect_value_types(model)

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
    for scope in grouped_indices.keys():
        scope_path = scope[1]
        for idx, node in enumerate(model.graph.node):
            if node_instances[idx] != scope_path:
                continue
            name_scopes = name_scopes_list[idx]
            class_scopes = class_scopes_list[idx]
            if any(token in scope_name for token in norm_tokens for scope_name in name_scopes) or any(
                token in scope_name for token in norm_tokens for scope_name in class_scopes
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
        if len(indices) < min_nodes:
            continue
        group_index_set = set(indices)
        changed = True
        while changed:
            changed = False
            group_nodes = [model.graph.node[i] for i in sorted(group_index_set)]
            outputs_set = set()
            for node in group_nodes:
                for out_name in node.output:
                    if out_name:
                        outputs_set.add(out_name)
            for idx, node in enumerate(model.graph.node):
                if idx in group_index_set or node.metadata_props:
                    continue
                if any(
                    any(consumer_idx in group_index_set for consumer_idx in consumers.get(out_name, set()))
                    for out_name in node.output
                    if out_name
                ):
                    group_index_set.add(idx)
                    changed = True
            boundary_inputs = []
            boundary_inputs_set = set()
            for node in group_nodes:
                for inp in node.input:
                    if not inp or inp in outputs_set or inp in boundary_inputs_set:
                        continue
                    boundary_inputs_set.add(inp)
                    boundary_inputs.append(inp)
            for inp in boundary_inputs:
                prod_idx = producers.get(inp)
                if prod_idx is None or prod_idx in group_index_set:
                    continue
                prod_node = model.graph.node[prod_idx]
                if not prod_node.metadata_props:
                    group_index_set.add(prod_idx)
                    changed = True
                    continue
                if any(in_name in outputs_set for in_name in prod_node.input):
                    group_index_set.add(prod_idx)
                    changed = True

        group_nodes = [model.graph.node[i] for i in sorted(group_index_set)]

        outputs_set = set()
        outputs_order = []
        for node in group_nodes:
            for out_name in node.output:
                if out_name and out_name not in outputs_set:
                    outputs_set.add(out_name)
                    outputs_order.append(out_name)

        hidden_input = hidden_input_by_scope.get(scope_path)
        boundary_inputs = []
        boundary_inputs_set = set()
        for node in group_nodes:
            for inp in node.input:
                if not inp or inp in outputs_set or inp in boundary_inputs_set:
                    continue
                boundary_inputs_set.add(inp)
                boundary_inputs.append(inp)

        internal_copy_indices: set[int] = set()
        for inp in boundary_inputs:
            if not inp or inp == hidden_input or _is_past_name(inp) or inp in graph_inputs:
                continue
            prod_idx = producers.get(inp)
            if prod_idx is None or prod_idx in group_index_set:
                continue
            prod_node = model.graph.node[prod_idx]
            if prod_node.metadata_props and node_instances[prod_idx] is None:
                continue
            stack = [prod_idx]
            while stack:
                idx = stack.pop()
                if idx in group_index_set or idx in internal_copy_indices:
                    continue
                node = model.graph.node[idx]
                if node.metadata_props and node_instances[idx] is None:
                    continue
                internal_copy_indices.add(idx)
                for in_name in node.input:
                    if not in_name or in_name == hidden_input or _is_past_name(in_name) or in_name in graph_inputs:
                        continue
                    prod_idx = producers.get(in_name)
                    if prod_idx is None or prod_idx in group_index_set or prod_idx in internal_copy_indices:
                        continue
                    prod_node = model.graph.node[prod_idx]
                    if prod_node.metadata_props and node_instances[prod_idx] is None:
                        continue
                    stack.append(prod_idx)

        function_index_set = set(group_index_set)
        function_index_set.update(internal_copy_indices)
        function_nodes = [model.graph.node[i] for i in sorted(function_index_set)]

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

        past_outputs = [name for name in outputs_order if _is_past_name(name)]
        desired_outputs_set = set(past_outputs)
        primary_output = primary_output_by_scope.get(scope_path)
        if primary_output:
            desired_outputs_set.add(primary_output)

        boundary_outputs = []
        if past_outputs:
            boundary_outputs.extend(past_outputs)
        if primary_output and primary_output in outputs_set and primary_output not in boundary_outputs:
            boundary_outputs.append(primary_output)

        boundary_outputs_set = set(boundary_outputs)
        for out_name in outputs_order:
            if out_name in boundary_outputs_set:
                continue
            consumer_indices = consumers.get(out_name, set())
            used_outside = any(
                model.graph.node[idx].metadata_props and node_instances[idx] is None for idx in consumer_indices
            )
            if primary_output is None and (out_name in graph_outputs or used_outside):
                boundary_outputs_set.add(out_name)
                boundary_outputs.append(out_name)
                primary_output = out_name
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

        normalized_inputs = tuple(
            (
                "hidden_state" if hidden_input and name == hidden_input else _normalize_name(name),
                value_types.get(name, 0),
            )
            for name in ordered_inputs
        )
        normalized_outputs = tuple((_normalize_name(name), value_types.get(name, 0)) for name in boundary_outputs)
        signature = (normalized_inputs, normalized_outputs)
        if signature not in signature_to_name:
            class_index = class_name_counts.get(matched_class, 0)
            function_name = _make_function_name(matched_class, class_index)
            class_name_counts[matched_class] = class_index + 1
            signature_to_name[signature] = function_name
            signature_templates[signature] = {
                "nodes": function_nodes,
                "inputs": ordered_inputs,
                "outputs": boundary_outputs,
            }

        group_entries.append(
            {
                "indices": sorted(group_index_set),
                "boundary_inputs": ordered_inputs,
                "boundary_outputs": boundary_outputs,
                "signature": signature,
                "scope": scope_path,
            }
        )

    group_entries.sort(key=lambda entry: max(entry["indices"]), reverse=True)
    for call_idx, entry in enumerate(group_entries):
        signature = entry["signature"]
        function_name = signature_to_name.get(signature)
        if not function_name:
            continue
        boundary_inputs = entry["boundary_inputs"]
        boundary_outputs = entry["boundary_outputs"]
        scope_path = entry.get("scope")
        scope_index = scope_to_index.get(scope_path) if scope_path else None
        call_id = scope_index if scope_index is not None else call_idx
        call_outputs = [f"{function_name}_out_{call_id}_{i}" for i in range(len(boundary_outputs))]
        replacement_map = dict(zip(boundary_outputs, call_outputs))

        for idx, node in enumerate(model.graph.node):
            if idx in entry["indices"]:
                continue
            for input_idx, inp in enumerate(node.input):
                if inp in replacement_map:
                    node.input[input_idx] = replacement_map[inp]

        for output in model.graph.output:
            if output.name in replacement_map:
                output.name = replacement_map[output.name]

        for idx in sorted(entry["indices"], reverse=True):
            del model.graph.node[idx]

        call_node = helper.make_node(function_name, boundary_inputs, call_outputs, domain=function_domain)
        insert_index = min(entry["indices"])
        model.graph.node.insert(insert_index, call_node)

    for signature, template in signature_templates.items():
        _ensure_function_domain_import(model, function_domain)
        func_nodes = [copy.deepcopy(node) for node in template["nodes"]]
        function_proto = helper.make_function(
            function_domain,
            signature_to_name[signature],
            template["inputs"],
            template["outputs"],
            func_nodes,
            opset_imports=model.opset_import,
        )
        model.functions.append(function_proto)

    for fn in model.functions:
        _topo_sort_function(fn)
    _topo_sort_graph(model.graph)

    if model.functions:
        used = _collect_used_functions(model)
        kept = [fn for fn in model.functions if (fn.domain, fn.name) in used]
        del model.functions[:]
        model.functions.extend(kept)

    return model
