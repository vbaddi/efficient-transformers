# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import ast
import copy
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import onnx
from onnx import helper

_NAME_SCOPES_KEY = "pkg.torch.onnx.name_scopes"
_CLASS_HIERARCHY_KEY = "pkg.torch.onnx.class_hierarchy"

# ============================================================================
# Metadata and Scope Parsing
# ============================================================================


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
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed if str(item)]
        if isinstance(parsed, str):
            return [item for item in parsed.split("/") if item]
        if parsed is not None:
            return [str(parsed)]
    except (ValueError, SyntaxError):
        pass
    return [item for item in value.split("/") if item]


def _get_scopes(node: onnx.NodeProto) -> Tuple[List[str], List[str]]:
    class_hierarchy = _get_metadata_value(node, _CLASS_HIERARCHY_KEY)
    name_scopes = _get_metadata_value(node, _NAME_SCOPES_KEY)
    return _parse_scopes(class_hierarchy), _parse_scopes(name_scopes)


def _get_instance_key_scopes(name_scopes: List[str]) -> Optional[List[str]]:
    patterns = [r"(.*\.h\.\d+)", r"(.*\.layers\.\d+)", r"(.*\.layer\.\d+)", r"(.*\.blocks?\.\d+)"]
    for idx, scope in enumerate(name_scopes):
        for pattern in patterns:
            match = re.search(pattern, scope)
            if match:
                return name_scopes[:idx] + [match.group(1)]
    return None


# ============================================================================
# Naming and Identification Utilities
# ============================================================================


def _make_function_name(target_classname: str, index: int) -> str:
    return target_classname if index == 0 else f"{target_classname}_{index}"


def _normalize_name(name: str) -> str:
    name = re.sub(r"\.h\.\d+", ".h.#", name)
    name = re.sub(r"\.\d+", ".#", name)
    return re.sub(r"\d+", "#", name)


def _parse_block_index(scope_path: str) -> Optional[int]:
    patterns = [r"\.h\.(\d+)", r"\.layers\.(\d+)", r"\.layer\.(\d+)", r"\.blocks?\.(\d+)"]
    for pattern in patterns:
        match = re.search(pattern, scope_path)
        if match:
            return int(match.group(1))
    return None


def _is_past_name(name: str) -> bool:
    if name.endswith("_RetainedState") or "_InternalRetainedState" in name:
        return True
    tokens = ("past_key", "past_value", "past_key_values", "key_cache", "value_cache")
    return any(token in name for token in tokens)


def _is_standard_domain(domain: str) -> bool:
    return domain in ("", "ai.onnx")


# ============================================================================
# Graph Analysis Utilities
# ============================================================================


def _collect_consumers(nodes: Iterable[onnx.NodeProto]) -> Dict[str, Set[int]]:
    consumers = {}
    for idx, node in enumerate(nodes):
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, set()).add(idx)
    return consumers


def _collect_producers(nodes: Iterable[onnx.NodeProto]) -> Dict[str, int]:
    producers = {}
    for idx, node in enumerate(nodes):
        for out in node.output:
            if out:
                producers[out] = idx
    return producers


def _collect_value_types(model: onnx.ModelProto) -> Dict[str, int]:
    types = {}
    for inp in model.graph.input:
        if inp.type.tensor_type.HasField("elem_type"):
            types[inp.name] = inp.type.tensor_type.elem_type
    for vi in model.graph.value_info:
        if vi.type.tensor_type.HasField("elem_type"):
            types[vi.name] = vi.type.tensor_type.elem_type
    for init in model.graph.initializer:
        types[init.name] = init.data_type
    return types


def _collect_value_info(model: onnx.ModelProto) -> Dict[str, onnx.ValueInfoProto]:
    infos = {}
    for item in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        infos[item.name] = item
    return infos


def _ensure_initializer_value_info(model: onnx.ModelProto) -> None:
    existing = (
        {vi.name for vi in model.graph.value_info}
        | {inp.name for inp in model.graph.input}
        | {out.name for out in model.graph.output}
    )
    for init in model.graph.initializer:
        if init.name not in existing:
            model.graph.value_info.append(helper.make_tensor_value_info(init.name, init.data_type, init.dims))


# ============================================================================
# Shape Node Detection
# ============================================================================


def _collect_shape_node_indices(model: onnx.ModelProto, max_scalar_size: int = 16) -> Set[int]:
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

    init_sizes = {init.name: max(1, int(sum(init.dims) if init.dims else 0)) for init in model.graph.initializer}
    shape_values = {out for node in model.graph.node if node.op_type in shape_seed_ops for out in node.output if out}

    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            if node.op_type in shape_ops and node.output:
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
        if any(out in shape_values for out in node.output if out):
            if all(
                model.graph.node[c_idx].op_type in shape_ops
                for out in node.output
                if out
                for c_idx in consumers.get(out, set())
            ):
                shape_node_indices.add(idx)
    return shape_node_indices


# ============================================================================
# Topological Sorting
# ============================================================================


def _topo_sort_nodes(nodes: List[onnx.NodeProto], available_inputs: Set[str]) -> List[onnx.NodeProto]:
    available = set(available_inputs)
    sorted_nodes, remaining = [], list(nodes)

    while remaining:
        progress, next_remaining = False, []
        for node in remaining:
            if all((not inp) or inp in available for inp in node.input):
                sorted_nodes.append(node)
                available.update(out for out in node.output if out)
                progress = True
            else:
                next_remaining.append(node)
        if not progress:
            sorted_nodes.extend(next_remaining)
            break
        remaining = next_remaining
    return sorted_nodes


def _topo_sort_graph(graph: onnx.GraphProto) -> None:
    available = {inp.name for inp in graph.input} | {init.name for init in graph.initializer}
    sorted_nodes = _topo_sort_nodes(list(graph.node), available)
    del graph.node[:]
    graph.node.extend(sorted_nodes)


def _topo_sort_function(fn: onnx.FunctionProto) -> None:
    sorted_nodes = _topo_sort_nodes(list(fn.node), set(fn.input))
    del fn.node[:]
    fn.node.extend(sorted_nodes)


# ============================================================================
# Group Expansion
# ============================================================================


def _expand_group_indices(
    nodes: Sequence[onnx.NodeProto],
    indices: Sequence[int],
    producers: Dict[str, int],
    consumers: Dict[str, Set[int]],
    graph_inputs: Set[str],
    graph_outputs: Set[str],
    node_instances: Sequence[Optional[str]],
    scope_path: str,
) -> Set[int]:
    expanded = set(indices)
    changed = True
    while changed:
        changed = False
        for idx in list(expanded):
            for inp in nodes[idx].input:
                if not inp or inp in graph_inputs:
                    continue
                prod_idx = producers.get(inp)
                if prod_idx is None or prod_idx in expanded:
                    continue
                prod_scope = node_instances[prod_idx]
                if prod_scope != scope_path and prod_scope is not None:
                    continue
                prod_node = nodes[prod_idx]
                only_used_by_group = all(
                    out in graph_outputs or all(c in expanded for c in consumers.get(out, set()))
                    for out in prod_node.output
                    if out
                )
                if only_used_by_group:
                    expanded.add(prod_idx)
                    changed = True
    return expanded


# ============================================================================
# Model Utilities
# ============================================================================


def _ensure_function_domain_import(model: onnx.ModelProto, function_domain: str) -> None:
    if not any(opset.domain == function_domain for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid(function_domain, 1))


def _collect_used_functions(model: onnx.ModelProto) -> Set[Tuple[str, str]]:
    used, queue = set(), [(node.domain, node.op_type) for node in model.graph.node]
    functions_by_key = {(fn.domain, fn.name): fn for fn in model.functions}

    while queue:
        key = queue.pop()
        if key in used:
            continue
        used.add(key)
        fn = functions_by_key.get(key)
        if fn:
            queue.extend((fn_node.domain, fn_node.op_type) for fn_node in fn.node)
    return used


def _collect_function_required_inputs(fn: onnx.FunctionProto) -> List[str]:
    produced = {out for node in fn.node for out in node.output if out}
    required, seen = [], set()
    for node in fn.node:
        for inp in node.input:
            if inp and inp not in produced and inp not in seen:
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
            if node.domain == function_domain and node.op_type == fn.name:
                new_inputs = []
                for inp_name in required_inputs:
                    idx = old_index.get(inp_name)
                    new_inputs.append(node.input[idx] if idx is not None and idx < len(node.input) else inp_name)
                node.input[:] = new_inputs


def _swap_value_names(model: onnx.ModelProto, name_a: str, name_b: str) -> None:
    swap = lambda n: name_b if n == name_a else (name_a if n == name_b else n)

    for item in (
        list(model.graph.input)
        + list(model.graph.output)
        + list(model.graph.value_info)
        + list(model.graph.initializer)
    ):
        if item.name in {name_a, name_b}:
            item.name = swap(item.name)

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

    pos_type, pk_type = inputs["position_ids"], inputs["past_key.0"]
    pos_elem = pos_type.elem_type if pos_type.HasField("elem_type") else None
    pk_elem = pk_type.elem_type if pk_type.HasField("elem_type") else None

    if (
        pos_elem == onnx.TensorProto.FLOAT
        and pk_elem == onnx.TensorProto.INT64
        and len(pos_type.shape.dim) > len(pk_type.shape.dim)
    ):
        _swap_value_names(model, "position_ids", "past_key.0")
    return model


# ============================================================================
# Scope Matching Helper
# ============================================================================


def _match_scope_and_class(
    class_scopes: List[str], name_scopes: List[str], target_module_classnames: Sequence[str]
) -> Tuple[Optional[str], Optional[List[str]]]:
    """Match class and determine key scopes."""
    matched_class, key_scopes = None, None

    # Try class scopes first
    if class_scopes:
        match_idx = next(
            (i for i, scope in enumerate(class_scopes) if any(cls in scope for cls in target_module_classnames)), None
        )
        if match_idx is not None:
            matched_class = next((cls for cls in target_module_classnames if cls in class_scopes[match_idx]), None)
            instance_scopes = _get_instance_key_scopes(name_scopes)
            if instance_scopes:
                key_scopes = instance_scopes
            elif name_scopes and len(name_scopes) > match_idx:
                key_scopes = name_scopes[: match_idx + 1]
            else:
                key_scopes = class_scopes[: match_idx + 1]
            return matched_class, key_scopes

    # Try name scopes
    match_idx = next(
        (i for i, scope in enumerate(name_scopes) if any(cls in scope for cls in target_module_classnames)), None
    )
    if match_idx is not None:
        matched_class = next((cls for cls in target_module_classnames if cls in name_scopes[match_idx]), None)
        instance_scopes = _get_instance_key_scopes(name_scopes)
        key_scopes = instance_scopes if instance_scopes else (name_scopes[: match_idx + 1] if name_scopes else [])

    return matched_class, key_scopes


# ============================================================================
# Main Function Extraction Logic
# ============================================================================


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

    # Parse scopes for all nodes
    decoder_like = any("DecoderLayer" in cls or cls.endswith("Block") for cls in target_module_classnames)
    class_scopes_list, name_scopes_list = [], []
    for node in model.graph.node:
        class_scopes, name_scopes = _get_scopes(node)
        class_scopes_list.append(class_scopes)
        name_scopes_list.append(name_scopes)

    # Determine node instances
    node_instances = []
    for class_scopes, name_scopes in zip(class_scopes_list, name_scopes_list):
        match_idx = next(
            (i for i, scope in enumerate(class_scopes) if any(cls in scope for cls in target_module_classnames)), None
        )
        if match_idx is None:
            match_idx = next(
                (i for i, scope in enumerate(name_scopes) if any(cls in scope for cls in target_module_classnames)),
                None,
            )

        if match_idx is None:
            instance_scopes = _get_instance_key_scopes(name_scopes)
            node_instances.append("/".join(instance_scopes) if instance_scopes and decoder_like else None)
            continue

        instance_scopes = _get_instance_key_scopes(name_scopes)
        if instance_scopes:
            key_scopes = instance_scopes
        elif name_scopes and len(name_scopes) > match_idx:
            key_scopes = name_scopes[: match_idx + 1]
        else:
            key_scopes = class_scopes[: match_idx + 1]
        node_instances.append("/".join(key_scopes) if key_scopes else None)

    # Collect graph metadata
    shape_node_indices = _collect_shape_node_indices(model)
    graph_outputs = {output.name for output in model.graph.output}
    graph_inputs = {inp.name for inp in model.graph.input}
    initializer_map = {init.name: init for init in model.graph.initializer}
    graph_inputs.update(initializer_map.keys())

    producers = _collect_producers(model.graph.node)
    consumers = _collect_consumers(model.graph.node)
    value_types = _collect_value_types(model)
    value_info_map = _collect_value_info(model)

    # Find values reaching non-past outputs
    non_past_outputs = [output.name for output in model.graph.output if not _is_past_name(output.name)]
    values_reaching_outputs = set(non_past_outputs)
    stack = list(non_past_outputs)
    while stack:
        value = stack.pop()
        prod_idx = producers.get(value)
        if prod_idx is not None:
            for inp in model.graph.node[prod_idx].input:
                if inp and inp not in values_reaching_outputs:
                    values_reaching_outputs.add(inp)
                    stack.append(inp)

    # Group nodes by target class and scope
    groups = []
    for idx, node in enumerate(model.graph.node):
        if idx in shape_node_indices:
            continue

        matched_class, key_scopes = _match_scope_and_class(
            class_scopes_list[idx], name_scopes_list[idx], target_module_classnames
        )

        if not key_scopes or matched_class is None:
            continue

        scope_path = "/".join(key_scopes)
        groups.append({"class": matched_class, "scope": scope_path, "idx": idx})

    if not groups:
        return model

    # Group by (class, scope)
    grouped_indices = {}
    for entry in groups:
        key = (entry["class"], entry["scope"])
        grouped_indices.setdefault(key, []).append(entry["idx"])

    # Parse block indices
    scope_to_index = {
        scope: _parse_block_index(scope) for _, scope in grouped_indices.keys() if _parse_block_index(scope) is not None
    }
    index_to_scope = {idx: scope for scope, idx in scope_to_index.items()}

    # Find hidden inputs and primary outputs
    norm_tokens = (
        "ln_1",
        "input_layernorm",
        "input_layer_norm",
        "pre_attention_layernorm",
        "rms_norm",
        "layer_norm",
        "norm",
    )
    norm_op_types = {"LayerNormalization", "RMSNormalization", "RMSNorm", "CustomRMSNorm"}

    hidden_input_by_scope = {}
    for scope in grouped_indices.keys():
        scope_path = scope[1]
        for idx, node in enumerate(model.graph.node):
            if not _is_standard_domain(node.domain) or node_instances[idx] != scope_path:
                continue
            name_scopes, class_scopes = name_scopes_list[idx], class_scopes_list[idx]
            if (
                any(token in scope_name for token in norm_tokens for scope_name in name_scopes + class_scopes)
                or node.op_type in norm_op_types
            ):
                if node.input:
                    hidden_input_by_scope[scope_path] = node.input[0]
                break

    primary_output_by_scope = {}
    for scope_path, block_index in scope_to_index.items():
        next_scope = index_to_scope.get(block_index + 1)
        if next_scope and next_scope in hidden_input_by_scope:
            primary_output_by_scope[scope_path] = hidden_input_by_scope[next_scope]

    # Build function signatures and templates
    signature_to_name, signature_templates = {}, {}
    class_name_counts = {}
    group_entries = []

    for (matched_class, scope_path), indices in grouped_indices.items():
        expanded_indices = _expand_group_indices(
            model.graph.node,
            sorted(indices),
            producers,
            consumers,
            graph_inputs,
            graph_outputs,
            node_instances,
            scope_path,
        )

        # Add all nodes in scope
        for idx, scope in enumerate(node_instances):
            if scope == scope_path and idx not in shape_node_indices:
                expanded_indices.add(idx)

        if len(expanded_indices) < min_nodes:
            continue

        group_nodes = [model.graph.node[i] for i in sorted(expanded_indices)]

        # Collect outputs
        outputs_set, outputs_order = set(), []
        for node in group_nodes:
            for out_name in node.output:
                if out_name and out_name not in outputs_set:
                    outputs_set.add(out_name)
                    outputs_order.append(out_name)

        # Determine primary output
        primary_output = primary_output_by_scope.get(scope_path)
        if primary_output is None:
            for out_name in reversed(outputs_order):
                if not out_name or _is_past_name(out_name):
                    continue
                consumer_indices = consumers.get(out_name, set())
                if out_name in graph_outputs or any(idx not in expanded_indices for idx in consumer_indices):
                    primary_output = out_name
                    break

        if primary_output is None:
            for out_name in outputs_order:
                if out_name in values_reaching_outputs and not _is_past_name(out_name):
                    primary_output = out_name
                    break

        # Collect inputs
        function_inputs = []
        for node in group_nodes:
            for inp in node.input:
                if inp and inp not in outputs_set and inp not in function_inputs:
                    function_inputs.append(inp)

        # Collect boundary outputs
        past_outputs = [name for name in outputs_order if _is_past_name(name)]
        boundary_outputs = list(past_outputs)
        boundary_outputs_set = set(boundary_outputs)

        for out_name in outputs_order:
            if out_name in boundary_outputs_set:
                continue
            consumer_indices = consumers.get(out_name, set())
            used_outside = any(idx not in expanded_indices for idx in consumer_indices)
            if (primary_output and out_name == primary_output) or out_name in graph_outputs or used_outside:
                boundary_outputs_set.add(out_name)
                boundary_outputs.append(out_name)

        if not boundary_outputs:
            continue

        # Order inputs
        hidden_input = hidden_input_by_scope.get(scope_path)
        ordered_inputs = []
        if hidden_input and hidden_input in function_inputs:
            ordered_inputs.append(hidden_input)

        past_inputs = sorted([name for name in function_inputs if _is_past_name(name)], key=_normalize_name)
        ordered_inputs.extend(past_inputs)

        remaining_inputs = sorted([name for name in function_inputs if name not in ordered_inputs], key=_normalize_name)
        ordered_inputs.extend(remaining_inputs)

        # Create signature
        hidden_sig_input = hidden_input if hidden_input in ordered_inputs else None
        if hidden_sig_input is None:
            skip_hidden = {"input_ids", "position_ids", "attention_mask", "token_type_ids", "input_mask"}
            for name in ordered_inputs:
                if name not in skip_hidden and not _is_past_name(name) and name not in initializer_map:
                    hidden_sig_input = name
                    break

        normalized_inputs = tuple(
            (
                "hidden_state" if hidden_sig_input and name == hidden_sig_input else _normalize_name(name),
                value_types.get(name, 0),
            )
            for name in ordered_inputs
        )

        normalized_outputs = tuple(
            (
                "hidden_state_out" if primary_output and name == primary_output else _normalize_name(name),
                value_types.get(name, 0),
            )
            for name in boundary_outputs
        )

        signature = (matched_class, normalized_inputs, normalized_outputs)

        if signature not in signature_to_name:
            class_index = class_name_counts.get(matched_class, 0)
            function_name = _make_function_name(matched_class, class_index)
            class_name_counts[matched_class] = class_index + 1
            signature_to_name[signature] = function_name
            signature_templates[signature] = {
                "nodes": group_nodes,
                "inputs": ordered_inputs,
                "outputs": boundary_outputs,
            }

        group_entries.append(
            {
                "indices": sorted(expanded_indices),
                "boundary_inputs": ordered_inputs,
                "boundary_outputs": boundary_outputs,
                "signature": signature,
                "scope": scope_path,
            }
        )

    # Replace groups with function calls
    group_entries.sort(key=lambda e: max(e["indices"]), reverse=True)

    for call_idx, entry in enumerate(group_entries):
        function_name = signature_to_name.get(entry["signature"])
        if not function_name:
            continue

        scope_path = entry.get("scope")
        scope_index = scope_to_index.get(scope_path) if scope_path else None
        call_id = scope_index if scope_index is not None else call_idx

        if preserve_graph:
            call_outputs = [f"{function_name}_out_{call_id}_{i}" for i in range(len(entry["boundary_outputs"]))]
        else:
            call_outputs = list(entry["boundary_outputs"])

        replacement_map = dict(zip(entry["boundary_outputs"], call_outputs))

        if not preserve_graph:
            # Add value info for new outputs
            existing_vi_names = {vi.name for vi in model.graph.value_info} | {
                output.name for output in model.graph.output
            }
            new_value_infos = []

            for old_name, new_name in replacement_map.items():
                if new_name in existing_vi_names:
                    continue
                vi = value_info_map.get(old_name)
                if vi:
                    vi_copy = copy.deepcopy(vi)
                    vi_copy.name = new_name
                    new_value_infos.append(vi_copy)
                else:
                    elem_type = value_types.get(old_name)
                    if elem_type:
                        new_value_infos.append(helper.make_tensor_value_info(new_name, elem_type, None))
                existing_vi_names.add(new_name)

            if new_value_infos:
                model.graph.value_info.extend(new_value_infos)

            # Update references
            for idx, node in enumerate(model.graph.node):
                if idx not in entry["indices"]:
                    for i, inp in enumerate(node.input):
                        if inp in replacement_map:
                            node.input[i] = replacement_map[inp]

            for output in model.graph.output:
                if output.name in replacement_map:
                    output.name = replacement_map[output.name]

            # Remove replaced nodes and insert call node
            for idx in sorted(entry["indices"], reverse=True):
                del model.graph.node[idx]

            producer_index = {out: idx for idx, node in enumerate(model.graph.node) for out in node.output}
            max_producer = max((producer_index.get(inp, -1) for inp in entry["boundary_inputs"] if inp), default=-1)
            insert_index = max_producer + 1

            call_node = helper.make_node(function_name, entry["boundary_inputs"], call_outputs, domain=function_domain)
            model.graph.node.insert(insert_index, call_node)
        else:
            call_node = helper.make_node(function_name, entry["boundary_inputs"], call_outputs, domain=function_domain)
            model.graph.node.append(call_node)

    # Create function protos
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

    # Clean up
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
