# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""ONNX rewrites used by the DeepSeek V4 prefill export path."""

from numbers import Integral

import onnx
from onnx import numpy_helper

from QEfficient.base.onnx_transforms import BaseOnnxTransform


class DeepseekV4StaticPrefillLoopTransform(BaseOnnxTransform):
    """Give export-time ``torch.while_loop`` nodes fixed trip counts.

    PyTorch emits a missing ONNX trip-count input for condition-controlled
    loops.  QAIC can compile the loop body more reliably when the compile-time
    trip count is explicit.  The caller supplies counts in the same order as
    the dynamic loops appear in the exported graph.
    """

    _apply_with_pipeline_kwargs = True
    _VALUE_PREFIX = "qeff_deepseek_v4_prefill_loop"

    @staticmethod
    def _body(node: onnx.NodeProto) -> onnx.GraphProto | None:
        return next((attr.g for attr in node.attribute if attr.name == "body" and attr.HasField("g")), None)

    @classmethod
    def _used_names(cls, graph: onnx.GraphProto) -> set[str]:
        used = {value.name for value in (*graph.input, *graph.output, *graph.value_info, *graph.initializer)}
        for node in graph.node:
            used.update(name for name in (*node.input, *node.output) if name)
        return used

    @staticmethod
    def _used_function_names(function: onnx.FunctionProto) -> set[str]:
        used = set(function.input) | set(function.output)
        for node in function.node:
            used.update(name for name in (*node.input, *node.output) if name)
        return used

    @staticmethod
    def _unique(base: str, used: set[str]) -> str:
        name = base
        suffix = 0
        while name in used:
            suffix += 1
            name = f"{base}_{suffix}"
        used.add(name)
        return name

    @classmethod
    def _rewrite_body(cls, body: onnx.GraphProto, index: int) -> None:
        if not body.output:
            raise ValueError("DeepSeek V4 prefill Loop body has no condition output.")
        used = cls._used_names(body)
        cond_name = cls._unique(f"{cls._VALUE_PREFIX}_{index}_body_cond_true", used)
        body.initializer.append(onnx.helper.make_tensor(cond_name, onnx.TensorProto.BOOL, [], [True]))
        body.output[0].name = cond_name

    @classmethod
    def _rewrite_loop(
        cls,
        node: onnx.NodeProto,
        trip_count: int,
        index: int,
        used: set[str],
    ) -> tuple[onnx.NodeProto, list[onnx.NodeProto]]:
        body = cls._body(node)
        if body is None:
            raise ValueError("DeepSeek V4 prefill Loop has no body graph.")
        cls._rewrite_body(body, index)
        trip_name = cls._unique(f"{cls._VALUE_PREFIX}_{index}_trip_count", used)
        cond_name = cls._unique(f"{cls._VALUE_PREFIX}_{index}_cond_true", used)
        node.input[0] = trip_name
        node.input[1] = cond_name
        constants = [
            onnx.helper.make_node(
                "Constant",
                [],
                [trip_name],
                value=onnx.helper.make_tensor(f"{trip_name}_value", onnx.TensorProto.INT64, [], [trip_count]),
            ),
            onnx.helper.make_node(
                "Constant",
                [],
                [cond_name],
                value=onnx.helper.make_tensor(f"{cond_name}_value", onnx.TensorProto.BOOL, [], [True]),
            ),
        ]
        return node, constants

    @classmethod
    def _rewrite_nodes(
        cls,
        nodes,
        trip_counts: tuple[int, ...],
        index: int,
        used: set[str],
    ) -> tuple[list[onnx.NodeProto], int, bool]:
        rewritten = []
        changed = False
        for node in nodes:
            body = cls._body(node) if node.op_type == "Loop" else None
            if body is None or not node.input or node.input[0]:
                rewritten.append(node)
                continue
            if index >= len(trip_counts):
                raise ValueError("Not enough DeepSeek V4 prefill Loop trip counts for exported loops.")
            node, constants = cls._rewrite_loop(node, trip_counts[index], index, used)
            rewritten.extend(constants)
            rewritten.append(node)
            index += 1
            changed = True
        return rewritten, index, changed

    @classmethod
    def _rewrite_graph(
        cls,
        graph: onnx.GraphProto,
        trip_counts: tuple[int, ...],
        index: int,
    ) -> tuple[int, bool]:
        used = cls._used_names(graph)
        changed = False
        for node in graph.node:
            for attr in node.attribute:
                nested = [attr.g] if attr.HasField("g") else []
                nested.extend(attr.graphs)
                for child in nested:
                    index, nested_changed = cls._rewrite_graph(child, trip_counts, index)
                    changed |= nested_changed
        rewritten, index, changed_nodes = cls._rewrite_nodes(graph.node, trip_counts, index, used)
        if changed_nodes:
            del graph.node[:]
            graph.node.extend(rewritten)
            changed |= changed_nodes
        return index, changed

    @classmethod
    def _rewrite_function(
        cls,
        function: onnx.FunctionProto,
        trip_counts: tuple[int, ...],
        index: int,
    ) -> tuple[int, bool]:
        used = cls._used_function_names(function)
        changed = False
        for node in function.node:
            for attr in node.attribute:
                nested = [attr.g] if attr.HasField("g") else []
                nested.extend(attr.graphs)
                for child in nested:
                    index, nested_changed = cls._rewrite_graph(child, trip_counts, index)
                    changed |= nested_changed
        rewritten, index, changed_nodes = cls._rewrite_nodes(function.node, trip_counts, index, used)
        if changed_nodes:
            del function.node[:]
            function.node.extend(rewritten)
            changed |= changed_nodes
        return index, changed

    @classmethod
    def _dynamic_loops(cls, graph: onnx.GraphProto) -> list[onnx.NodeProto]:
        loops = []
        for node in graph.node:
            if node.op_type == "Loop" and node.input and not node.input[0] and cls._body(node) is not None:
                loops.append(node)
            for attr in node.attribute:
                nested = [attr.g] if attr.HasField("g") else []
                nested.extend(attr.graphs)
                for child in nested:
                    loops.extend(cls._dynamic_loops(child))
        return loops

    @classmethod
    def _dynamic_function_loops(cls, function: onnx.FunctionProto) -> list[onnx.NodeProto]:
        loops = [node for node in function.node if node.op_type == "Loop" and node.input and not node.input[0]]
        for node in function.node:
            for attr in node.attribute:
                nested = [attr.g] if attr.HasField("g") else []
                nested.extend(attr.graphs)
                for child in nested:
                    loops.extend(cls._dynamic_loops(child))
        return loops

    @classmethod
    def _infer_trip_count(cls, node: onnx.NodeProto, functions: list[onnx.FunctionProto]) -> int | None:
        body = cls._body(node)
        if body is None:
            return None
        condition_functions = {function.name: function for function in functions if "cond_graph" in function.name}
        for body_node in body.node:
            function = condition_functions.get(body_node.op_type)
            if function is None:
                continue
            for function_node in function.node:
                if function_node.op_type != "Constant":
                    continue
                tensor = next((attr.t for attr in function_node.attribute if attr.name == "value"), None)
                if tensor is None:
                    continue
                value = numpy_helper.to_array(tensor)
                if value.ndim == 0 and int(value) > 0:
                    return int(value)
        return None

    @classmethod
    def apply(cls, model: onnx.ModelProto, *, deepseek_v4_prefill_loop_trip_counts=None, **kwargs) -> bool:
        del kwargs
        dynamic_loops = cls._dynamic_loops(model.graph)
        for function in model.functions:
            dynamic_loops.extend(cls._dynamic_function_loops(function))
        if deepseek_v4_prefill_loop_trip_counts is None:
            # This transform is registered on the shared causal-LM wrapper;
            # without the DeepSeek prefill marker it must be a no-op for other
            # model families that may also contain condition-controlled loops.
            return False
        elif isinstance(deepseek_v4_prefill_loop_trip_counts, Integral) and not isinstance(
            deepseek_v4_prefill_loop_trip_counts, bool
        ):
            trip_counts = (int(deepseek_v4_prefill_loop_trip_counts),)
        else:
            trip_counts = tuple(deepseek_v4_prefill_loop_trip_counts)
        if not trip_counts or any(
            isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 for value in trip_counts
        ):
            raise ValueError("DeepSeek V4 prefill Loop trip counts must be positive integers.")

        # The export caller may know the MoE loop counts before ONNX capture,
        # while the graph can also contain attention/indexer loops discovered
        # only after capture.  Prefer the explicit list when it covers every
        # dynamic loop; otherwise infer the complete list from each loop's
        # condition function.
        if len(trip_counts) != len(dynamic_loops):
            inferred_counts = tuple(cls._infer_trip_count(node, list(model.functions)) for node in dynamic_loops)
            if len(inferred_counts) != len(dynamic_loops) or any(value is None for value in inferred_counts):
                raise ValueError(
                    f"DeepSeek V4 prefill export supplied {len(trip_counts)} Loop trip counts but found "
                    f"{len(dynamic_loops)} loops, and the missing counts could not be inferred."
                )
            trip_counts = tuple(inferred_counts)

        index, changed = cls._rewrite_graph(model.graph, trip_counts, 0)
        for function in model.functions:
            index, function_changed = cls._rewrite_function(function, trip_counts, index)
            changed |= function_changed
        if index != len(trip_counts):
            raise ValueError(
                f"DeepSeek V4 prefill export supplied {len(trip_counts)} Loop trip counts but found {index} loops."
            )
        return changed
