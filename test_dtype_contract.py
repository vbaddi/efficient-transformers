# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import collections
from pathlib import Path

import numpy as np
import onnx
import onnxscript
import torch
from onnx.reference import ReferenceEvaluator

from QEfficient.customop.onnxscript_utils import get_dynamo_onnxscript_func, qeff_custom_op
from QEfficient.utils import constants

DOMAIN = "qefficient.precision"
CTX = 131072
BLOCKS = 64
HEADS = 2
DIM = 4
OUT = Path("/tmp/qeff_dtype_contract_poc")
OUT.mkdir(exist_ok=True)
OPS = getattr(onnxscript, "opset" + str(constants.ONNX_LEGACY_EXPORT_OPSET))


@qeff_custom_op(DOMAIN, 1)
def PrecisionHint(x: onnxscript.FLOAT, region: str, role: str, precision: str) -> onnxscript.FLOAT:
    return OPS.Identity(x)


@torch.library.custom_op("qefficient::precision_hint_poc", mutates_args=())
def precision_hint_poc(x: torch.Tensor, region: str, role: str, precision: str) -> torch.Tensor:
    return x.clone()


@precision_hint_poc.register_fake
def _(x, region, role, precision):
    return torch.empty_like(x)


class Region:
    def __init__(self, name):
        self.name = name
        self.outer_dtype = None

    def enter(self, *xs):
        out = []
        for x in xs:
            self.outer_dtype = self.outer_dtype or x.dtype
            x = x.float()
            x = torch.ops.qefficient.precision_hint_poc(x, self.name, "begin", "fp32")
            out.append(x)
        return out[0] if len(out) == 1 else tuple(out)

    def exit(self, x):
        x = torch.ops.qefficient.precision_hint_poc(x, self.name, "end", "fp32")
        return x.to(self.outer_dtype)


class Merge(torch.nn.Module):
    def __init__(self, use_region):
        super().__init__()
        self.use_region = use_region

    def forward(self, scores, values):
        region = Region("merge") if self.use_region else None
        block_size = scores.shape[-1] // BLOCKS
        maxima, sums, outputs = [], [], []
        for block in range(BLOCKS):
            sl = scores[:, block * block_size : (block + 1) * block_size]
            m = sl.max(dim=-1).values
            e = torch.exp(sl - m.unsqueeze(-1))
            block_sum = e.sum(dim=-1)
            o = torch.bmm(e.unsqueeze(1), values[:, block * block_size : (block + 1) * block_size]).squeeze(1)
            if region is not None:
                m, block_sum, o = region.enter(m, block_sum, o)
            maxima.append(m)
            sums.append(block_sum)
            outputs.append(o)
        maxima, sums, outputs = torch.stack(maxima), torch.stack(sums), torch.stack(outputs)
        global_max = maxima.max(dim=0).values
        weights = torch.exp(maxima - global_max)
        denominator = (weights * sums).sum(dim=0)
        numerator = (weights.unsqueeze(-1) * outputs).sum(dim=0)
        result = numerator / denominator.unsqueeze(-1)
        return region.exit(result) if region is not None else result


def export_dynamo(model, path, args):
    program = torch.onnx.export(
        model.eval(),
        args,
        dynamo=True,
        optimize=False,
        custom_translation_table={
            torch.ops.qefficient.precision_hint_poc.default: get_dynamo_onnxscript_func(PrecisionHint)
        },
        input_names=["scores", "values"],
        output_names=["out"],
    )
    program.save(str(path))
    return onnx.load(path)


def clone_model(model):
    return onnx.load_model_from_string(model.SerializeToString())


def _attribute(node, name):
    for attribute in node.attribute:
        if attribute.name == name:
            return attribute.s.decode()
    raise KeyError(f"{node.name or node.op_type}: missing attribute {name}")


def prepare_compiler_input(model, annotate):
    model = clone_model(model)
    nodes = list(model.graph.node)
    producer = {output: node for node in nodes for output in node.output if output}
    consumers = collections.defaultdict(list)
    markers = [node for node in nodes if node.domain == DOMAIN and node.op_type == "PrecisionHint"]
    begins = [node for node in markers if _attribute(node, "role") == "begin"]
    ends = [node for node in markers if _attribute(node, "role") == "end"]

    for node in nodes:
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)

    begin_ids = {id(node) for node in begins}
    end_ids = {id(node) for node in ends}
    forward = {}
    stack = [node.output[0] for node in begins]
    while stack:
        tensor = stack.pop()
        for consumer in consumers.get(tensor, []):
            if id(consumer) in forward or id(consumer) in begin_ids or id(consumer) in end_ids:
                continue
            forward[id(consumer)] = consumer
            stack.extend(output for output in consumer.output if output)

    backward = {}
    stack = [node.input[0] for node in ends]
    while stack:
        tensor = stack.pop()
        producer_node = producer.get(tensor)
        if producer_node is None or id(producer_node) in backward or id(producer_node) in begin_ids:
            continue
        backward[id(producer_node)] = producer_node
        stack.extend(input_name for input_name in producer_node.input if input_name)

    region_nodes = [node for node in nodes if id(node) in forward and id(node) in backward and node not in markers]
    if annotate:
        for node in region_nodes:
            node.metadata_props.add(key="qaic.precision", value="fp32")
            node.metadata_props.add(key="qaic.precision_region", value="merge")

    replacements = {node.output[0]: node.input[0] for node in markers}
    kept = [node for node in nodes if node not in markers]

    def resolve(name):
        while name in replacements:
            name = replacements[name]
        return name

    for node in kept:
        for index, input_name in enumerate(node.input):
            node.input[index] = resolve(input_name)
    del model.graph.node[:]
    model.graph.node.extend(kept)

    functions = [
        function for function in model.functions if not (function.domain == DOMAIN and function.name == "PrecisionHint")
    ]
    del model.functions[:]
    model.functions.extend(functions)
    opsets = [opset for opset in model.opset_import if opset.domain != DOMAIN]
    del model.opset_import[:]
    model.opset_import.extend(opsets)
    if annotate and "metadata_props" in onnx.NodeProto.DESCRIPTOR.fields_by_name:
        model.ir_version = max(model.ir_version, 10)
    return model, len(markers), len(region_nodes)


def demote_declared_fp32(model):
    model = clone_model(model)
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        for attribute in node.attribute:
            if attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT:
                attribute.i = onnx.TensorProto.FLOAT16
    return model


def main():
    scores = torch.zeros(HEADS, CTX, dtype=torch.float16)
    values = torch.ones(HEADS, CTX, DIM, dtype=torch.float16)
    reference = Merge(False)(scores.double(), values.double()).numpy()
    feeds = {"scores": scores.numpy(), "values": values.numpy()}

    def run(model):
        return ReferenceEvaluator(model).run(None, feeds)[0]

    plain = export_dynamo(Merge(False), OUT / "dynamo_plain.onnx", (scores, values))
    upcast = export_dynamo(Merge(True), OUT / "dynamo_upcast.onnx", (scores, values))
    compiler_input, marker_count, region_count = prepare_compiler_input(upcast, annotate=False)
    compiler_input_metadata, _, metadata_region_count = prepare_compiler_input(upcast, annotate=True)
    forced_fp16 = demote_declared_fp32(compiler_input)

    onnx.save(compiler_input, OUT / "dynamo_compiler_input.onnx")
    onnx.save(compiler_input_metadata, OUT / "dynamo_compiler_input_metadata.onnx")
    onnx.save(forced_fp16, OUT / "dynamo_forced_fp16.onnx")

    plain_output = run(plain)
    upcast_output = run(compiler_input)
    forced_output = run(forced_fp16)
    error = float(np.abs(upcast_output.astype(np.float64) - reference).max())
    metadata_nodes = sum(
        any(prop.key == "qaic.precision" for prop in node.metadata_props) for node in compiler_input_metadata.graph.node
    )
    final_reductions = sum(
        node.op_type == "ReduceSum" and any(prop.key == "qaic.precision" for prop in node.metadata_props)
        for node in compiler_input_metadata.graph.node
    )

    for model in (plain, compiler_input, compiler_input_metadata, forced_fp16):
        onnx.checker.check_model(model)

    print(
        {
            "plain_finite": bool(np.isfinite(plain_output).all()),
            "upcast_finite": bool(np.isfinite(upcast_output).all()),
            "upcast_max_error": error,
            "forced_fp16_finite": bool(np.isfinite(forced_output).all()),
            "markers": marker_count,
            "region_nodes": region_count,
            "metadata_nodes": metadata_nodes,
            "metadata_region_nodes": metadata_region_count,
            "fp32_reduce_sums": final_reductions,
            "remaining_marker_nodes": sum(node.domain == DOMAIN for node in compiler_input_metadata.graph.node),
            "remaining_precision_functions": sum(
                function.domain == DOMAIN for function in compiler_input_metadata.functions
            ),
        }
    )

    assert not np.isfinite(plain_output).all()
    assert np.isfinite(upcast_output).all() and error < 1e-5
    assert not np.isfinite(forced_output).all()
    assert marker_count > 0 and region_count > 0
    assert metadata_nodes == metadata_region_count == region_count
    assert final_reductions == 2
    assert not any(node.domain == DOMAIN for node in compiler_input_metadata.graph.node)
    assert not any(function.domain == DOMAIN for function in compiler_input_metadata.functions)


if __name__ == "__main__":
    main()
