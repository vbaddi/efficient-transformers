# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Weight-free export ONNX structure and ORT parity tests.

All tests export with weight_free=True (set on the QEff model) and use_onnx_subfunctions=True.
Each test validates:
  - ONNX graph structure: _RetainedState outputs, subfunctions, naming
  - Weight-free-specific: unique graph input names, no int64 KV cache inputs
  - HF PT == ORT token parity (weights injected from HF cache via load_weight_free_ort_inputs)

CPU-only. No QAIC hardware required.
"""

from __future__ import annotations

import json
from pathlib import Path

import onnx
import pytest
import torch
from transformers import AutoConfig, Qwen2Config, Qwen2ForCausalLM

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.exporter.weight_free import prepare_mxfp6_checkpoint, resolve_weight_spec_path
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils import get_num_layers_from_config
from QEfficient.utils.run_utils import ApiRunner

from ._helpers import (
    BATCH_SIZE,
    CTX_LEN,
    PROMPT_LEN,
    WEIGHT_FREE_CAUSAL_LM_MODEL_IDS,
    assert_has_subfunctions,
    assert_no_int64_kv_cache_inputs,
    assert_retained_state_outputs,
    assert_subfunction_names_match_decoder_class,
    assert_unique_graph_input_names,
    exported_onnx_path,
    load_hf_model,
    load_tokenizer,
    run_weight_free_ort,
    skip_on_model_fetch_error,
)


@pytest.mark.weight_free
@pytest.mark.weight_free_export
def test_tiny_qwen_mxfp6_weight_free_dynamo_export(tmp_path, monkeypatch):
    """Validate the MXFP6 graph/checkpoint/spec contract on a local seeded Qwen."""
    monkeypatch.setattr(
        "QEfficient.transformers.models.modeling_auto.validate_dynamo_export_requirements",
        lambda _feature: None,
    )
    monkeypatch.setattr(
        "QEfficient.utils.export_utils.validate_dynamo_export_requirements",
        lambda _feature: None,
    )
    source = tmp_path / "source"
    prepared = tmp_path / "prepared"
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    torch.manual_seed(7)
    Qwen2ForCausalLM(config).eval().to(torch.bfloat16).save_pretrained(source, safe_serialization=True)
    prepare_mxfp6_checkpoint(source, prepared, target_dtype=torch.bfloat16)

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(str(prepared), weight_free=True)
    onnx_path = Path(
        qeff_model.export(
            str(tmp_path / "export"),
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    manifest = json.loads((prepared / "qeff_mx_manifest.json").read_text())
    spec = json.loads(onnx_path.with_name("weight_spec.json").read_text())
    assert spec["version"] == 6
    assert (
        json.loads(next(prop.value for prop in onnx_model.metadata_props if prop.key == "com.qti.aisw.extdata")) == spec
    )
    mxfp6_nodes = [
        node for node in onnx_model.graph.node if node.domain == "com.qualcomm.qeff" and node.op_type == "MXDequantize"
    ]
    assert len(mxfp6_nodes) == 14
    attrs = {attribute.name: attribute for attribute in mxfp6_nodes[0].attribute}
    assert attrs["format"].s == b"MXFP6_E2M3"
    assert attrs["axis"].i == -1
    assert attrs["block_size"].i == 32
    assert attrs["axis_size"].i == 64
    assert attrs["output_dtype"].s == b"BFLOAT16"
    assert all(
        {attribute.name: attribute for attribute in node.attribute}["output_dtype"].s == b"BFLOAT16"
        for node in mxfp6_nodes
    )
    assert all(
        key not in {initializer.name for initializer in onnx_model.graph.initializer}
        for key in manifest["selected_keys"]
    )
    graph_inputs = {value.name: value for value in onnx_model.graph.input}
    for key in manifest["selected_keys"]:
        assert graph_inputs[key].type.tensor_type.elem_type == onnx.TensorProto.UINT8
        shape = tuple(dim.dim_value for dim in graph_inputs[key].type.tensor_type.shape.dim)
        assert shape == tuple(manifest["tensors"][key]["physical_shape"])
    onnx.checker.check_model(onnx_model)
    assert (onnx_path.parent / "checkpoint").is_dir()


def test_compile_enable_mxfp6_prepares_from_original_weight_free_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    Qwen2ForCausalLM(config).eval().to(torch.bfloat16).save_pretrained(source, safe_serialization=True)
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(str(source), config=config, weight_free=True)
    captured = {}

    def fake_compile(self, **kwargs):
        captured.update(kwargs)
        return "compiler-not-run"

    monkeypatch.setattr(QEFFBaseModel, "_compile", fake_compile)
    assert (
        qeff_model.compile(
            use_onnx_subfunctions=True,
            dynamo=True,
            enable_mxfp6=True,
        )
        == "compiler-not-run"
    )
    prepared = Path(qeff_model.hash_params["pretrained_model_name_or_path"])
    assert (prepared / "qeff_mx_manifest.json").is_file()
    assert captured["dynamo"] is True
    assert captured["use_onnx_subfunctions"] is True
    assert "enable_mxfp6" not in captured


@pytest.mark.weight_free
@pytest.mark.weight_free_export
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS),
)
def test_weight_free_export_onnx_structure(model_type, model_id, tmp_export_dir):
    """Export with weight_free=True and use_onnx_subfunctions=True, then validate:
    - Correct _RetainedState output count (2 x num_layers)
    - Subfunctions renamed to decoder class names
    - No duplicate graph input names (guards position_ids aliasing regression)
    - No int64 tensor named past_key.X / past_value.X (same regression, dtype check)

    CPU-only. No QAIC hardware or weight injection required.
    """
    try:
        # Build meta-device model — no weights loaded, only shapes.
        # pretrained_model_name_or_path is carried in the QEff model so the export
        # can write weight_spec.json pointing at the HF cache checkpoint.
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = 2
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    # Shared structure checks
    num_layers = qeff_model.model.config.num_hidden_layers
    assert_retained_state_outputs(onnx_path, expected_count=2 * num_layers)
    assert_has_subfunctions(onnx_path, qeff_model)
    assert_subfunction_names_match_decoder_class(onnx_path, qeff_model)

    # Weight-free-specific regression guards
    assert_unique_graph_input_names(onnx_path)
    assert_no_int64_kv_cache_inputs(onnx_path)


@pytest.mark.weight_free
@pytest.mark.weight_free_export
@pytest.mark.parametrize(
    "model_type,model_id",
    sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS.items()),
    ids=sorted(WEIGHT_FREE_CAUSAL_LM_MODEL_IDS),
)
def test_weight_free_export_ort_parity(model_type, model_id, tmp_export_dir):
    """Export with weight_free=True, inject real weights into ORT, then validate
    HF PyTorch == ORT token parity.

    Unlike regular ONNX (which embeds weights), weight-free ONNX stores weights externally
    in a weight_spec.json. load_weight_free_ort_inputs reads the checkpoint from HF cache
    and injects the tensors as ORT inputs at each generation step.

    CPU-only. No QAIC hardware required.
    """
    try:
        model_hf = load_hf_model(model_id)
        tokenizer = load_tokenizer(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    api_runner = ApiRunner(
        batch_size=BATCH_SIZE,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=["hello world"],
        prompt_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        full_batch_size=None,
    )

    # HF PyTorch reference tokens (real weights)
    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)

    # Weight-free export uses a meta-device model — no weights in the ONNX.
    # The real weights are read from the HF cache at ORT inference time via
    # load_weight_free_ort_inputs(weight_spec_path, inputs).
    # The exported model must have the same layer count as model_hf, or the
    # HF PT and ORT token streams come from architecturally different models.
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.num_hidden_layers = get_num_layers_from_config(model_hf.config)
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config, weight_free=True)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    onnx_path = exported_onnx_path(
        qeff_model.export(
            tmp_export_dir,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    weight_spec_path = resolve_weight_spec_path(onnx_path)

    # ORT inference with weights injected from HF cache
    ort_tokens = run_weight_free_ort(api_runner, onnx_path, weight_spec_path)

    assert hf_tokens is not None and ort_tokens is not None
    assert hf_tokens.flatten().tolist() == ort_tokens.flatten().tolist(), (
        f"HF PT vs weight-free ORT parity failed for {model_hf.__class__.__name__}: "
        f"HF={hf_tokens.flatten().tolist()}, ORT={ort_tokens.flatten().tolist()}"
    )
