# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Unit tests for BlockingAttentionTransform in QEfficient.transformers.models.pytorch_transforms.

Verifies that:
  1. BlockingAttentionTransform attaches attn_blocking_config to QEff attention modules
  2. Works correctly for supported model families
  3. Preserves blocking mode/config values
  4. Re-applying overrides the previous config
  5. Handles wrapper config fallback and preserves fast CPU parity

All tests run on CPU only, using tiny in-memory models.
KVCacheTransform must be applied before BlockingAttentionTransform because the
blocking transform matches against QEff attention class types (the *values* of
KVCacheTransform._module_mapping), not the raw HF attention class types (the keys).
"""

from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from QEfficient.blocking import attention_blocking
from QEfficient.blocking.attention_blocking import AttentionBlockingConfig, BlockingMode

VOCAB_SIZE = 500
CTX_LEN = 32


# ---------------------------------------------------------------------------
# Tiny model factories
# ---------------------------------------------------------------------------


def make_tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return LlamaForCausalLM(cfg).eval()


def make_tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
    )
    return Qwen3ForCausalLM(cfg).eval()


def make_tiny_qwen3_vl():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    text_cfg = Qwen3VLTextConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
    )
    vision_cfg = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=32,
        num_heads=2,
        intermediate_size=64,
        out_hidden_size=64,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    cfg = Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg)
    return Qwen3VLForConditionalGeneration(cfg).eval()


def make_tiny_gpt_oss():
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    cfg = GptOssConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=64,
        head_dim=32,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        num_local_experts=4,
        num_experts_per_tok=2,
        sliding_window=CTX_LEN,
        rope_parameters={"rope_type": "default"},
    )
    return GptOssForCausalLM(cfg).eval()


def make_tiny_gemma():
    from transformers import GemmaConfig, GemmaForCausalLM

    cfg = GemmaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
    )
    return GemmaForCausalLM(cfg).eval()


def make_tiny_gemma2():
    from transformers import Gemma2Config, Gemma2ForCausalLM

    cfg = Gemma2Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
        sliding_window=CTX_LEN,
    )
    return Gemma2ForCausalLM(cfg).eval()


def make_tiny_mistral():
    from transformers import MistralConfig, MistralForCausalLM

    cfg = MistralConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return MistralForCausalLM(cfg).eval()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_FACTORIES = [
    (make_tiny_llama, "llama"),
    (make_tiny_qwen3, "qwen3"),
    (make_tiny_qwen3_vl, "qwen3_vl"),
    (make_tiny_gpt_oss, "gpt_oss"),
    (make_tiny_gemma, "gemma"),
    (make_tiny_gemma2, "gemma2"),
    (make_tiny_mistral, "mistral"),
]
_MODEL_IDS = [label for _, label in _MODEL_FACTORIES]


def _qeff_attention_modules(model):
    """Return all modules whose type is in KVCacheTransform supported attention classes."""
    from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform

    supported = {
        qeff_cls for qeff_cls in KVCacheTransform._module_mapping.values() if qeff_cls.__name__.endswith("Attention")
    }
    return [m for m in model.modules() if type(m) in supported]


def _blocking_cfg(**kwargs):
    return AttentionBlockingConfig(**kwargs)


def _make_qeff_inputs(input_ids, config, ctx_len=CTX_LEN):
    batch, seq = input_ids.shape
    position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    n_layers = config.num_hidden_layers
    n_attn = config.num_attention_heads
    n_kv = getattr(config, "num_key_value_heads", n_attn)
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // n_attn)
    past_key_values = tuple(
        (
            torch.zeros(batch, n_kv, ctx_len, head_dim, dtype=torch.float32),
            torch.zeros(batch, n_kv, ctx_len, head_dim, dtype=torch.float32),
        )
        for _ in range(n_layers)
    )
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
    }


# ---------------------------------------------------------------------------
# Tests: basic application per model family
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestBlockingTransformApplied:
    """BlockingAttentionTransform must attach attn_blocking_config to all QEff attention modules."""

    @pytest.mark.parametrize("make_model,label", _MODEL_FACTORIES, ids=_MODEL_IDS)
    def test_config_attached_to_all_attn_modules(self, make_model, label):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

        model = make_model()
        model, _ = KVCacheTransform.apply(model)
        config = _blocking_cfg(mode=BlockingMode.KV, num_kv_blocks=2)
        model, transformed = BlockingAttentionTransform.apply(model, config)

        assert transformed
        attn_mods = _qeff_attention_modules(model)
        assert attn_mods, f"[{label}] no QEff attention modules found after KVCacheTransform"

        for m in attn_mods:
            assert hasattr(m, "attn_blocking_config"), f"[{label}] {type(m).__name__} missing attn_blocking_config"
            assert m.attn_blocking_config is config, (
                f"[{label}] attn_blocking_config must be the same object that was passed in"
            )


# ---------------------------------------------------------------------------
# Tests: blocking modes
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestBlockingModes:
    """BlockingAttentionTransform must preserve the BlockingMode in the attached config."""

    @pytest.mark.parametrize(
        "mode",
        [BlockingMode.NONE, BlockingMode.KV, BlockingMode.Q, BlockingMode.H, BlockingMode.QKV],
    )
    def test_blocking_mode_preserved_on_llama(self, mode):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

        model = make_tiny_llama()
        model, _ = KVCacheTransform.apply(model)
        config = AttentionBlockingConfig(mode=mode, head_block_size=8, num_kv_blocks=2, num_q_blocks=2, ctx_len=128)
        model, transformed = BlockingAttentionTransform.apply(model, config)

        assert transformed
        for m in _qeff_attention_modules(model):
            assert m.attn_blocking_config.mode == mode, f"Expected mode={mode}, got {m.attn_blocking_config.mode}"

    def test_all_config_fields_preserved(self):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

        model = make_tiny_llama()
        model, _ = KVCacheTransform.apply(model)
        config = AttentionBlockingConfig(
            mode=BlockingMode.KV,
            num_kv_blocks=4,
            num_q_blocks=2,
            head_block_size=8,
            skip_kv=False,
            num_batch_blocks=1,
            ctx_len=128,
        )
        model, transformed = BlockingAttentionTransform.apply(model, config)

        assert transformed
        for m in _qeff_attention_modules(model):
            c = m.attn_blocking_config
            assert c.mode == BlockingMode.KV
            assert c.num_kv_blocks == 4
            assert c.num_q_blocks == 2
            assert c.head_block_size == 8
            assert c.skip_kv is False
            assert c.num_batch_blocks == 1


@pytest.mark.transforms
def test_generic_blocked_attention_infers_prefill_only_from_mode(monkeypatch):
    class Cache:
        def __init__(self):
            self.write_only_calls = []

        def write_only(self, key, value, layer_idx, cache_kwargs):
            self.write_only_calls.append((key, value, layer_idx, cache_kwargs))

    cache = Cache()
    query = torch.ones(1, 1, 1, 1)
    key = torch.ones(1, 1, 1, 1)
    value = torch.ones(1, 1, 1, 1)
    strategy_calls = []

    def prefill_strategy(**kwargs):
        strategy_calls.append(kwargs)
        return kwargs["query"], None

    monkeypatch.setitem(attention_blocking._STRATEGIES, BlockingMode.PREFILL_Q, prefill_strategy)

    output, weights = attention_blocking.generic_blocked_attention_interface(
        module=type("Attention", (), {"layer_idx": 0})(),
        query=query,
        key=key,
        value=value,
        past_key_value=cache,
        blocking_config=AttentionBlockingConfig(mode=BlockingMode.PREFILL_Q, num_q_blocks=1),
    )

    assert torch.equal(output, query)
    assert weights is None
    assert len(cache.write_only_calls) == 1
    assert len(strategy_calls) == 1


@pytest.mark.transforms
def test_kv_batch_fold_preserves_optional_gdn_num_head_blocks():
    from QEfficient.blocking.blocking_configurator import build_transformer_blocking_config_for_transform

    config = build_transformer_blocking_config_for_transform(
        model_config=object(),
        ctx_len=1024,
        seq_len=1,
        bs=512,
        num_devices=4,
        qaic_config={
            "blocking_mode": "kv_batch_fold",
            "num_kv_blocks": 16,
            "gdn_num_head_blocks": 8,
        },
    )

    assert config.mode == BlockingMode.KV_BATCH_FOLD
    assert config.batch_fold is True
    assert config.gdn_num_head_blocks == 8


# ---------------------------------------------------------------------------
# Tests: re-application overrides the previous config
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestBlockingTransformIdempotent:
    """Applying BlockingAttentionTransform twice must replace the first config with the second."""

    def test_second_apply(self):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

        model = make_tiny_llama()
        model, _ = KVCacheTransform.apply(model)

        config1 = AttentionBlockingConfig(mode=BlockingMode.KV, num_kv_blocks=2, ctx_len=1024)
        config2 = AttentionBlockingConfig(mode=BlockingMode.Q, num_q_blocks=4)

        model, _ = BlockingAttentionTransform.apply(model, config1)
        model, transformed = BlockingAttentionTransform.apply(model, config2)

        assert transformed, "Reapplication of BlockingAttentionTransform did not succeed"

        for m in _qeff_attention_modules(model):
            assert m.attn_blocking_config is config2, (
                "Second BlockingAttentionTransform.apply must override the first config"
            )
            assert m.attn_blocking_config.mode == BlockingMode.Q


# ---------------------------------------------------------------------------
# Tests: wrapper config fallback + CPU parity
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestBlockingWrapperFallbackAndParity:
    """Regression guards for wrapper config lookup and CPU parity checks."""

    def test_wrapper_without_config_uses_nested_model_config(self):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform

        class _DummyAttention(nn.Module):
            pass

        class _DeepseekContainer(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Cfg", (), {"architectures": ["DeepseekV3ForCausalLM"]})()
                self.attn = _DummyAttention()

        class _WrapperWithoutConfig(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        cfg = AttentionBlockingConfig(mode=BlockingMode.KV, num_kv_blocks=2, ctx_len=128)
        wrapped = _WrapperWithoutConfig(_DeepseekContainer())
        wrapped, transformed = BlockingAttentionTransform.apply(wrapped, cfg)

        assert transformed, "BlockingAttentionTransform must use nested wrapper model config"
        assert wrapped.model.attn.attn_blocking_config is cfg

    @pytest.mark.parametrize(
        "blocking_cfg",
        [
            AttentionBlockingConfig(mode=BlockingMode.NONE),
            AttentionBlockingConfig(mode=BlockingMode.KV, num_kv_blocks=2, ctx_len=128),
        ],
        ids=["mode_none", "mode_kv"],
    )
    def test_cpu_parity_original_vs_transformed_with_same_input(self, blocking_cfg):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

        torch.manual_seed(7)
        base = make_tiny_llama()
        original = deepcopy(base).eval()
        transformed = deepcopy(base).eval()
        transformed, _ = KVCacheTransform.apply(transformed)
        transformed, applied = BlockingAttentionTransform.apply(transformed, blocking_cfg)
        assert applied

        input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
        qeff_inputs = _make_qeff_inputs(input_ids, transformed.config)

        with torch.no_grad():
            original_token = original(input_ids=input_ids).logits[:, -1, :].argmax(-1)
            transformed_token = transformed(**qeff_inputs).logits[:, -1, :].argmax(-1)

        assert torch.equal(original_token, transformed_token), (
            "Original and transformed model outputs diverged for same CPU input"
        )


def _headpar_pipeline_inputs(ctx_len, positions, groups=4, dtype=torch.float32, continuous_batching=False):
    """Tiny GQA decode tensors; no model download or device required."""
    torch.manual_seed(53)
    batch = len(positions)
    cache_batch = batch + 1 if continuous_batching else batch
    query = torch.randn(batch, 2 * groups, 1, 8, dtype=dtype)
    key = torch.randn(cache_batch, 2, ctx_len, 8, dtype=dtype)
    value = torch.randn_like(key)
    position_ids = torch.tensor(positions, dtype=torch.int64).reshape(batch, 1)
    batch_index = torch.tensor([[2], [0]]) if continuous_batching else None
    return query, key, value, position_ids, batch_index


def _headpar_pipeline_forward(
    query, key, value, position_ids, batch_index, blocks, split, skip_kv, sinks=None, ctx_len=None
):
    from QEfficient.blocking.blocked_attention_forwards import blocked_kv_attention_forward_headpar_offline
    from QEfficient.transformers.cache_utils import QEffDynamicCache

    module = nn.Module()
    module.num_key_value_groups = query.shape[1] // key.shape[1]
    cache = QEffDynamicCache.from_legacy_cache(((key, value),))
    output, _ = blocked_kv_attention_forward_headpar_offline(
        module,
        query,
        key,
        value,
        None,
        query.shape[-1] ** -0.5,
        blocks,
        {"position_ids": position_ids, "batch_index": batch_index},
        0,
        cache,
        key.shape[2] if ctx_len is None else ctx_len,
        configured_split=split,
        skip_kv=skip_kv,
        sinks=sinks,
    )
    return output


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("groups", [1, 4])
@pytest.mark.parametrize("skip_kv", [False, True])
@pytest.mark.parametrize(
    "ctx_len,blocks,split,positions,continuous_batching,with_sinks",
    [
        (32, 4, 2, (31,), False, False),
        (17, 4, 3, (16,), False, False),
        (17, 1, 4, (0,), False, True),
        (32, 4, 2, (7,), False, False),
        (32, 4, 2, (8,), False, False),
        (32, 4, 2, (0, 17), True, True),
        (17, 4, 3, (4, 16), False, False),
        (32, 4, 4, (31,), False, True),
    ],
)
def test_headpar_pipeline_dense_parity(
    dtype, groups, skip_kv, ctx_len, blocks, split, positions, continuous_batching, with_sinks
):
    query, key, value, position_ids, batch_index = _headpar_pipeline_inputs(
        ctx_len, positions, groups, dtype, continuous_batching
    )
    saved_key, saved_value = key.clone(), value.clone()
    sinks = torch.linspace(-1, 1, query.shape[1], dtype=dtype) if with_sinks else None
    actual = _headpar_pipeline_forward(query, key, value, position_ids, batch_index, blocks, split, skip_kv, sinks)

    # Independent dense attention oracle, including per-row positions and sinks.
    dense_key = key if batch_index is None else key[batch_index.flatten()]
    dense_value = value if batch_index is None else value[batch_index.flatten()]
    dense_key = dense_key.repeat_interleave(groups, dim=1).float()
    dense_value = dense_value.repeat_interleave(groups, dim=1).float()
    logits = torch.matmul(query.float(), dense_key.transpose(-1, -2)) * (query.shape[-1] ** -0.5)
    mask = torch.arange(ctx_len).reshape(1, 1, 1, -1) > position_ids.reshape(-1, 1, 1, 1)
    logits = logits.masked_fill(mask, float("-inf"))
    if sinks is not None:
        sink_logits = sinks.float().reshape(1, -1, 1, 1).expand(query.shape[0], -1, -1, -1)
        logits = torch.cat((logits, sink_logits), dim=-1)
    probabilities = torch.softmax(logits, dim=-1)[..., :ctx_len]
    expected = torch.matmul(probabilities, dense_value).transpose(1, 2)
    tolerance = {torch.float32: 2e-6, torch.float16: 3e-3, torch.bfloat16: 3e-2}[dtype]
    torch.testing.assert_close(actual.float(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(key, saved_key, atol=0, rtol=0)
    torch.testing.assert_close(value, saved_value, atol=0, rtol=0)


@pytest.mark.parametrize("blocks", [1, 4])
@pytest.mark.parametrize("position", [0, 4, 16])
@pytest.mark.parametrize("skip_kv", [False, True])
def test_headpar_pipeline_read_order(monkeypatch, blocks, position, skip_kv):
    from QEfficient.transformers.cache_utils import QEffDynamicCache

    events = []
    original_matmul = torch.matmul
    original_k = QEffDynamicCache.read_only_blocked_K
    original_v = QEffDynamicCache.read_only_blocked_V

    def read_k(self, start, end, *args):
        assert 0 <= start < end <= 17
        events.append(("K", start, end))
        return original_k(self, start, end, *args)

    def read_v(self, start, end, *args):
        assert 0 <= start < end <= 17
        events.append(("V", start, end))
        return original_v(self, start, end, *args)

    def matmul(*args, **kwargs):
        events.append(("compute",))
        return original_matmul(*args, **kwargs)

    monkeypatch.setattr(QEffDynamicCache, "read_only_blocked_K", read_k)
    monkeypatch.setattr(QEffDynamicCache, "read_only_blocked_V", read_v)
    monkeypatch.setattr(torch, "matmul", matmul)
    inputs = _headpar_pipeline_inputs(17, (position,))
    _headpar_pipeline_forward(*inputs, blocks, 3, skip_kv)

    block_size = -(-17 // blocks)
    count = min(blocks, position // block_size + 1) if skip_kv else blocks
    expected = [("K", 0, min(block_size, 17))]
    for index in range(count):
        start, end = index * block_size, min((index + 1) * block_size, 17)
        expected.extend([("V", start, end), ("compute",)])
        if index + 1 < count:
            expected.append(("K", end, min(end + block_size, 17)))
        expected.append(("compute",))
    assert events == expected


@pytest.mark.parametrize("skip_kv", [False, True])
def test_headpar_pipeline_onnx_parity(tmp_path, monkeypatch, skip_kv):
    import onnx
    import onnxruntime as ort

    from QEfficient.transformers.cache_utils import InvalidIndexProvider

    # Use QEff's existing ORT-safe invalid-index convention. The alternative
    # INT32_MAX sentinel is understood by QAIC but is out of bounds in ORT.
    monkeypatch.setattr(InvalidIndexProvider, "SUBFUNC_ENABLED", True)

    class Decode(nn.Module):
        def forward(self, query, key, value, position_ids):
            return _headpar_pipeline_forward(query, key, value, position_ids, None, 4, 3, skip_kv, ctx_len=17)

    model = Decode().eval()
    inputs = _headpar_pipeline_inputs(17, (16,))[:4]
    path = tmp_path / "headpar_pipeline.onnx"
    torch.onnx.export(
        model,
        inputs,
        str(path),
        input_names=["query", "key_cache", "value_cache", "position_ids"],
        output_names=["output"],
        # Match retained-cache export: keep the context axis symbolic so the
        # legacy custom gather's input-like type does not fold a full-cache
        # zeros_like into a block-sized V mask. This test specializes CL=17.
        dynamic_axes={"key_cache": {2: "ctx_len"}, "value_cache": {2: "ctx_len"}},
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    schedule = []
    for node in graph.graph.node:
        if node.op_type == "CtxGatherBlockedKV":
            schedule.append({"key_cache": "K", "value_cache": "V"}[node.input[0]])
        elif node.op_type == "MatMul":
            schedule.append("MatMul")
    expected_schedule = ["K"]
    for index in range(4):
        expected_schedule.extend(["V", "MatMul"])
        if index < 3:
            expected_schedule.append("K")
        expected_schedule.append("MatMul")
    assert schedule == expected_schedule

    # Verify the raw graph numerically; QAIC, not ORT, owns production scheduling.
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    for position in (0, 4, 5, 16):
        run_inputs = (*inputs[:3], torch.tensor([[position]], dtype=torch.int64))
        feeds = dict(zip(("query", "key_cache", "value_cache", "position_ids"), run_inputs))
        actual = torch.from_numpy(session.run(None, {name: tensor.numpy() for name, tensor in feeds.items()})[0])
        with torch.no_grad():
            expected = model(*run_inputs)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
