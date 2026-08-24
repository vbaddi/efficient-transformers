# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from argparse import Namespace
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from QEfficient.blocking.blocking_configurator import build_transformer_blocking_config_for_transform
from QEfficient.blocking.gqa_attention import GQA_IMPLEMENTATIONS, gqa_head_parallel_attention
from QEfficient.blocking.moe import MOE_IMPLEMENTATIONS
from QEfficient.utils.layer_benchmarks.__main__ import _parser
from QEfficient.utils.layer_benchmarks.artifacts import parse_device_ids
from QEfficient.utils.layer_benchmarks.contracts import RuntimeOptions
from QEfficient.utils.layer_benchmarks.gqa import GQABenchmark
from QEfficient.utils.layer_benchmarks.gqa_cache import pack_gqa_cache, unpack_gqa_cache
from QEfficient.utils.layer_benchmarks.mla import MLABenchmark
from QEfficient.utils.layer_benchmarks.moe import MoEBenchmark
from QEfficient.utils.layer_benchmarks.registry import built_in_benchmarks
from QEfficient.utils.layer_benchmarks.runner import run_benchmark


def _gqa_args(**overrides):
    values = {
        "phase": "decode",
        "attn_blocking_mode": "kv",
        "implementation": None,
        "batch_size": 1,
        "seq_len": None,
        "ctx_len": 16,
        "max_position_embeddings": 16,
        "start_position": 7,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_kv_blocks": 4,
        "query_block_size": 2,
        "rope_theta": 10_000.0,
        "rms_norm_epsilon": 1e-6,
        "qk_norm": True,
        "skip_kv": True,
        "kv_cache_dtype": "float16",
        "dtype": "float16",
        "continuous_batching": False,
        "num_layers": 1,
        "num_cores_per_device": None,
        "repeat_kv_heads": 1,
        "q_head_block_chunk": 1,
        "chunk_kv_size": 4,
        "chunk_kv_n": 2,
        "kv_block_unroll": 2,
    }
    values.update(overrides)
    return Namespace(**values)


def _full_attention(query, key, value, position_ids):
    groups = query.shape[1] // key.shape[1]
    key = (
        key[:, :, None]
        .expand(-1, -1, groups, -1, -1)
        .reshape(query.shape[0], query.shape[1], key.shape[2], key.shape[3])
    )
    value = value[:, :, None].expand(-1, -1, groups, -1, -1).reshape_as(key)
    scores = torch.matmul(query, key.transpose(-1, -2)) / query.shape[-1] ** 0.5
    positions = torch.arange(key.shape[-2]).view(1, 1, 1, -1)
    scores = scores.masked_fill(positions > position_ids[:, None, :, None], float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), value)


@pytest.mark.parametrize("query_len", [1, 4], ids=["decode", "prefill"])
def test_gqa_shared_kernel_matches_full_attention(query_len):
    torch.manual_seed(11)
    query = torch.randn(2, 4, query_len, 8)
    key = torch.randn(2, 2, 16, 8)
    value = torch.randn(2, 2, 16, 8)
    position_ids = torch.arange(5, 5 + query_len).expand(2, -1)

    expected = _full_attention(query, key, value, position_ids)
    actual = gqa_head_parallel_attention(
        query,
        key,
        value,
        position_ids,
        scaling=8**-0.5,
        num_kv_blocks=4,
        query_block_size=2,
    )
    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)


def test_registry_and_device_group_validation():
    assert list(built_in_benchmarks()) == ["gqa", "mla", "moe"]
    assert parse_device_ids("[0, 2]") == (0, 2)
    with pytest.raises(ValueError, match="duplicate"):
        parse_device_ids("[0,0]")


def test_gqa_phase_rejects_wrong_implementation():
    with pytest.raises(ValueError, match="requires implementation"):
        GQABenchmark().resolved_config(_gqa_args(implementation="prefill_attn_parallel"))


@pytest.mark.parametrize("implementation", GQA_IMPLEMENTATIONS)
def test_all_gqa_implementations_match_reference(implementation):
    phase = "prefill" if implementation.startswith("prefill") else "decode"
    config = GQABenchmark().resolved_config(
        _gqa_args(
            phase=phase,
            implementation=implementation,
            seq_len=4 if phase == "prefill" else 1,
            batch_size=2,
            dtype="float32",
            num_cores_per_device=2,
        )
    )
    case = GQABenchmark().build_case(config, seed=1234)
    with torch.no_grad():
        expected = case.reference(*case.inputs.values())
        actual = case.module(*case.inputs.values())
    assert all(torch.allclose(lhs, rhs, atol=1e-5, rtol=1e-5) for lhs, rhs in zip(expected, actual))


def _moe_args(implementation):
    return Namespace(
        implementation=implementation,
        batch_size=2,
        seq_len=1 if implementation.startswith("decode") else 4,
        hidden_size=8,
        intermediate_size=16,
        num_nsp=2,
        local_experts=2,
        total_experts=None,
        num_experts_per_token=2,
        avg_valid_rows=1,
        pattern="random",
        packed_chunk_size=2,
        ffn_blocking_mode="default",
        ffn_token_block_size=None,
        ffn_weight_block_size=None,
        experts_per_soc=None,
        tree_reduce=False,
        dtype="float32",
        dynamo=False,
    )


@pytest.mark.parametrize("implementation", MOE_IMPLEMENTATIONS)
def test_all_moe_implementations_match_reference(implementation):
    benchmark = MoEBenchmark()
    case = benchmark.build_case(benchmark.resolved_config(_moe_args(implementation)), seed=1234)
    with torch.no_grad():
        expected = case.reference(*case.inputs.values())
        actual = case.module(*case.inputs.values())
    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "benchmark_name,implementation",
    [
        *(("gqa", implementation) for implementation in GQA_IMPLEMENTATIONS),
        *(("moe", implementation) for implementation in MOE_IMPLEMENTATIONS),
    ],
)
def test_all_layer_implementations_export_legacy_onnx(tmp_path, benchmark_name, implementation):
    if benchmark_name == "gqa":
        benchmark = GQABenchmark()
        phase = "prefill" if implementation.startswith("prefill") else "decode"
        args = _gqa_args(
            phase=phase,
            implementation=implementation,
            seq_len=4 if phase == "prefill" else 1,
            dtype="float32",
            num_cores_per_device=2,
        )
    else:
        benchmark = MoEBenchmark()
        args = _moe_args(implementation)
    case = benchmark.build_case(benchmark.resolved_config(args), seed=1234)
    torch.onnx.export(
        case.module,
        tuple(case.inputs.values()),
        str(tmp_path / f"{benchmark_name}-{implementation}.onnx"),
        input_names=case.input_names,
        output_names=list(case.output_names),
        dynamic_axes=dict(case.dynamic_axes),
        opset_version=17,
        dynamo=False,
    )


def test_standalone_gqa_cli_aliases_resolve_to_qeff_config():
    args = _parser().parse_args(
        [
            "run",
            "gqa",
            "--attn-blocking-mode",
            "kv",
            "--attn-num-kv-blocks",
            "4",
            "--start-pos-id",
            "9214",
            "--seq-len",
            "1",
            "--ctx-len",
            "10240",
            "--impls",
            "decode_attn_headpar_batch_split",
            "--continuous-batching",
            "--rms-norm-eps",
            "1e-6",
            "--repeat-kv-heads",
            "1",
            "--q-head-block-chunk",
            "1",
            "--compile-num-cores",
            "16",
            "--hw-warmup",
            "5",
            "--hw-iters",
            "50",
            "--run-perf",
        ]
    )

    config = GQABenchmark().resolved_config(args)

    assert config["phase"] == "decode"
    assert config["num_kv_blocks"] == 4
    assert config["start_position"] == 9214
    assert config["implementation"] == "decode_attn_headpar_batch_split"
    assert config["continuous_batching"] is True
    assert args.num_cores == 16
    assert args.warmup == 5
    assert args.iterations == 50
    assert args.run_perf is True


def test_continuous_batching_adapter_preserves_logical_order():
    benchmark = GQABenchmark()
    config = benchmark.resolved_config(_gqa_args(batch_size=4, continuous_batching=True))
    case = benchmark.build_case(config, seed=1234)
    inputs = tuple(case.inputs.values())

    with torch.no_grad():
        expected = case.reference(*inputs)
        actual = case.module(*inputs)

    assert list(case.inputs)[-1] == "batch_index"
    assert all(
        torch.allclose(reference, candidate, atol=1e-4, rtol=1e-4) for reference, candidate in zip(expected, actual)
    )


def test_qwen3_gqa_v1_matches_existing_blocked_path():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from QEfficient.blocking.attention_blocking import AttentionBlockingConfig, BlockingMode
    from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

    torch.manual_seed(17)
    config = Qwen3Config(
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=32,
        intermediate_size=64,
        vocab_size=100,
        max_position_embeddings=16,
        head_dim=8,
    )
    base = Qwen3ForCausalLM(config).eval()
    existing = deepcopy(base)
    candidate = deepcopy(base)
    existing, _ = KVCacheTransform.apply(existing)
    candidate, _ = KVCacheTransform.apply(candidate)
    existing, _ = BlockingAttentionTransform.apply(
        existing, AttentionBlockingConfig(mode=BlockingMode.KV, num_kv_blocks=4)
    )
    candidate, _ = BlockingAttentionTransform.apply(
        candidate,
        AttentionBlockingConfig(mode=BlockingMode.KV, num_kv_blocks=4, implementation="gqa_v1", query_block_size=2),
    )
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "position_ids": torch.arange(4).reshape(1, -1),
        "past_key_values": ((torch.zeros(1, 2, 16, 8), torch.zeros(1, 2, 16, 8)),),
    }
    with torch.no_grad():
        expected = existing(**inputs).logits
        actual = candidate(**inputs).logits
    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)


def test_qwen3_gqa_v1_prefill_to_decode_cache_handoff():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from QEfficient.blocking.attention_blocking import AttentionBlockingConfig, BlockingMode
    from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

    torch.manual_seed(19)
    config = Qwen3Config(
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=32,
        intermediate_size=64,
        vocab_size=100,
        max_position_embeddings=16,
        head_dim=8,
        use_cache=True,
    )
    base = Qwen3ForCausalLM(config).eval()
    existing, _ = KVCacheTransform.apply(deepcopy(base))
    candidate, _ = KVCacheTransform.apply(deepcopy(base))
    existing, _ = BlockingAttentionTransform.apply(
        existing, AttentionBlockingConfig(mode=BlockingMode.KV, num_kv_blocks=4)
    )
    candidate, _ = BlockingAttentionTransform.apply(
        candidate,
        AttentionBlockingConfig(mode=BlockingMode.KV, num_kv_blocks=4, implementation="gqa_v1", query_block_size=2),
    )

    def empty_cache():
        return ((torch.zeros(1, 2, 16, 8), torch.zeros(1, 2, 16, 8)),)

    prefill = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "position_ids": torch.arange(4).reshape(1, -1),
        "use_cache": True,
    }
    with torch.no_grad():
        expected_prefill = existing(**prefill, past_key_values=empty_cache())
        actual_prefill = candidate(**prefill, past_key_values=empty_cache())

    assert torch.allclose(expected_prefill.logits, actual_prefill.logits, atol=1e-5, rtol=1e-5)
    for expected_layer, actual_layer in zip(expected_prefill.past_key_values, actual_prefill.past_key_values):
        assert all(torch.equal(expected, actual) for expected, actual in zip(expected_layer, actual_layer))

    decode = {
        "input_ids": torch.tensor([[5]]),
        "position_ids": torch.tensor([[4]]),
        "use_cache": True,
    }
    with torch.no_grad():
        expected_decode = existing(**decode, past_key_values=expected_prefill.past_key_values)
        actual_decode = candidate(**decode, past_key_values=actual_prefill.past_key_values)

    assert torch.allclose(expected_decode.logits, actual_decode.logits, atol=1e-5, rtol=1e-5)
    for expected_layer, actual_layer in zip(expected_decode.past_key_values, actual_decode.past_key_values):
        assert all(torch.equal(expected, actual) for expected, actual in zip(expected_layer, actual_layer))


def test_gqa_v1_can_be_selected_from_qeff_compile_config():
    blocking = build_transformer_blocking_config_for_transform(
        SimpleNamespace(),
        ctx_len=16,
        seq_len=1,
        qaic_config={
            "enable_blocking": True,
            "blocking_mode": "kv",
            "num_kv_blocks": 4,
            "attention_implementation": "gqa_v1",
            "query_block_size": 2,
        },
    )

    assert blocking.implementation == "gqa_v1"
    assert blocking.num_kv_blocks == 4
    assert blocking.query_block_size == 2


def test_prepare_writes_versioned_result_and_reproducible_assets(tmp_path):
    benchmark = GQABenchmark()
    config = benchmark.resolved_config(_gqa_args())
    options = RuntimeOptions(
        stage="prepare",
        artifact_root=tmp_path,
        artifact_dir=None,
        seed=1234,
        atol=1e-4,
        rtol=1e-4,
        hw_version="ai100",
        num_cores=16,
        device_ids=(0,),
        warmup=1,
        iterations=1,
        perf_iterations=2,
        profile_start_iteration=1,
        profile_samples=1,
        perf_stats_level=70,
        enable_mxfp6=True,
        allow_mxint8_mdp_io=False,
    )

    run_dir, result = run_benchmark(benchmark, config, options)

    assert result["schema_version"] == 1
    assert result["status"] == "passed"
    assert all(item["passed"] for item in result["stages"]["parity"])
    for relative_path in (
        "gqa.onnx",
        "inputs.npz",
        "inputs.json",
        "specializations.json",
        "custom_io.yaml",
        "compile.sh",
        "io/aic_batch_io.json",
        "perf_dump/compile_perf.sh",
        "perf_dump/run_perf.sh",
        "perf_dump/decode_perf.sh",
        "result.json",
    ):
        assert (run_dir / relative_path).is_file()


def test_packed_gqa_cache_round_trip():
    cache = torch.randn(2, 2, 16, 8)
    packed = pack_gqa_cache(cache, ways=4, packing=2)

    assert packed.shape == (2, 8, 4, 8)
    assert torch.equal(unpack_gqa_cache(packed, context_length=16, num_heads=2, ways=4, packing=2), cache)


def test_chunk_implementation_uses_packed_retained_state():
    benchmark = GQABenchmark()
    config = benchmark.resolved_config(
        _gqa_args(
            implementation="decode_attn_headpar_chunk_kv",
            batch_size=2,
            num_cores_per_device=2,
        )
    )
    case = benchmark.build_case(config, seed=1234)

    assert case.inputs["past_key.0"].shape == (2, 8, 4, 8)
    assert case.inputs["past_value.0"].shape == (2, 8, 4, 8)


def _mla_args(**overrides):
    values = {
        "implementation": "mla",
        "model_profile": "kimi_k25",
        "batch_size": 1,
        "seq_len": 1,
        "ctx_len": 16,
        "start_position": 7,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "q_lora_rank": 8,
        "qk_rope_head_dim": 4,
        "kv_lora_rank": 8,
        "v_head_dim": 4,
        "qk_nope_head_dim": 4,
        "rope_theta": 50_000.0,
        "max_position_embeddings": 32,
        "mla_absorption": True,
        "mla_online": False,
        "attn_blocking_mode": "none",
        "num_kv_blocks": None,
        "head_block_size": None,
        "par_num_split": None,
        "repeat_kv_heads": 2,
        "dsa_topk": 8,
        "dsa_index_head_dim": 8,
        "dsa_index_n_heads": 4,
        "kv_cache_dtype": "float16",
        "dtype": "float32",
    }
    values.update(overrides)
    return Namespace(**values)


def test_mla_model_profile_supplies_defaults():
    benchmark = MLABenchmark()
    config = benchmark.resolved_config(
        _mla_args(
            model_profile="glm5",
            hidden_size=None,
            num_attention_heads=None,
            q_lora_rank=None,
            qk_rope_head_dim=None,
            kv_lora_rank=None,
            v_head_dim=None,
            qk_nope_head_dim=None,
            rope_theta=None,
            max_position_embeddings=None,
        )
    )

    assert config["hidden_size"] == 6144
    assert config["num_attention_heads"] == 64
    assert config["q_lora_rank"] == 2048
    assert config["v_head_dim"] == 256
    assert config["qk_nope_head_dim"] == 192
    assert config["rope_theta"] == 1_000_000.0
    assert config["max_position_embeddings"] == 202_752


@pytest.mark.parametrize(
    "mode,absorption,online",
    [
        ("none", True, False),
        ("none", True, True),
        ("none", False, False),
        ("kv", True, False),
        ("kv", True, True),
        ("kv", False, False),
        ("h", True, False),
        ("h", True, True),
        ("h", False, False),
    ],
)
def test_mla_readme_variants_match_reference(mode, absorption, online):
    benchmark = MLABenchmark()
    config = benchmark.resolved_config(
        _mla_args(
            attn_blocking_mode=mode,
            mla_absorption=absorption,
            mla_online=online,
            num_kv_blocks=4 if mode == "kv" else None,
            head_block_size=2 if mode == "h" else None,
        )
    )
    case = benchmark.build_case(config, seed=1234)
    with torch.no_grad():
        expected = case.reference(*case.inputs.values())
        actual = case.module(*case.inputs.values())
    assert all(torch.allclose(lhs, rhs, atol=1e-5, rtol=1e-5) for lhs, rhs in zip(expected, actual))


@pytest.mark.parametrize("mode", ["none", "kv", "h"])
def test_mla_readme_variants_export_legacy_onnx(tmp_path, mode):
    benchmark = MLABenchmark()
    config = benchmark.resolved_config(
        _mla_args(
            attn_blocking_mode=mode,
            num_kv_blocks=4 if mode == "kv" else None,
            head_block_size=2 if mode == "h" else None,
        )
    )
    case = benchmark.build_case(config, seed=1234)
    torch.onnx.export(
        case.module,
        tuple(case.inputs.values()),
        str(tmp_path / f"mla-{mode}.onnx"),
        input_names=case.input_names,
        output_names=list(case.output_names),
        dynamic_axes=dict(case.dynamic_axes),
        opset_version=17,
        dynamo=False,
    )


@pytest.mark.parametrize(
    "implementation,mode,seq_len,start_position",
    [
        ("mla", "par", 1, 7),
        ("mla", "prefill_par", 4, 4),
        ("mla", "prefill_par_online", 4, 4),
        ("dsa_par", "par", 1, 7),
        ("dsa_par_blocked", "par", 1, 7),
    ],
)
def test_mla_advanced_variants_match_reference_and_export(tmp_path, implementation, mode, seq_len, start_position):
    benchmark = MLABenchmark()
    config = benchmark.resolved_config(
        _mla_args(
            implementation=implementation,
            attn_blocking_mode=mode,
            seq_len=seq_len,
            start_position=start_position,
            num_kv_blocks=4,
            par_num_split=2,
            dsa_topk=8,
        )
    )
    case = benchmark.build_case(config, seed=1234)
    with torch.no_grad():
        expected = case.reference(*case.inputs.values())
        actual = case.module(*case.inputs.values())
    assert all(torch.allclose(lhs, rhs, atol=1e-5, rtol=1e-5) for lhs, rhs in zip(expected, actual))

    torch.onnx.export(
        case.module,
        tuple(case.inputs.values()),
        str(tmp_path / f"mla-{implementation}-{mode}.onnx"),
        input_names=case.input_names,
        output_names=list(case.output_names),
        dynamic_axes=dict(case.dynamic_axes),
        opset_version=17,
        dynamo=False,
    )


@pytest.mark.parametrize("mode", ["kv", "h"])
def test_mla_retained_cache_handoff_matches_reference(mode):
    benchmark = MLABenchmark()
    config = benchmark.resolved_config(
        _mla_args(
            attn_blocking_mode=mode,
            seq_len=1,
            start_position=0,
            num_kv_blocks=4 if mode == "kv" else None,
            head_block_size=2 if mode == "h" else None,
        )
    )
    case = benchmark.build_case(config, seed=1234)
    module = case.module
    generator = torch.Generator().manual_seed(4321)
    prefill_hidden = torch.randn(1, 4, config["hidden_size"], generator=generator)
    decode_hidden = torch.randn(1, 1, config["hidden_size"], generator=generator)
    prefill_positions = torch.arange(4).unsqueeze(0)
    decode_position = torch.tensor([[4]])
    empty_ckv = torch.zeros_like(case.inputs["compressed_kv.0"])
    empty_rope = torch.zeros_like(case.inputs["k_pe.0"])

    with torch.no_grad():
        reference_prefill = module.reference(prefill_hidden, prefill_positions, empty_ckv, empty_rope)
        reference_decode = module.reference(decode_hidden, decode_position, reference_prefill[1], reference_prefill[2])
        candidate_prefill = module(prefill_hidden, prefill_positions, empty_ckv, empty_rope)
        candidate_decode = module(decode_hidden, decode_position, candidate_prefill[1], candidate_prefill[2])

    assert all(
        torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)
        for expected, actual in zip(reference_decode, candidate_decode)
    )
