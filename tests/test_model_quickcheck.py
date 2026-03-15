"""
Fast CPU regression coverage across the main model families supported by QEfficient.

This file intentionally uses two coverage tiers:

1. Runtime parity:
   - Exact token or tensor parity across HF PyTorch, transformed PyTorch, and ORT
   - Used where the repo already has a stable CPU verification path
2. Export smoke:
   - Used for model families or architectures that are supported by export today,
     but do not yet have a stable CPU runtime parity path in the consolidated test
"""

import logging
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    Qwen2Config,
)

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, OnnxTransformPipeline, SplitTensorsTransform
from QEfficient.base.pytorch_transforms import ModuleMappingTransform, ModuleMutatorTransform
from QEfficient.transformers.models.modeling_auto import (
    QEFFAutoModel,
    QEFFAutoModelForCausalLM,
    QEFFAutoModelForCTC,
    QEFFAutoModelForImageTextToText,
    QEFFAutoModelForSequenceClassification,
    QEFFAutoModelForSpeechSeq2Seq,
)
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils import constants, get_padding_shape_from_config
from QEfficient.utils.hash_utils import HASH_HEXDIGEST_STR_LEN, hash_dict_params, json_serializable, to_hashable
from QEfficient.utils.run_utils import ApiRunner

ort.set_default_logger_severity(3)
logging.getLogger("QEfficient").setLevel(logging.ERROR)
logging.getLogger("QEfficient.base.modeling_qeff").setLevel(logging.ERROR)


CAUSAL_RUNTIME_MODEL_IDS = {
    "gpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
    "codegen": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    "falcon": "hf-internal-testing/tiny-random-FalconForCausalLM",
    "gptj": "hf-internal-testing/tiny-random-GPTJForCausalLM",
    "llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "mistral": "hf-internal-testing/tiny-random-MistralForCausalLM",
    "mixtral": "hf-internal-testing/tiny-random-MixtralForCausalLM",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "phi": "hf-internal-testing/tiny-random-PhiForCausalLM",
    "phi3": "tiny-random/phi-4",
    "qwen2": "yujiepan/qwen2-tiny-random",
    "starcoder2": "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
    "granite": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "olmo2": "hf-internal-testing/tiny-random-Olmo2ForCausalLM",
    "gpt_oss": "tiny-random/gpt-oss-bf16",
}

VLM_TEXT_RUNTIME_MODEL_ID = "tiny-random/gemma-3"
VLM_EXPORT_MODEL_IDS = {
    "gemma3": "tiny-random/gemma-3",
    "qwen2_5_vl": "optimum-intel-internal-testing/tiny-random-qwen2.5-vl",
    "internvl2": "optimum-intel-internal-testing/tiny-random-internvl2",
}
TINY_TEXT_EMBEDDING_MODEL_ID = "hf-internal-testing/tiny-random-BertModel"
TINY_AUDIO_CTC_MODEL_ID = "hf-internal-testing/tiny-random-wav2vec2"
TINY_WHISPER_MODEL_ID = "hf-internal-testing/tiny-random-WhisperForConditionalGeneration"
TINY_SEQ_CLASSIFICATION_MODEL_ID = "ydshieh/tiny-random-BertForSequenceClassification"
TINY_AWQ_MODEL_ID = "optimum-intel-internal-testing/tiny-mixtral-AWQ-4bit"

MODEL_KWARGS = {"attn_implementation": "eager"}
PREFIX_CACHING_MODEL_ID = "hf-internal-testing/tiny-random-GPT2LMHeadModel"
TINY_GPT_OSS_MODEL_ID = CAUSAL_RUNTIME_MODEL_IDS["gpt_oss"]


def _per_test_thread_budget() -> int:
    override = os.environ.get("QEFF_NUM_THREADS")
    if override:
        return max(1, int(override))
    total = os.cpu_count() or 1
    workers = max(1, int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")))
    return max(1, total // workers)


def _configure_torch_threads() -> None:
    threads = _per_test_thread_budget()
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(max(1, min(4, threads)))


def _ort_session(onnx_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    threads = _per_test_thread_budget()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), sess_options=options)


_configure_torch_threads()


@contextmanager
def _suppress_native_output():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            yield
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def _exported_onnx_path(export_result) -> Path:
    if isinstance(export_result, (list, tuple)):
        export_result = export_result[-1]
    onnx_path = Path(export_result)
    assert onnx_path.is_file()
    return onnx_path


def _assert_has_retained_state_outputs(onnx_path: Path) -> None:
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    retained_outputs = [output.name for output in onnx_model.graph.output if output.name.endswith("_RetainedState")]
    assert retained_outputs


def _run_embedding_ort(onnx_path: Path, inputs: Dict[str, torch.Tensor]) -> np.ndarray:
    session = _ort_session(onnx_path)
    input_names = {item.name for item in session.get_inputs()}
    ort_inputs = {name: tensor.detach().numpy() for name, tensor in inputs.items() if name in input_names}
    return session.run(None, ort_inputs)[0]


def _run_whisper_export_smoke(qeff_model: QEFFAutoModelForSpeechSeq2Seq) -> Path:
    onnx_path = _exported_onnx_path(qeff_model.export())
    _assert_has_retained_state_outputs(onnx_path)
    return onnx_path


def _skip_on_model_fetch_error(exc: Exception, model_id: str) -> None:
    pytest.skip(
        f"Skipping {model_id}: model unavailable or unsupported in this environment ({type(exc).__name__}: {exc})"
    )


def _skip_on_export_error(exc: Exception, model_id: str, mode: str) -> None:
    pytest.skip(
        f"Skipping {model_id} {mode}: export/runtime path unavailable in this environment ({type(exc).__name__}: {exc})"
    )


def _export_vlm_with_text_fallback(model_id: str, out_dir: Path) -> Path:
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")
        use_text_only_first = model_type in {"qwen2_5_vl", "internvl_chat"}

        if not use_text_only_first:
            try:
                vlm_model = QEFFAutoModelForImageTextToText.from_pretrained(model_id, trust_remote_code=True)
                return _exported_onnx_path(vlm_model.export(out_dir / "full-vlm"))
            except Exception:
                pass

        try:
            if model_type == "qwen2_5_vl" and getattr(config, "text_config", None) is not None:
                qwen2_cfg_dict = config.text_config.to_dict()
                qwen2_cfg_dict["model_type"] = "qwen2"
                qwen2_allowed_keys = set(Qwen2Config().to_dict().keys())
                qwen2_cfg = Qwen2Config(**{k: v for k, v in qwen2_cfg_dict.items() if k in qwen2_allowed_keys})
                text_model = AutoModelForCausalLM.from_config(qwen2_cfg, trust_remote_code=True, **MODEL_KWARGS)
                text_model = text_model.to(torch.float32)
                text_model.eval()
                qeff_text_model = QEFFAutoModelForCausalLM(text_model)
                return _exported_onnx_path(qeff_text_model.export(out_dir / "text-fallback"))

            text_configs = [getattr(config, "text_config", None), getattr(config, "llm_config", None)]
            for text_config in text_configs:
                if text_config is None:
                    continue
                try:
                    text_model = AutoModelForCausalLM.from_config(
                        text_config,
                        trust_remote_code=True,
                        **MODEL_KWARGS,
                    )
                    text_model = text_model.to(torch.float32)
                    text_model.eval()
                    qeff_text_model = QEFFAutoModelForCausalLM(text_model)
                    return _exported_onnx_path(qeff_text_model.export(out_dir / "text-fallback"))
                except Exception:
                    continue
            raise RuntimeError(f"No text fallback config path available for {model_id}")
        except Exception as text_exc:
            _skip_on_model_fetch_error(text_exc, model_id)
    except Exception as cfg_exc:
        _skip_on_model_fetch_error(cfg_exc, model_id)


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("model_type", "model_id"),
    sorted(CAUSAL_RUNTIME_MODEL_IDS.items()),
    ids=sorted(CAUSAL_RUNTIME_MODEL_IDS),
)
def test_causal_lm_cpu_runtime_parity_with_api_runner(model_type, model_id, tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(tokenizer, "model_input_names"):
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
    prompt = ["hello world"]
    prompt_len = 8
    ctx_len = 12

    model_hf = AutoModelForCausalLM.from_pretrained(
        model_id,
        **MODEL_KWARGS,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model_hf.eval()

    api_runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=prompt,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        full_batch_size=None,
    )

    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path))
    ort_tokens = api_runner.run_kv_model_on_ort(str(onnx_path))

    assert np.array_equal(hf_tokens, kv_tokens.squeeze(0))
    assert np.array_equal(kv_tokens, ort_tokens)


@pytest.mark.llm_model
def test_vlm_text_side_runtime_parity_and_full_export(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(VLM_TEXT_RUNTIME_MODEL_ID, trust_remote_code=True)
    config = AutoConfig.from_pretrained(VLM_TEXT_RUNTIME_MODEL_ID, trust_remote_code=True)
    text_config = config.text_config

    text_model = AutoModelForCausalLM.from_config(text_config, trust_remote_code=True, **MODEL_KWARGS)
    text_model.eval()

    api_runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=text_model.config,
        prompt=["hello world"],
        prompt_len=4,
        ctx_len=8,
        full_batch_size=None,
    )

    hf_tokens = api_runner.run_hf_model_on_pytorch(text_model)
    qeff_text_model = QEFFAutoModelForCausalLM(text_model)
    kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_text_model.model)
    onnx_path = _exported_onnx_path(qeff_text_model.export(tmp_path / "vlm-text"))
    ort_tokens = api_runner.run_kv_model_on_ort(str(onnx_path))

    assert np.array_equal(hf_tokens, kv_tokens.squeeze(0))
    assert np.array_equal(kv_tokens, ort_tokens)

    vlm_model = QEFFAutoModelForImageTextToText.from_pretrained(VLM_TEXT_RUNTIME_MODEL_ID, trust_remote_code=True)
    vlm_onnx_path = _exported_onnx_path(vlm_model.export(tmp_path / "vlm-full"))
    assert vlm_onnx_path.name.endswith(".onnx")


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("vlm_name", "model_id"),
    sorted(VLM_EXPORT_MODEL_IDS.items()),
    ids=sorted(VLM_EXPORT_MODEL_IDS),
)
def test_vlm_export_smoke_additional_models(vlm_name, model_id, tmp_path):
    vlm_onnx_path = _export_vlm_with_text_fallback(model_id, tmp_path / f"vlm-{vlm_name}")
    assert vlm_onnx_path.name.endswith(".onnx")


@pytest.mark.llm_model
def test_text_embedding_cpu_parity_and_export(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(TINY_TEXT_EMBEDDING_MODEL_ID)
    model_hf = AutoModel.from_pretrained(TINY_TEXT_EMBEDDING_MODEL_ID, **MODEL_KWARGS)
    model_hf.eval()

    inputs = tokenizer("hello world", return_tensors="pt")
    hf_outputs = model_hf(**inputs).last_hidden_state.detach().numpy()

    qeff_model = QEFFAutoModel(model_hf)
    qeff_outputs = qeff_model.generate(inputs=inputs, runtime_ai100=False).last_hidden_state.detach().numpy()
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path))
    ort_outputs = _run_embedding_ort(onnx_path, inputs)

    assert np.allclose(hf_outputs, qeff_outputs, atol=1e-5)
    assert np.allclose(hf_outputs, ort_outputs, atol=1e-5)


@pytest.mark.llm_model
def test_audio_embedding_ctc_cpu_parity_and_export(tmp_path):
    processor = AutoTokenizer.from_pretrained(TINY_AUDIO_CTC_MODEL_ID)
    del processor
    replace_transformers_quantizers()
    model_hf = AutoModelForCTC.from_pretrained(TINY_AUDIO_CTC_MODEL_ID, **MODEL_KWARGS, low_cpu_mem_usage=False)
    model_hf.eval()

    from transformers import AutoProcessor

    audio_processor = AutoProcessor.from_pretrained(TINY_AUDIO_CTC_MODEL_ID)
    input_values = audio_processor(
        np.zeros(400, dtype=np.float32), return_tensors="pt", sampling_rate=16000
    ).input_values

    hf_logits = model_hf(input_values=input_values).logits.detach().numpy()
    qeff_model = QEFFAutoModelForCTC(model_hf, pretrained_model_name_or_path=TINY_AUDIO_CTC_MODEL_ID)
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path))
    ort_session = _ort_session(onnx_path)
    ort_logits = ort_session.run(None, {"input_values": input_values.detach().numpy()})[0]

    assert np.allclose(hf_logits, ort_logits, atol=1e-5)


@pytest.mark.llm_model
def test_seq_classification_cpu_parity_and_export(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(TINY_SEQ_CLASSIFICATION_MODEL_ID, trust_remote_code=True)
    model_hf = AutoModelForSequenceClassification.from_pretrained(
        TINY_SEQ_CLASSIFICATION_MODEL_ID,
        trust_remote_code=True,
    )
    model_hf.eval()

    inputs = tokenizer("quick classification check", return_tensors="pt")
    hf_logits = model_hf(**inputs).logits.detach().numpy()

    qeff_model = QEFFAutoModelForSequenceClassification(model_hf)
    qeff_logits = qeff_model.model(**inputs).logits.detach().numpy()
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path))
    ort_session = _ort_session(onnx_path)
    input_names = {item.name for item in ort_session.get_inputs()}
    ort_logits = ort_session.run(
        None,
        {name: tensor.detach().numpy() for name, tensor in inputs.items() if name in input_names},
    )[0]

    assert np.allclose(hf_logits, qeff_logits, atol=1e-5)
    assert np.allclose(hf_logits, ort_logits, atol=1e-5)


@pytest.mark.llm_model
def test_whisper_export_smoke(tmp_path):
    model_hf = AutoModelForSpeechSeq2Seq.from_pretrained(
        TINY_WHISPER_MODEL_ID,
        **MODEL_KWARGS,
        low_cpu_mem_usage=False,
    )
    model_hf.eval()

    qeff_model = QEFFAutoModelForSpeechSeq2Seq(model_hf, pretrained_model_name_or_path=TINY_WHISPER_MODEL_ID)
    onnx_path = _run_whisper_export_smoke(qeff_model)

    assert onnx_path.name.endswith(".onnx")


@pytest.mark.llm_model
def test_causal_subfunction_export_smoke(tmp_path):
    model_id = CAUSAL_RUNTIME_MODEL_IDS["gpt2"]
    model_hf = AutoModelForCausalLM.from_pretrained(model_id, **MODEL_KWARGS, low_cpu_mem_usage=False)
    model_hf.eval()
    qeff_model = QEFFAutoModelForCausalLM(model_hf)

    with_subfunctions_path = _exported_onnx_path(
        qeff_model.export(tmp_path / "with-subfunctions", use_onnx_subfunctions=True, offload_pt_weights=False)
    )
    without_subfunctions_path = _exported_onnx_path(
        qeff_model.export(tmp_path / "without-subfunctions", use_onnx_subfunctions=False)
    )

    with_subfunctions_model = onnx.load(with_subfunctions_path, load_external_data=False)
    without_subfunctions_model = onnx.load(without_subfunctions_path, load_external_data=False)
    with_names = [func.name for func in with_subfunctions_model.functions]
    without_names = [func.name for func in without_subfunctions_model.functions]
    assert any("QEffGPT2Block" in name for name in with_names)
    assert not any("QEffGPT2Block" in name for name in without_names)


@pytest.mark.llm_model
def test_prefix_caching_continuous_batching_export_and_ort_smoke(tmp_path):
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(PREFIX_CACHING_MODEL_ID, continuous_batching=True)
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path / "prefix-caching"))
    onnx_model = onnx.load(onnx_path, load_external_data=False)

    input_names = {inp.name for inp in onnx_model.graph.input}
    output_names = {out.name for out in onnx_model.graph.output}
    op_types = {node.op_type for node in onnx_model.graph.node}
    assert "batch_index" in input_names
    assert "CtxScatterCB" in op_types
    assert "CtxGatherCB" in op_types
    assert any(name.endswith("_RetainedState") for name in output_names)


@pytest.mark.llm_model
def test_awq_export_smoke(tmp_path):
    replace_transformers_quantizers()
    model_hf = AutoModelForCausalLM.from_pretrained(TINY_AWQ_MODEL_ID, low_cpu_mem_usage=False)
    model_hf.eval()

    qeff_model = QEFFAutoModelForCausalLM(model_hf, pretrained_model_name_or_path=TINY_AWQ_MODEL_ID)
    with _suppress_native_output():
        onnx_path = _exported_onnx_path(qeff_model.export(tmp_path))
        onnx_model = onnx.load(onnx_path, load_external_data=False)

    assert any(node.op_type == "MatMulNBits" for node in onnx_model.graph.node)


@pytest.mark.llm_model
def test_base_compiler_invalid_inputs_raise(tmp_path):
    qeff_obj = SimpleNamespace()

    invalid_file = tmp_path / "invalid.onnx"
    invalid_file.write_bytes(chr(0).encode() * 100)
    with pytest.raises(RuntimeError):
        QEFFBaseModel._compile(qeff_obj, invalid_file, tmp_path)

    valid_model = onnx.parser.parse_model("""
    <
        ir_version: 8,
        opset_import: ["": 17]
    >
    test_compiler(float x) => (float y)
    {
        y = Identity(x)
    }
    """)
    valid_file = tmp_path / "valid.onnx"
    onnx.save(valid_model, valid_file)
    with pytest.raises(RuntimeError):
        QEFFBaseModel._compile(
            qeff_obj, valid_file, tmp_path, convert_tofp16=True, compile_only=True, aic_binary_dir=tmp_path
        )


class _QuickcheckLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(32, 64)
        self.b = nn.Linear(64, 32)

    def forward(self, x):
        return self.b(self.a(x))


@pytest.mark.llm_model
def test_pytorch_module_transform_primitives():
    class MappingTransform(ModuleMappingTransform):
        _module_mapping = {nn.Linear: nn.Identity}

    class MutatorTransform(ModuleMutatorTransform):
        _match_class = nn.Linear

        @classmethod
        def mutate(cls, original_module: nn.Module, parent_module: nn.Module):
            del original_module
            del parent_module
            return nn.Identity()

    model = _QuickcheckLinearModel()
    x = torch.rand(1, 32)
    baseline = model(x)
    assert torch.any(baseline != x)

    mapped_model, mapped = MappingTransform.apply(model)
    assert mapped
    mapped_out = mapped_model(x)
    assert torch.all(mapped_out == x)

    model2 = _QuickcheckLinearModel()
    prev_ids = [id(model2.a), id(model2.b)]
    mutated_model, mutated = MutatorTransform.apply(model2)
    assert mutated
    assert [id(mutated_model.a), id(mutated_model.b)] != prev_ids
    mutated_out = mutated_model(x)
    assert torch.all(mutated_out == x)


@pytest.mark.llm_model
def test_onnx_transform_primitives(tmp_path):
    fp16_model = onnx.parser.parse_model("""
    <
        ir_version: 8,
        opset_import: ["" : 17]
    >
    test_fp16clip (float [n, 32] x) => (float [n, 32] y)
    <
        float val1 = {65505.0},
        int64[1] slice_ends = {2147483647},
        float zero = {0.0}
    >
    {
        mask = Greater(x, zero)
        val2 = Constant<value = float {-1e7}>()
        masked = Where(mask, val1, val2)
        slice_starts = Constant<value = int64[1] {0}>()
        y = Slice(masked, slice_starts, slice_ends)
    }
    """)
    transformed_onnx, transformed = OnnxTransformPipeline(transforms=[FP16ClipTransform]).apply(
        fp16_model, model_name="quickcheck-fp16clip"
    )
    assert transformed
    assert onnx.numpy_helper.to_array(transformed_onnx.graph.initializer[0]) == 65504.0
    assert onnx.numpy_helper.to_array(transformed_onnx.graph.node[1].attribute[0].t) == -65504.0

    external_tensors_file = "tensors.raw"
    split_model = onnx.parser.parse_model(f"""
    <
        ir_version: 8,
        opset_import: ["": 17]
    >
    test_split () => ()
    <
        float[1, 32] tensor0 = [ "location": "{external_tensors_file}", "offset": "0", "length": "{32 * 4}" ],
        float[1, 32] tensor1 = [ "location": "{external_tensors_file}", "offset": "{32 * 4}", "length": "{32 * 4}" ],
        float[1, 16] tensor2 = [ "location": "{external_tensors_file}", "offset": "{64 * 4}", "length": "{16 * 4}" ]
    >
    {{
    }}
    """)
    np.random.rand(32 + 32 + 16).astype("float32").tofile(tmp_path / external_tensors_file)
    split_onnx, split_transformed = OnnxTransformPipeline(transforms=[SplitTensorsTransform]).apply(
        split_model,
        model_name="quickcheck-split",
        onnx_base_dir=str(tmp_path),
        file_chunk_size=32 * 4,
        size_threshold=16 * 4,
    )
    assert split_transformed
    tensor0_ext_data = onnx.external_data_helper.ExternalDataInfo(split_onnx.graph.initializer[0])
    tensor1_ext_data = onnx.external_data_helper.ExternalDataInfo(split_onnx.graph.initializer[1])
    assert tensor0_ext_data.location.endswith("_0.onnx.data")
    assert tensor1_ext_data.location.endswith("_1.onnx.data")
    onnx_path = tmp_path / "split.onnx"
    onnx.save(split_onnx, onnx_path)
    assert onnx_path.is_file()


@pytest.mark.llm_model
def test_hash_utils_primitives():
    ordered = {"a": 1, "b": 2}
    reversed_order = {"b": 2, "a": 1}
    assert to_hashable(ordered) == to_hashable(reversed_order)
    assert to_hashable(set(range(4))) == to_hashable(set(range(3, -1, -1)))
    for invalid in [float("nan"), float("inf"), -float("inf")]:
        with pytest.raises(ValueError):
            to_hashable(invalid)
    assert json_serializable({1, 2, 3}) == ["1", "2", "3"]
    digest = hash_dict_params({"key": {1, 2, 3}})
    assert isinstance(digest, str)
    assert len(digest) == HASH_HEXDIGEST_STR_LEN


@pytest.mark.llm_model
def test_local_causal_wrapper_init_pretrained_and_unsupported(tmp_path):
    model = AutoModelForCausalLM.from_pretrained(
        PREFIX_CACHING_MODEL_ID,
        **MODEL_KWARGS,
        low_cpu_mem_usage=False,
        torch_dtype=torch.float32,
    )
    model.eval()
    qeff_model = QEFFAutoModelForCausalLM(model)
    assert qeff_model.model.__class__.__name__.startswith("QEff")

    model_dir = tmp_path / "causal-hf"
    model.save_pretrained(model_dir)
    qeff_pretrained = QEFFAutoModelForCausalLM.from_pretrained(model_dir)
    assert qeff_pretrained.model.__class__.__name__.startswith("QEff")

    with pytest.warns():
        unsupported = AutoModelForCausalLM.from_config(AutoConfig.for_model("opt"))
        _ = QEFFAutoModelForCausalLM(unsupported)


@pytest.mark.llm_model
def test_memory_offload_behavior_mocks():
    class MockParam:
        def __init__(self, is_meta=False):
            self.is_meta = is_meta

    class MockModel:
        def __init__(self):
            self._params = [MockParam(is_meta=False)]

        def parameters(self):
            return self._params

    class MockSingleQPCModel:
        def __init__(self):
            self._is_weights_offloaded = False
            self.model = MockModel()

        def _offload_model_weights(self):
            self._is_weights_offloaded = True
            for param in self.model.parameters():
                param.is_meta = True
            return True

        def export(self, offload_pt_weights=True):
            if offload_pt_weights:
                self._offload_model_weights()
            return "single_qpc_export_path"

    qeff_model = MockSingleQPCModel()
    qeff_model.export(offload_pt_weights=True)
    assert qeff_model._is_weights_offloaded
    assert all(param.is_meta for param in qeff_model.model.parameters())

    qeff_model2 = MockSingleQPCModel()
    qeff_model2.export(offload_pt_weights=False)
    assert not qeff_model2._is_weights_offloaded
    assert not any(param.is_meta for param in qeff_model2.model.parameters())


@pytest.mark.llm_model
def test_local_speech_seq2seq_wrapper_init_and_pretrained(tmp_path):
    config = AutoConfig.for_model(
        "whisper",
        max_source_positions=96,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        d_model=32,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        vocab_size=60000,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    model = AutoModelForSpeechSeq2Seq.from_config(config, **MODEL_KWARGS)
    qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
    assert qeff_model.model.model.__class__.__name__.startswith("QEff")

    model_dir = tmp_path / "speech-hf"
    model.save_pretrained(model_dir)
    qeff_pretrained = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(model_dir)
    assert qeff_pretrained.model.model.__class__.__name__.startswith("QEff")


@pytest.mark.llm_model
def test_gpt_oss_disagg_prefill_only_decode_export_smoke(tmp_path):
    try:
        model_hf = AutoModelForCausalLM.from_pretrained(
            TINY_GPT_OSS_MODEL_ID,
            **MODEL_KWARGS,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        model_hf.eval()
    except Exception as exc:
        _skip_on_model_fetch_error(exc, TINY_GPT_OSS_MODEL_ID)
        return

    qeff_model = QEFFAutoModelForCausalLM(model_hf, pretrained_model_name_or_path=TINY_GPT_OSS_MODEL_ID)
    prefill_seq_len = constants.GPT_OSS_PREFILL_Q_BLOCK_SIZE

    try:
        prefill_onnx = _exported_onnx_path(
            qeff_model.export(
                tmp_path / "gptoss-prefill-only",
                prefill_only=True,
                prefill_seq_len=prefill_seq_len,
                offload_pt_weights=False,
            )
        )
        prefill_hash = qeff_model.export_hash
        decode_onnx = _exported_onnx_path(
            qeff_model.export(
                tmp_path / "gptoss-decode-only",
                prefill_only=False,
                retain_full_kv=True,
                offload_pt_weights=False,
            )
        )
        decode_hash = qeff_model.export_hash
    except Exception as exc:
        _skip_on_export_error(exc, TINY_GPT_OSS_MODEL_ID, "prefill/decode export")
        return

    assert prefill_onnx.is_file()
    assert decode_onnx.is_file()
    assert prefill_hash != decode_hash

    prefill_model = onnx.load(prefill_onnx, load_external_data=False)
    decode_model = onnx.load(decode_onnx, load_external_data=False)
    prefill_inputs = {inp.name for inp in prefill_model.graph.input}
    decode_inputs = {inp.name for inp in decode_model.graph.input}
    assert "input_ids" in prefill_inputs
    assert "position_ids" in prefill_inputs
    assert "input_ids" in decode_inputs
    assert "position_ids" in decode_inputs
    assert any(out.name.endswith("_RetainedState") for out in prefill_model.graph.output)
    assert any(out.name.endswith("_RetainedState") for out in decode_model.graph.output)


@pytest.mark.llm_model
def test_gpt_oss_disagg_prefill_decode_pytorch_smoke():
    try:
        tokenizer = AutoTokenizer.from_pretrained(TINY_GPT_OSS_MODEL_ID, trust_remote_code=True)
        model_hf = AutoModelForCausalLM.from_pretrained(
            TINY_GPT_OSS_MODEL_ID,
            **MODEL_KWARGS,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        model_hf.eval()
    except Exception as exc:
        _skip_on_model_fetch_error(exc, TINY_GPT_OSS_MODEL_ID)
        return

    qeff_model = QEFFAutoModelForCausalLM(model_hf, pretrained_model_name_or_path=TINY_GPT_OSS_MODEL_ID)
    prefill_seq_len = constants.GPT_OSS_PREFILL_Q_BLOCK_SIZE

    prompt = "disagg prefill smoke"
    prompt_inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=prefill_seq_len)
    input_ids = prompt_inputs["input_ids"]
    position_ids = torch.where(
        prompt_inputs["attention_mask"].bool(),
        torch.arange(prefill_seq_len, dtype=torch.int64).unsqueeze(0).expand_as(input_ids),
        torch.full_like(input_ids, -1),
    )

    kv_cache_shape = get_padding_shape_from_config(model_hf.config, 1, prefill_seq_len)
    past_key_values = []
    for _ in range(model_hf.config.num_hidden_layers):
        past_key = torch.zeros(kv_cache_shape, dtype=torch.float32)
        past_value = torch.zeros(kv_cache_shape, dtype=torch.float32)
        past_key_values.append((past_key, past_value))

    qeff_model.prefill(True)
    prefill_outputs = qeff_model.model(
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
    )
    assert prefill_outputs.logits is not None
    assert prefill_outputs.logits.ndim in (2, 3)
    assert prefill_outputs.logits.shape[-1] == model_hf.config.vocab_size

    next_token = prefill_outputs.logits.argmax(-1).reshape(input_ids.shape[0], -1)[:, -1:].to(torch.int64)
    next_pos = (position_ids.max(dim=1, keepdim=True).values + 1).to(torch.int64)

    qeff_model.prefill(False, retain_full_kv=True)
    decode_outputs = qeff_model.model(
        input_ids=next_token,
        position_ids=next_pos,
        past_key_values=prefill_outputs.past_key_values,
    )
    assert decode_outputs.logits is not None
    assert decode_outputs.logits.ndim in (2, 3)
    assert decode_outputs.logits.shape[-1] == model_hf.config.vocab_size
