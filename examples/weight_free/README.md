# Qwen MXFP6 weight-free reference

This example converts a local dense Qwen BF16 or FP16 safetensors checkpoint,
then exports it through QEfficient's current Dynamo weight-free path:

```bash
python examples/weight_free/prepare_mxfp6_qwen.py /path/to/qwen /tmp/qwen-mxfp6
python - <<'PY'
from QEfficient import QEFFAutoModelForCausalLM

model = QEFFAutoModelForCausalLM.from_pretrained("/tmp/qwen-mxfp6", weight_free=True)
model.export("/tmp/qwen-mxfp6-export", use_onnx_subfunctions=True, offload_pt_weights=False)
PY
```

The public weight-free model construction stays unchanged. For a one-call
compile flow starting from the ordinary local Qwen checkpoint, use the same
API and enable offline preparation at compile time:

```python
model = QEFFAutoModelForCausalLM.from_pretrained(
    args.model_name,
    config=config,
    weight_free=True,
)
model.compile(
    dynamo=True,
    use_onnx_subfunctions=True,
    enable_mxfp6=True,
)
```

`enable_mxfp6` is an offline Qwen checkpoint-preparation option. It is
separate from the existing compiler `mxfp6_matmul` option and is not passed as
a compiler flag. The explicit preparation command above remains useful when
conversion and export need to be rerun independently.

The converter selects only the seven dense decoder projections (`q_proj`,
`k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`) from the
Qwen module naming contract. Other tensors retain the normal checkpoint path.

The proposed checkpoint ABI is `inline_e8m0_fp6_lsb_v1`: every `[O, K]` tensor
becomes flattened UINT8 `[O, 25*ceil(K/32)]`. Each 25-byte block is one E8M0
scale byte followed by 24 bytes containing 32 little-significant-bit-first FP6
codes. Each block stores its E8M0 scale byte, including the all-zero-block
scale exponent `-2`. The `MXDequantize` node is emitted in domain
`com.qualcomm.qeff`, opset 1, with `format="MXFP6_E2M3"`,
`axis=-1`, `block_size=32`, `axis_size=K`, and `output_dtype` set to the
logical weight dtype (`"BFLOAT16"`, `"FLOAT16"`, or `"FLOAT"`).

The export directory contains an actual `checkpoint/` copy, `model.onnx`, and
`weight_spec.json`. The sidecar and `com.qti.aisw.extdata` metadata contain the
same v6 spec. The operator and byte layout are proposed for compiler review;
this task does not attempt compilation or accelerator execution.
