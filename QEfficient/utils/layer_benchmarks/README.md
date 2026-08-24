# QEfficient Layer Benchmarks

Layer benchmarks keep kernel experimentation close to the QEff model code while
centralizing parity, export, compilation, runtime, profiling, and result capture.
They are configured with Python CLI arguments; no YAML file is required.

See [DESIGN.md](DESIGN.md) for architecture, extension rules, validation gates,
and the model-promotion roadmap.

```bash
python -m QEfficient.utils.layer_benchmarks list

python -m QEfficient.utils.layer_benchmarks run gqa \
  --phase decode \
  --implementation decode_attn_headpar_batch_split \
  --ctx-len 32768 \
  --num-kv-blocks 8 \
  --stage prepare

python -m QEfficient.utils.layer_benchmarks run moe \
  --impls cumsum_scatter_gather_update \
  --seq-len 128 \
  --hidden-size 2048 \
  --intermediate-size 768 \
  --num-nsp 16 \
  --local-experts 8 \
  --packed-chunk-size 32 \
  --stage prepare
```

Stages are cumulative: `prepare` performs CPU parity and ONNX export, `compile`
also creates a QPC, `run` measures the QPC through `QAICInferenceSession`, and
`profile` runs the generated QAIC runner/opstats scripts. Every invocation writes
a versioned `result.json` under `.cache/qeff-layer-benchmarks` by default.

## Standalone GQA Compatibility

The QEff module CLI accepts the established GQA microbenchmark option names.
Replace the standalone script path with the module entry point and the `run gqa`
subcommand:

```bash
python -W ignore -m QEfficient.utils.layer_benchmarks run gqa \
  --attn-blocking-mode kv \
  --attn-num-kv-blocks 4 \
  --hidden-size 4096 \
  --num-attention-heads 64 \
  --num-key-value-heads 4 \
  --head-dim 128 \
  --rope-theta 5000000 \
  --rms-norm-eps 1e-6 \
  --max-position-embeddings 262144 \
  --batch-size 64 \
  --seq-len 1 \
  --ctx-len 10240 \
  --start-pos-id 9214 \
  --num-layers 1 \
  --kv-cache-dtype mxint8 \
  --num-devices 4 \
  --device-group '[0,1,2,3]' \
  --compile-num-cores 16 \
  --hw-version ai100 \
  --hw-warmup 5 \
  --hw-iters 50 \
  --artifact-dir gqa_kv_ctx10k_batch_index_4dev \
  --impls decode_attn_headpar_batch_split \
  --continuous-batching \
  --qk-norm \
  --skip-kv \
  --dump-io \
  --run-compile \
  --run-perf \
  --perf-num-iters 30 \
  --perf-profile-start-iter 20 \
  --perf-num-samples 2 \
  --perf-stats-level 70 \
  --dtype float16 \
  --seed 1234 \
  --num-cores-per-device 16 \
  --repeat-kv-heads 1 \
  --enable-mxfp6 \
  --allow-mxint8-mdp-io \
  --q-head-block-chunk 1
```

`--run-compile`, `--run-hw`, and `--run-perf` select the cumulative `compile`,
`run`, and `profile` stages respectively. V1 supports one layer and no KV-head
replication; unsupported values fail validation rather than being ignored.

The GQA benchmark exposes every graph from the standalone implementation:

- Decode: `decode_attn_headpar`, `decode_attn_headpar_chunk_kv`,
  `decode_attn_headpar_chunk_kv_unroll`, `decode_attn_headpar_batch_split`, and
  `decode_attn_headpar_batch_split_unroll`.
- Prefill: `prefill_attn_parallel`, `prefill_attn_parallel_chunk_kv`, and
  `prefill_attn_online_prefill`.

Use `--chunk-kv-size`, `--chunk-kv-n`, and `--kv-block-unroll` for the matching
variants. Continuous batching is restricted to the two batch-split decode
graphs.

## MoE Benchmarks

The MoE benchmark accepts the standalone names `reference`, `gather`,
`packed_chunk`, `packed_chunk_post_gather`,
`cumsum_scatter_gather_update`,
`cumsum_scatter_gather_update_with_router`,
`decode_qwen3vl_cumsum_scatter_gather_update_with_router`,
`decode_gather_bmm`, and `decode_gather_bmm_loop`. It deliberately uses legacy
ONNX export; `--dynamo` is rejected until the Dynamo path is promoted.

Small hardware smoke test:

```bash
newgrp qaic
pyenv activate qeff
python -m QEfficient.utils.layer_benchmarks run moe \
  --stage run \
  --impls cumsum_scatter_gather_update \
  --batch-size 1 --seq-len 4 \
  --hidden-size 8 --intermediate-size 16 \
  --num-nsp 2 --local-experts 2 \
  --num-experts-per-token 2 --avg-valid-rows 2 \
  --packed-chunk-size 2 --dtype float16 \
  --device-group '[0]' --compile-num-cores 16
```

## MLA Benchmarks

The MLA adapter reuses QEff production DeepSeek attention methods and supports
the README-documented `mla` implementation with `none`, `kv`, and `h` blocking,
absorption and online-absorption toggles, repeated KV heads, and the `kimi_k25`,
`deepseek_v32`, and `glm5` model profiles. Explicit dimension flags override the
selected profile, which is useful for compiler and hardware smoke tests.

```bash
python -W ignore -m QEfficient.utils.layer_benchmarks run mla \
  --stage run --impls mla \
  --batch-size 1 --seq-len 1 --ctx-len 32 --start-pos-id 7 \
  --hidden-size 32 --num-attention-heads 8 \
  --q-lora-rank 8 --qk-rope-head-dim 4 \
  --kv-lora-rank 8 --v-head-dim 4 --qk-nope-head-dim 4 \
  --attn-blocking-mode kv --attn-num-kv-blocks 4 \
  --repeat-kv-heads 1 --mla-absorption --no-mla-online \
  --device-group "[0]" --compile-num-cores 16
```

Use `--no-enable-mxfp6` when validating strict FP16 hardware parity. MXFP6 runs
should use an explicitly reviewed tolerance and retain the measured error in
`result.json`. The standalone-only `par`, `prefill_par`, `prefill_par_online`,
`dsa_par`, and `dsa_par_blocked` kernels are not registered yet; they must first
move into QEff-owned modules with their IO and cache contracts.

## A/B Comparison

Compile the standalone and integrated runs with identical dimensions, compiler
flags, device mapping, and input seed, then compare their artifact directories
with one runtime loop:

```bash
python -m QEfficient.utils.layer_benchmarks compare \
  /tmp/standalone-gqa /tmp/qeff-gqa \
  --device-group '[0]' --warmup 5 --iterations 1000 --samples 3
```

The command reports each sample, median latency, and candidate delta. The input
manifest is used to exclude retained-state buffers, so both QPCs begin with the
same device-owned cache semantics.

## Adding A Benchmark

Implement `LayerBenchmark` in a focused module and add one explicit entry to
`built_in_benchmarks()`. The benchmark owns its optimized module, reference
calculation, correlated input construction, and export metadata. The common
runner owns the lifecycle and artifacts.

Production candidates must live with their owning QEff subsystem and be imported
by the benchmark. Do not copy a benchmark kernel into a model wrapper. Promoting
a candidate requires CPU and ONNX parity, QAIC parity and profiling, and a model
prefill-to-decode cache-handoff test before changing any default implementation.

For another implementation with unchanged inputs and outputs, add its public name
to the owning kernel registry and one dispatch branch in that kernel. The existing
benchmark case and common runner automatically reuse input generation, reference
parity, ONNX export, artifacts, compile, hardware execution, profiling, and JSON
reporting. Add a focused `build_case` branch only when the implementation changes
an IO shape or meaning, such as GQA packed-KV retained state.

The GQA V1 candidate can be enabled in a QEff model compile configuration without
changing the default attention path:

```python
qaic_config = {
    "enable_blocking": True,
    "blocking_mode": "kv",
    "num_kv_blocks": 4,
    "skip_kv": True,
    "attention_implementation": "gqa_v1",
    "query_block_size": 256,
}
```

The Dynamo exporter requires the separate PyTorch 2.13 environment described in
`examples/dynamo/causal_lm/README.md`; the GQA V1 benchmark uses legacy export.
