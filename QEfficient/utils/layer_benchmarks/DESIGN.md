# QEfficient Layer Benchmark Design

## Status

This document describes the implemented module-level benchmark framework and the
process for extending it. The current implementation supports the complete set
of named implementations from the supplied standalone GQA and MoE benchmarks.

GQA implementations:

- `decode_attn_headpar`
- `decode_attn_headpar_chunk_kv`
- `decode_attn_headpar_chunk_kv_unroll`
- `decode_attn_headpar_batch_split`
- `decode_attn_headpar_batch_split_unroll`
- `prefill_attn_parallel`
- `prefill_attn_parallel_chunk_kv`
- `prefill_attn_online_prefill`

MLA implementations:

- `mla` with `none`, `kv`, and `h` blocking

MoE implementations:

- `reference`
- `gather`
- `packed_chunk`
- `packed_chunk_post_gather`
- `cumsum_scatter_gather_update`
- `cumsum_scatter_gather_update_with_router`
- `decode_qwen3vl_cumsum_scatter_gather_update_with_router`
- `decode_gather_bmm`
- `decode_gather_bmm_loop`

## Problem

Standalone microbenchmarks are useful for rapid compiler experiments, but each
script historically duplicated input generation, correctness checks, ONNX
export, compiler commands, runtime setup, profiling scripts, and result parsing.
That duplication creates two recurring costs:

1. A new optimization spends engineering time rebuilding benchmark plumbing.
2. Moving a successful kernel into QEff exposes cache, IO, or graph mismatches
   late in the optimization cycle.

The framework makes the optimized component the unit of change and keeps the
execution lifecycle stable across implementations.

## Goals

- Let kernel authors focus on the optimized PyTorch module and dispatch branch.
- Run every candidate through the same parity, export, compile, runtime, and
  profiling lifecycle.
- Produce reproducible artifacts and machine-readable results.
- Preserve the graph structure that the compiler optimization is intended to
  measure.
- Keep benchmark kernels in QEff ownership so successful candidates can be
  enabled in model code without a second implementation.
- Make IO and retained-state layout changes explicit from the first benchmark.
- Support fair A/B measurement against an existing standalone QPC.

## Non-Goals

- Automatically select the fastest implementation for every model or shape.
- Hide meaningful IO changes behind an incompatible adapter.
- Promote a benchmark result to the default model path without model-level
  prefill/decode and cache-handoff validation.
- Maintain a YAML-based benchmark matrix. CI or orchestration may invoke the CLI
  repeatedly when a matrix is needed.
- Enable Dynamo export for MoE before its custom-op path is validated. MoE uses
  legacy ONNX export for now.

## Architecture

```text
CLI / Python caller
        |
        v
Benchmark registry
        |
        v
LayerBenchmark adapter ----------------------+
  - resolves module-specific arguments       |
  - builds correlated inputs                 |
  - declares IO/export metadata              |
  - provides trusted reference               |
        |                                     |
        v                                     |
Owning QEff kernel module                     |
  - implementation registry                  |
  - implementation dispatch                  |
  - optimized tensor program                 |
        |                                     |
        +------------------+------------------+
                           v
Common runner
  parity -> ONNX -> artifacts -> compile -> QAIC run -> profile -> result.json
```

The important ownership boundary is between the adapter and the runner:

- The adapter owns behavior that differs by operation or IO contract.
- The runner owns behavior that must be identical for every benchmark.
- The optimized kernel lives under `QEfficient/blocking`, not inside benchmark
  orchestration code.

## Components

### Contracts and Registry

`contracts.py` defines `LayerBenchmark`, `BenchmarkCase`, and `RuntimeOptions`.
A benchmark adapter resolves command-line values into an immutable configuration
and returns one `BenchmarkCase`. The case includes:

- optimized module and reference callable;
- ordered inputs and output names;
- dynamic axes and specialization values;
- custom IO dtype declarations;
- retained-state tensor names.

`registry.py` contains only explicit built-in benchmark registrations. This
keeps CLI discovery deterministic and avoids import-time plugin behavior.

### Common Runner

`runner.py` owns the lifecycle:

1. Seed and construct one correlated benchmark case.
2. Run CPU reference-to-candidate parity.
3. Export legacy ONNX.
4. Write input, specialization, custom IO, and runner manifests.
5. Emit reproducible compile, run, and decode-profile scripts.
6. Compile a QPC when requested.
7. Run hardware correctness and latency when requested.
8. Write `result.json` even when a later stage fails.

Stages are cumulative:

- `prepare`: parity, export, and artifact generation.
- `compile`: `prepare` plus QPC compilation.
- `run`: `compile` plus QAIC correctness and latency.
- `profile`: generated compile/run/decode scripts and opstats collection.

### Artifact Contract

Each run directory contains the files needed to reproduce or compare the run:

```text
<run>/
  result.json
  inputs.npz
  inputs.json
  <benchmark>.onnx
  specializations.json
  custom_io.yaml
  compile.sh
  io/aic_batch_io.json
  perf_dump/compile_perf.sh
  perf_dump/run_perf.sh
  perf_dump/decode_perf.sh
  qpc/                         # after compile
```

`result.json` records the resolved benchmark configuration, runtime options,
environment versions, artifact paths, compiler command, stage results, parity
errors, latency, and failures.

## Public Usage

Source-tree CLI:

```bash
python -m QEfficient.utils.layer_benchmarks list
python -m QEfficient.utils.layer_benchmarks run gqa [options]
python -m QEfficient.utils.layer_benchmarks run mla [options]
python -m QEfficient.utils.layer_benchmarks run moe [options]
python -m QEfficient.utils.layer_benchmarks compare BASELINE_DIR CANDIDATE_DIR [options]
```

The GQA command accepts the established standalone option aliases, including
`--impls`, `--attn-num-kv-blocks`, `--start-pos-id`, `--compile-num-cores`,
`--hw-warmup`, `--hw-iters`, and the `--perf-*` options. Existing commands can
therefore replace the script path with the module entry point.

A shorter installed command such as `qeff-layer-benchmark` is desirable, but it
requires a maintainer-approved `pyproject.toml` console-script change. The module
entry point is the supported interface until that packaging change is accepted.

Python callers may use `built_in_benchmarks()`, construct `RuntimeOptions`, and
call `run_benchmark()` directly. CLI and Python calls use the same contracts.

## Adding an Implementation

### Unchanged IO Contract

This is the normal path. The author should:

1. Add the public implementation name to the owning kernel's implementation
   tuple.
2. Implement the tensor program in the owning kernel module.
3. Add one explicit dispatch branch from the public name to that program.
4. Include the name in the existing parameterized parity/export tests.
5. Run `prepare`, a representative compile, and hardware profiling.

No new parity harness, export function, compiler wrapper, bash generator, JSON
writer, or benchmark file is required. The adapter reuses its existing inputs,
reference, and IO metadata.

### Changed IO Contract

An implementation needs an adapter branch when it changes any of these:

- input/output names or meanings;
- tensor rank or physical shape;
- retained-state ownership or layout;
- dtype/custom IO requirements;
- continuous-batching semantics;
- specialization dimensions.

The adapter branch must convert the common logical fixture to the physical
representation expected by the kernel. Its reference must convert retained
state back to the logical representation before correctness comparison. The
optimized path must not unpack merely to simplify execution, because that would
benchmark a different graph.

Packed-KV GQA is the current example. Its physical state is:

```text
[B, Hkv * N * C, ceil(context / (N * C)), D]
```

where `N` is `chunk_kv_n` and `C` is the core split. Scatter, gather, causal
position reconstruction, and output reduction all operate on this physical
layout. Only the reference path unpacks it to logical `[B, Hkv, context, D]`.

## Correctness Policy

Correctness is checked at several boundaries:

1. Optimized PyTorch output and state against the logical reference.
2. Legacy ONNX export for every registered implementation.
3. Compiler acceptance for representative graph families.
4. QAIC output against a reference with device-owned retained state semantics.
5. Model-level prefill-to-decode cache handoff before promotion.

FP16 hardware comparisons need tolerances appropriate to compiler precision.
The exact maximum and mean errors remain in `result.json`; a relaxed tolerance
must never hide a structural mismatch or an incorrect cache update.

Performance comparisons require identical shapes, compiler flags, device group,
warmup, iterations, seed, and retained-state semantics. The `compare` command
runs both QPCs through one measurement loop and reports samples, medians, and
candidate delta.

## GQA Design

The benchmark adapter contains Qwen3-style projections, optional Q/K RMS norm,
RoPE, retained-state updates, the logical reference, and fixture construction.
The optimized attention programs live in:

- `QEfficient/blocking/gqa_attention.py` for ordinary and batch-folded layouts;
- `QEfficient/blocking/gqa_packed.py` for physically packed chunk-KV layouts;
- `QEfficient/customop/ctx_chunk_scatter.py` for packed retained-state writes.

Continuous batching is supported only by batch-split decode variants. Chunk-KV
variants have a different physical cache contract and reject continuous
batching until a compatible indexed packed-cache design is implemented.

The first model integration is opt-in:

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

The default model path is unchanged. Packed layouts need an explicit model cache
contract and handoff test before they can be selected by this configuration.

## MoE Design

The adapter owns expert/router fixtures and the dense reference. Optimized FFN,
packing, cumsum scatter/gather, routed, decode BMM, and looped decode programs
live in `QEfficient/blocking/moe.py`.

MoE currently rejects `--dynamo`. The immediate goal is fidelity to the supplied
standalone graphs through legacy export. Dynamo support should be added only
after custom operators, export parity, and compiler behavior are validated in
the PyTorch environment documented by `examples/dynamo/causal_lm/README.md`.

## MLA Design

The MLA adapter builds the projection layer and retained compressed-KV/RoPE
state, while dispatching directly to `QEffDeepseekV3Attention` production
methods. It resolves Kimi-K2.5, DeepSeek-V3.2, and GLM-5 profile dimensions and
uses the matching standard or YaRN rotary behavior. Explicit dimension overrides
allow reduced validation without changing the profile semantics.

The README-supported `mla` paths (`none`, `kv`, and `h`) are present. Newer
standalone-only parallel, prefill-parallel, and DSA implementations remain
outside the registry until their kernels and changed IO contracts are owned by
QEff modules. The KV path also requires the blocked-gather symbolic result to
carry the gathered block length, not the full retained-cache length.

## Promotion Into QEff Models

A benchmark implementation is a candidate, not automatically a production
default. Promotion proceeds as follows:

1. Keep the kernel in its QEff owning module from the first benchmark revision.
2. Add an opt-in model configuration without changing the default path.
3. Reuse the same kernel from the model adapter.
4. Validate real model weights and shapes.
5. Validate prefill, decode, continuous batching where applicable, and retained
   cache handoff across calls.
6. Compare standalone benchmark, integrated benchmark, and model performance.
7. Change a default only after correctness and performance evidence is reviewed.

This avoids reimplementing a successful standalone optimization during porting.

## Validation Matrix

Every new implementation must pass:

| Boundary | Required evidence |
| --- | --- |
| CPU behavior | Output and state parity against the reference |
| Export | Legacy ONNX export succeeds |
| Compiler | Representative QPC compiles |
| Hardware | Output parity with recorded numeric error |
| Performance | Warmed multi-sample latency and profiler artifacts |
| Model promotion | Real-model prefill/decode and cache-handoff parity |

Shared runner or artifact changes require all registered implementation tests.
Kernel-only changes require all variants sharing that kernel family. IO changes
require explicit shape/layout tests in addition to numerical parity.

## Current Evidence and Limitations

- All eight GQA and nine MoE names have CPU parity and legacy ONNX export tests.
- MLA `none`, `kv`, and `h` modes have CPU output/state parity and legacy ONNX
  export coverage across absorption variants.
- Packed GQA cache round-trip and retained-state shapes are covered.
- Representative ordinary GQA, packed decode, packed prefill, routed MoE,
  cumsum MoE, and decode BMM graphs compile.
- Small MoE, packed GQA, and MLA KV cases have run on QAIC hardware. The MLA
  FP16 run passed at `0.460 ms` with `1.37e-4` maximum absolute error.
- A controlled non-packed GQA standalone/QEff comparison measured a `0.0015%`
  median difference (`0.500205 ms` versus `0.500213 ms`).
- QID 2 was in `Error` during validation; the supplied four-device production
  GQA command remains pending until all four devices are ready.
- Full-model validation that requires external Hugging Face assets is pending a
  configured local cache and model-level promotion target.

## Planned Work

1. Run the supplied four-device GQA configuration when QID 2 is healthy.
2. Add model-level cache-handoff coverage for the selected GQA candidate.
3. Define the production cache contract before enabling packed GQA in models.
4. Compare every performance candidate with identical standalone and integrated
   artifacts, then retain only graph variants with a demonstrated purpose.
5. Move the standalone MLA parallel, prefill-parallel, and DSA kernels into
   QEff-owned modules and validate their changed IO contracts.
6. Enable and validate the MoE Dynamo path in the designated PyTorch environment.
7. Propose the packaged console-script alias for maintainer review.
8. Add CI jobs for CPU parity/export; keep compiler and QAIC stages in a
   hardware-capable scheduled pipeline.

## Review Checklist

- Is the kernel located in the owning QEff module?
- Is each public implementation name mapped to one intentional graph?
- Does unchanged IO reuse the existing adapter without special casing?
- Are changed IO shapes and retained-state semantics explicit?
- Does parity cover outputs and mutable state?
- Are export, compile, and hardware results recorded in `result.json`?
- Is a performance claim based on identical A/B conditions?
- Is model integration opt-in until cache-handoff validation passes?
- Are known hardware, exporter, and precision limitations documented?
