# phenotype-omlx Architecture

## What It Is

Rust perf-core + Python orchestration for LLM inference optimization. Multi-engine dispatch (MLX, vLLM, SGLang, llama.cpp, TensorRT-LLM), speculative decoding, Metal GPU kernels, quantization, and fleet coordination. No language is forbidden — pick by measured perf.

## Directory Layout

```
phenotype-omlx/
├── perf-core/                # Rust workspace (~20 crates)
│   ├── spec-decode/          # Speculative decoding engine
│   ├── metal-runtime/        # Metal GPU kernel runtime (macOS)
│   ├── kernel-registry/      # Kernel lifecycle: discover → select → tune
│   ├── model-kernels/        # Per-model kernel mappings
│   ├── fleet-proto/          # Fleet coordination protocol
│   ├── turbo-quant/          # Quantization engine (Rust core)
│   ├── turbo-quant-{c,go,mojo,nim,zig}/  # Language bindings
│   ├── tree-attention/       # Tree attention + SPIR-V variant
│   ├── concurrent-exec/      # Concurrent execution engine + CUDA
│   └── eval-harness/         # Rust-side eval harness
├── python/                   # Python orchestration layer
│   └── omlx_research/
│       ├── backends/         # Engine adapters (MLX, vLLM, SGLang, llama.cpp, TensorRT)
│       ├── agents/           # Scheduler + runner agents (JetSpec, LatentMAS, TiDAR, SSD)
│       ├── nanovm/           # NanoVM plugin layer (multi-engine dispatch)
│       └── engines/          # Spec-decode, tree-attn, par-batch engines
├── python/ffi/               # Rust→Python FFI bridge (PyO3)
├── apps/bench-cockpit/       # Dashboard for benchmark results
└── cli/bin/                  # Unified launcher (omlx-research)
```

## Key Rust Crates

| Crate | Role |
|---|---|
| `spec-decode` | Proposal → verify → accept/reject speculative decoding pipeline |
| `metal-runtime` | Metal GPU dispatch: AdalN, RoPE3D, MoE, tree attention, pipeline cache |
| `kernel-registry` | Kernel lifecycle: candidate discovery → compatibility → selection → tuning |
| `fleet-proto` | Fleet coordination (JSON-RPC over ZeroMQ/TCP) |
| `turbo-quant` | Quantization engine (minmax, mixed-precision) |
| `tree-attention` | Memory-efficient tree-structured attention |
| `concurrent-exec` | Multi-stream concurrent execution scheduler |

## Key Python Modules

| Module | Role |
|---|---|
| `backends/mlx_backend.py` | Apple MLX engine integration |
| `backends/vllm_backend.py` | vLLM engine integration |
| `backends/sglang_backend.py` | SGLang engine integration |
| `backends/llamacpp_backend.py` | llama.cpp engine integration |
| `backends/tensorrt_backend.py` | TensorRT-LLM engine integration |
| `agents/scheduler.py` | Multi-engine task scheduler |
| `nanovm/plugins/` | Multi-engine dispatch plugin layer |

## Multi-Engine Dispatch

```
Request → Scheduler (select engine by model + hardware + policy)
  → Backend adapter (backends/base.py → concrete)
  → perf-core FFI (PyO3 → Rust: spec-decode, metal-runtime, turbo-quant)
  → Response
```

## ADRs

| ADR | Decision |
|---|---|
| Tiered Runtime | Rust for perf-cores (SIMD/fusion/kernels), Python for orchestration |
| Polyglot Policy | No language forbidden; Mojo, Zig, Nim get sandboxed crates |
| Concurrent Agents | Multiple runner agents execute in parallel across engines |

## Quick Start

```bash
cd perf-core && cargo build --release
cd python && pip install -e .
python -m omlx_research.cli --backend mlx --model qwen3.5-0.8b --prompt "Hello"
```
