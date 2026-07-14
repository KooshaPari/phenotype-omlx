# ADR-001: phenotype-omlx Foundation — Multi-Backend, Multi-Platform OMLX

## Status
Accepted

## Context
The OMLX project (`TheTom/oMLX`) is a macOS MLX inference app with GUI, CLI, and server
mode. Our fork extends it with:

1. **Perf-core Rust workspace** — speculative decoding, concurrent agent execution,
   turbo-quant CPU codepath, tree attention, fleet protocol.
2. **Multi-backend research stack** — vLLM, TensorRT-LLM, SGLang, llama.cpp, MLX, Metal.
3. **Cross-platform clients** — Windows (WinUI 3.0 / Python), Linux (Qt / Python).
4. **Concurrent agent execution** — LatentMAS, TiDAR, JetSpec, SSD adapted from their
   respective research repos for MLX/Metal + CUDA.
5. **hwLedger federation** — fleet capacity modeling, hw ledger tracking, ADRs.

## Decision
Create a self-contained monorepo `phenotype-omlx` that:

1. **Replaces the OMLX research launcher** (`omlx-research`) with a fully featured
   CLI that bridges MLX framework extensions, admin GUI endpoints, and desktop app
   modifications.
2. **Exposes perf-core through PyO3 FFI** so the Python research stack can call Rust
   speculative-decode and concurrent-exec routines natively.
3. **Maintains loose coupling** to the OMLX `.app` bundle — no modifications to the
   bundle itself except the one `turbo_kv_cache.py` injection.
4. **Provides multi-platform client installers** — Windows (PowerShell) and Linux (bash)
   that set up the Python research stack and connect to an OMLX server.

## Consequences

### Positive
- All research adaptations live under a single repo with a unified env script.
- Rust perf-core handles latency-sensitive code paths; Python handles orchestration.
- Cross-platform: macOS (MLX/Metal) + Linux + Windows (CUDA backends).
- ADR traceability via hwLedger merge.

### Negative
- Requires Rust toolchain for full perf-core builds (falls back to Python-only).
- `models` namespace collision: LatentMAS (`models.py`) vs TiDAR (`models/` package)
  requires separate PYTHONPATH invocations.

### Neutral
- OMLX app bundle not modified beyond single `turbo_kv_cache.py` injection.
