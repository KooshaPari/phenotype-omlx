# ADR-003: Multi-engine policy dispatch (MLX primary, vLLM/TensorRT/SGLang/llama.cpp optional)

**Date:** 2026-07-08
**Status:** Accepted
**Related:** ADR-001, ADR-002, `python/omlx_research/engines/hybrid_dispatch.py`

## Context

The research stack needs to talk to multiple inference engines:

- **MLX** — primary, Apple-Silicon-only, lowest latency on M-series
- **Metal** — direct kernel dispatch, Apple-Silicon-only
- **vLLM** — Linux + NVIDIA, high-throughput serving
- **TensorRT-LLM** — Linux + NVIDIA, max-throughput inference
- **SGLang** — Linux + NVIDIA, RadixAttention + structured generation
- **llama.cpp** — any platform, broadest model coverage (GGUF)

The user requested all six. The previous architecture hard-bound everything
to MLX via the upstream OMLX framework, which is fine on Apple Silicon but
excludes Windows / Linux clients.

## Decision

Adopt a `BackendBase` adapter pattern (`python/omlx_research/backends/base.py`)
where every engine exposes a uniform interface:

```python
class BackendBase(Protocol):
    def capabilities(self) -> BackendCapabilities: ...
    def is_available(self) -> bool: ...
    def generate(self, req: GenerateRequest) -> GenerateResponse: ...
```

The `HybridDispatch` engine (`python/omlx_research/engines/hybrid_dispatch.py`)
picks a backend per-request based on a `DispatchPolicy`:

- `auto` — first available in priority order
- `mlx` / `metal` / `vllm` / `tensorrt` / `sglang` / `llamacpp` — force one
- `lowest-latency` — pick the lowest p50 backend
- `highest-throughput` — pick the highest tokens/sec

The `BackendCapabilities` struct advertises what each backend supports
(`primary`, `quantizations`, `vision`, `audio`, `tool_use`, etc.) so
`HybridDispatch` can refuse a request that no backend can satisfy.

## Consequences

**Positive**

- Same Python surface for all engines — `omlx-research inference` works
  identically on macOS, Linux, and Windows.
- New engines can be added by implementing `BackendBase` + registering
  them with `HybridDispatch`.
- The SGLang adapter is a stub today, but the interface contract is
  stable so the implementation can land later.

**Negative**

- Adapter layer is a thin shim; we don't get the engine's full
  capabilities (e.g., vLLM's PagedAttention, SGLang's RadixAttention) —
  we get only what the adapter exposes.
- The policy dispatcher is a single point of failure. A bug in
  `HybridDispatch` blocks all inference.

**Mitigations**

- Each backend can be called directly (`omlx-research inference --policy vllm`)
  bypassing the dispatcher for debugging.
- The dispatcher is stateless — no I/O, just routing.
