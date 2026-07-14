---
repo: "phenotype-omlx"
aliases: ["omlx-research"]
role: canonical
status: active
archived: false
last_verified: 2026-07-08
---

# Intent — phenotype-omlx

## Intent Statement

Extend OMLX with a multi-backend, multi-platform research stack that bridges
MLX/Metal (Apple), CUDA (NVIDIA), and concurrent research agents (LatentMAS,
TiDAR, JetSpec, SSD) through a unified CLI + web GUI + desktop interface.

## Key Principles

1. **Performance-critical paths in Rust** — speculative decode, tree attention,
   concurrent execution, TurboQuant+ CPU fallback. Python for orchestration.
2. **Backend agnosticism** — same `generate()` interface for MLX, Metal, vLLM,
   TensorRT-LLM, SGLang, and llama.cpp. Switch at config time.
3. **Concurrent agents** — LatentMAS fan-out, TiDAR diffusion, JetSpec tree
   draft, SSD n-gram lookup all runnable in parallel via the scheduler.
4. **Cross-platform** — macOS (MLX/Metal native) + Linux (vLLM/SGLang/llama.cpp)
   + Windows (TensorRT-LLM/llama.cpp). Client installers for each.
5. **OMLX-respectful fork** — overlay, not replacement. Single `turbo_kv_cache.py`
   injection into the bundle; everything else lives in PYTHONPATH.
