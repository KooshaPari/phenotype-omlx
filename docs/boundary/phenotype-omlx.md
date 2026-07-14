---
repo: "phenotype-omlx"
aliases: ["omlx-research"]
role: canonical
status: active
archived: false
last_verified: 2026-07-08
---

# Boundary — phenotype-omlx

## Scope

- **Owns** the OMLX research overlay: TurboQuant+ adapter, multi-backend stack
  (MLX, Metal, vLLM, TensorRT, SGLang, llama.cpp), concurrent agent runners
  (LatentMAS, TiDAR, JetSpec, SSD), perf-core Rust workspace, CLI/GUI/Web bridge.
- **Depends on** OMLX `.app` bundle for the base MLX inference server.
- **Depends on** `KooshaPari/turboquant_plus` for the Python reference impl.
- **Depends on** `hao-ai-lab/JetSpec`, `irfannaqieb/TiDAR`, `Gen-Verse/LatentMAS`,
  `tanishqkumar/ssd` for research reference implementations.

## Out of Scope

- Does NOT modify the OMLX `.app` bundle (single `turbo_kv_cache.py` exception).
- Does NOT replace the OMLX inference server.
- Does NOT host models or weights.

## Interfaces

| Interface | Description |
|-----------|-------------|
| `cli/bin/omlx-cli` | Proxy to system OMLX CLI with PYTHONPATH injection |
| `scripts/phenotype-omlx-env.sh` | Source-able env activator |
| `gui/admin-extensions/` | Flask/FastAPI blueprint for web admin panel |
| `perf-core/` | Rust workspace: spec-decode, concurrent-exec, turbo-quant, tree-attn, fleet-proto |
| `python/ffi/` | PyO3 bindings to perf-core |
| `windows-client/install.ps1` | Windows client installer |
| `linux-client/install.sh` | Linux client installer |
