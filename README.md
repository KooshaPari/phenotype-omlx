# phenotype-omlx

**MLX-native, multi-backend, multi-platform OMLX research stack.**

phenotype-omlx is a fork of OMLX (`/Applications/oMLX.app`) that adds
performance cores, concurrent research agents, multi-backend support
(MLX / Metal / vLLM / TensorRT / SGLang / llama.cpp), and multi-platform
clients (macOS / Windows / Linux) on top of the upstream OMLX app.

## What's in this repo

| Tier | Path | Purpose |
| --- | --- | --- |
| **MLX framework** | `/Applications/oMLX.app/.../framework-mlx-base/lib/python3.11/site-packages` | Upstream OMLX Python 3.11 + TurboQuant+ injected into `mlx.nn.layers.turbo_kv_cache` |
| **CLI proxy** | `cli/bin/omlx-cli` | Pass-through to the upstream CLI with `PYTHONPATH` pre-set so the CLI sees the same TurboQuant+ as the GUI |
| **CLI research launcher** | `cli/bin/omlx-research` | Unified entry point: `repl`, `cli`, `gui`, `web`, `doctor`, `status`, `inference`, `spec-decode`, `latentmas`, `tidar`, `bench`, `fleet` |
| **Web admin** | `python/omlx_research/web.py` | Local HTTP server (`omlx-research web`) serving the research panel + REST endpoints |
| **GUI admin extensions** | `gui/admin-extensions/` | Drop-in extensions that mount inside the oMLX.app web admin (templates + static + API) |
| **Python surface** | `python/omlx_research/` | `backends/` (vLLM, TensorRT, SGLang, llama.cpp, MLX, Metal), `engines/` (spec-decode, tree-attn, par-batch, hybrid-dispatch), `agents/` (LatentMAS, TiDAR, SSD, JetSpec schedulers), `cli/` (subcommand CLI) |
| **Rust perf-core** | `perf-core/` | 5-crate workspace: `spec-decode`, `concurrent-exec`, `turbo-quant`, `tree-attention`, `fleet-proto`. The CPU/Metal hot path is in Rust; Metal kernels are loaded at runtime. |
| **Python FFI** | `python/ffi/src/lib.rs` | pyo3 bindings so Python can call into the Rust perf-core (compiled as `_phenotype_omlx_core`) |
| **Reference research repos** | `../turboquant_plus`, `../JetSpec`, `../ssd`, `../LatentMAS`, `../TiDAR` | Original third-party code, surfaced read-only via `phenotype-omlx-env.sh` |
| **Windows client** | `windows-client/` | PowerShell launcher + planned Tauri GUI |
| **Linux client** | `linux-client/` | bash launcher + planned Tauri GUI |
| **macOS desktop** | upstream `/Applications/oMLX.app` | The upstream OMLX app, with our admin-extensions mounted via `OMLX_ADMIN_EXTRA` |

## Why Rust + Python?

Three reasons:

1. **Latency on the hot path.** Speculative decoding, tree attention, and
   TurboQuant pack/unpack all run per-token. Rust is ~2-5× faster than
   Python on these CPU-bound inner loops, and Metal shader dispatch is
   significantly cleaner from Rust than from Python.
2. **Cross-platform FFI.** The same `perf-core` workspace compiles to a
   native `.so` / `.dylib` / `.dll` that pyo3 wraps for Python. The Rust
   surface is also the natural place for the `fleet-proto` JSON-RPC peer
   protocol used by the Windows / Linux clients.
3. **Optional Metal kernels.** MLX handles its own Metal dispatch, so the
   Rust side stays CPU-only for now. If we later want direct Metal calls
   (e.g., for the speculative tree attention kernel), the same workspace
   already has `metal` placeholder files in `spec-decode/src/metal.rs`.

## Quick start

```bash
# 1) Verify the stack (idempotent — compiles perf-core on first run)
./scripts/phenotype-omlx-ready

# 2) Interactive REPL with the full stack
./cli/bin/omlx-research

# 3) Doctor + status
./cli/bin/omlx-research doctor
./cli/bin/omlx-research status

# 4) Inference via the policy dispatcher
./cli/bin/omlx-research inference --prompt "Hello" --policy auto

# 5) Speculative decoding demo
./cli/bin/omlx-research spec-decode --mode ssd --gamma 5

# 6) LatentMAS fan-out demo
./cli/bin/omlx-research latentmas --prompt "Plan a 3-day trip" --n-agents 4

# 7) Web admin (research panel + REST)
./cli/bin/omlx-research web --port 8080

# 8) Launch the oMLX.app GUI with admin-extensions mounted
./cli/bin/omlx-research gui
```

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full diagram and tier
breakdown. Top-level decisions live in [`docs/adr/`](docs/adr/).

## Repository map (merge target for the upstream OMLX fork)

| OMLX tier | phenotype-omlx path | What changed |
| --- | --- | --- |
| MLX framework | `perf-core/turbo-quant/` + `/Applications/oMLX.app/.../turbo_kv_cache.py` | TurboQuant+ inject |
| CLI | `cli/bin/omlx-cli` + `cli/bin/omlx-research` | Pass-through proxy + unified launcher |
| GUI / web | `gui/admin-extensions/` | Research panel, REST API, static assets |
| Server | `python/omlx_research/web.py` | Local web admin |
| Engines | `python/omlx_research/engines/` + `perf-core/spec-decode/` | New engines with Rust perf-core |
| Backends | `python/omlx_research/backends/` | vLLM / TensorRT / SGLang / llama.cpp / MLX / Metal adapters |
| Agents | `python/omlx_research/agents/` | LatentMAS, TiDAR, SSD, JetSpec concurrent schedulers |
| Fleet | `perf-core/fleet-proto/` | JSON-RPC peer protocol + in-memory registry |

## Multi-platform

| Platform | Status | Entry point |
| --- | --- | --- |
| macOS (Apple Silicon) | ✅ Production | `/Applications/oMLX.app` + `cli/bin/omlx-research` |
| Linux | 🟡 Stub | `linux-client/omlx-research` (PyTorch + CUDA / ROCm fallback) |
| Windows | 🟡 Stub | `windows-client/omlx-research.ps1` (Tauri GUI planned) |

## Multi-engine

| Engine | Tier | Use case |
| --- | --- | --- |
| **MLX** (primary) | Apple Silicon | Lowest latency on M-series; required for TurboQuant+ |
| **Metal** | Apple Silicon | Direct Metal kernel dispatch (advanced) |
| **vLLM** | Linux / cloud | High-throughput serving on NVIDIA / ROCm |
| **TensorRT-LLM** | Linux / cloud | Max-throughput inference on NVIDIA |
| **SGLang** (planned) | Linux / cloud | RadixAttention + structured generation |
| **llama.cpp** | Any | CPU + GGUF quantization, broadest model support |

The `HybridDispatch` engine picks a backend per-request based on a policy
(`auto`, `mlx`, `metal`, `vllm`, `tensorrt`, `sglang`, `llamacpp`, `lowest-latency`,
`highest-throughput`).

## Performance cores (Rust)

```
perf-core/
├── Cargo.toml                      # workspace
├── spec-decode/                    # speculative decoding engine
│   ├── src/lib.rs
│   ├── src/backend.rs              # backend trait
│   ├── src/engine.rs               # draft + verify loop
│   ├── src/verify.rs               # target verification + acceptance
│   └── src/metal.rs                # Metal kernel placeholders
├── concurrent-exec/                # concurrent agent scheduler
│   ├── src/lib.rs
│   ├── src/plan.rs                 # execution plan / DAG
│   ├── src/latentmas.rs            # LatentMAS adapter
│   ├── src/tidar.rs                # TiDAR adapter
│   ├── src/ssd.rs                  # SSD adapter
│   └── src/jetspec.rs              # JetSpec adapter
├── turbo-quant/                    # CPU SIMD TurboQuant pack/unpack
│   └── src/lib.rs
├── tree-attention/                 # tree causal mask + verification
│   └── src/lib.rs
└── fleet-proto/                    # JSON-RPC peer protocol
    └── src/lib.rs
```

Test status: **5 / 5 Rust crates compile. 5 / 5 unit tests pass.**

## Repo merge history (hwLedger → phenotype-omlx)

The `hwLedger` research project (chore-overhaul-2026-06-30 worktree) is
**fully merged** into this repo as documentation only. See:

- `docs/adr/2026-06-18/ADR-035A-hwledger-reclassification.md`
- `docs/boundary/phenotype-omlx.md`
- `docs/intent/phenotype-omlx.md`

The hwLedger Rust core itself was not merged — it served a different
purpose (hardware capability ledger) and is now archived at
`docs/research/architectures/hwledger-archive/`.

## License

See upstream OMLX license for the framework files we proxy. The
phenotype-omlx additions (perf-core, omlx_research, admin extensions,
research panel) are MIT-licensed.
