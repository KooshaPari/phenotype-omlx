# phenotype-omlx Architecture

## Tier diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CLIENTS (macOS / Windows / Linux)                                        │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐  │
│  │  macOS             │  │  Windows           │  │  Linux             │  │
│  │  oMLX.app (Electron)│  │  Tauri GUI (planned)│  │  Tauri GUI (planned)│  │
│  │  + admin-extensions │  │  omlx-research.ps1 │  │  omlx-research     │  │
│  └─────────┬──────────┘  └─────────┬──────────┘  └─────────┬──────────┘  │
│            │                       │                       │              │
│            ▼                       ▼                       ▼              │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  cli/bin/omlx-research  (unified launcher)                        │    │
│  │  sources scripts/phenotype-omlx-env.sh                            │    │
│  │  • repl / cli / gui / web / python / pip / doctor / status        │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────┬────────────────────────────────────────────┘
                              │  PYTHONPATH
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PYTHON SURFACE  (python/omlx_research/)                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  backends/   │  │  engines/    │  │  agents/     │  │  cli/        │  │
│  │  vllm        │  │  spec-decode │  │  LatentMAS   │  │  status      │  │
│  │  tensorrt    │  │  tree-attn   │  │  TiDAR       │  │  inference   │  │
│  │  sglang      │  │  par-batch   │  │  SSD         │  │  spec-decode │  │
│  │  llamacpp    │  │  hybrid-     │  │  JetSpec     │  │  latentmas   │  │
│  │  mlx (★)     │  │    dispatch  │  │  scheduler   │  │  tidar       │  │
│  │  metal       │  │              │  │              │  │  bench       │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │                 │          │
│         │                 │                 │                 ▼          │
│         │                 │                 │       ┌──────────────────┐ │
│         │                 │                 │       │  web.py          │ │
│         │                 │                 │       │  HTTP admin      │ │
│         │                 │                 │       └──────────────────┘ │
└─────────┼─────────────────┼─────────────────┼────────────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  RUST PERF-CORE  (perf-core/)                                             │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐         │
│  │  spec-decode     │ │  concurrent-exec │ │  turbo-quant     │         │
│  │  • engine.rs     │ │  • plan.rs       │ │  • pack/unpack   │         │
│  │  • verify.rs     │ │  • latentmas.rs  │ │  • Lloyd-Max     │         │
│  │  • metal.rs      │ │  • tidar.rs      │ │  • Beta centroids│         │
│  │  • backend.rs    │ │  • ssd.rs        │ │                  │         │
│  │                  │ │  • jetspec.rs    │ │                  │         │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘         │
│  ┌──────────────────┐ ┌──────────────────┐                                │
│  │  tree-attention  │ │  fleet-proto     │                                │
│  │  • causal mask   │ │  • JSON-RPC      │                                │
│  │  • verify        │ │  • peer registry │                                │
│  └──────────────────┘ └──────────────────┘                                │
│           ▲                                                               │
│           │ pyo3 FFI                                                       │
│           │                                                               │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  python/ffi/src/lib.rs  →  _phenotype_omlx_core  (.so/.dylib/.dll)│    │
│  └──────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  BACKENDS  (compiled engines)                                             │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌─────────┐ │
│  │  MLX ★     │ │  Metal     │ │  vLLM      │ │  TensorRT  │ │  llama  │ │
│  │  Apple Sili│ │  Apple Sili│ │  Linux+    │ │  Linux+    │ │  .cpp   │ │
│  │  Primary   │ │  Direct    │ │  NVIDIA    │ │  NVIDIA    │ │  Any    │ │
│  │  TurboQ+   │ │  kernels   │ │  ROCm(opt) │ │  SGLang(opt)│ │  GGUF   │ │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘ └─────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  HARDWARE                                                                  │
│  ┌──────────────────────────┐  ┌──────────────────────────────────────┐  │
│  │  Apple Silicon (M1/M2/M3)│  │  NVIDIA / ROCm / CPU                 │  │
│  │  Metal + AMX + Neural Eng│  │  CUDA cores / ROCm / AVX-512         │  │
│  └──────────────────────────┘  └──────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

★ = primary / recommended path

## Tier responsibilities

### 1. Clients

Three client surfaces, all wired to the same `omlx_research` Python surface:

| Client | Status | Notes |
| --- | --- | --- |
| **macOS** (`oMLX.app`) | ✅ Production | The upstream OMLX Electron app. We inject our admin-extensions via `OMLX_ADMIN_EXTRA` and the TurboQuant+ module via the bundled MLX framework path. No app-bundle modification required. |
| **Windows** (`omlx-research.ps1`) | 🟡 Stub | PowerShell launcher that calls `python -m omlx_research.cli`. Tauri GUI planned. |
| **Linux** (`omlx-research`) | 🟡 Stub | bash launcher. Tauri GUI + ROCm/CUDA backends planned. |

### 2. Unified launcher

`cli/bin/omlx-research` is a single bash entry point that:

1. Sources `scripts/phenotype-omlx-env.sh` (sets `PYTHONPATH`, `PATH`, activates the venv, exposes the `use-latentmas` / `use-tidar` aliases).
2. Injects the OMLX framework's bundled MLX **first** on `PYTHONPATH` so TurboQuant+ is the version Python picks up.
3. Dispatches by subcommand to one of: `repl`, `cli` (proxy to upstream `omlx-cli`), `gui` (`open -a oMLX.app --args --admin-ext ...`), `web` (local HTTP admin), `python`, `pip`, `doctor`, `status`, or any `omlx_research.cli` subcommand.

`cli/bin/omlx-cli` is a thin bash proxy that re-execs `/Applications/oMLX.app/Contents/MacOS/omlx-cli` with the same env. Both files are committed and `chmod +x`-ed.

### 3. Python surface (`python/omlx_research/`)

- `backends/` — uniform `BackendBase` interface; six implementations (MLX, Metal, vLLM, TensorRT, SGLang, llama.cpp). Each advertises its `BackendCapabilities` so `HybridDispatch` can pick one.
- `engines/` — speculative, tree-attention, parallel-batch, hybrid-dispatch. These compose backends and add policy logic.
- `agents/` — adapters for LatentMAS, TiDAR, SSD, JetSpec. Each exposes `async step(prompt, state)`. The `ConcurrentScheduler` fans out, chains, or falls back between them.
- `cli/` — argparse subcommand CLI (see `python/omlx_research/cli/__init__.py::main`).
- `web.py` — small `http.server`-based local admin that serves the GUI extension's templates and exposes the same REST endpoints as JSON.

### 4. Rust perf-core (`perf-core/`)

Five-crate Cargo workspace, all `cargo check --workspace` clean, all `cargo test --workspace` green:

| Crate | Purpose | Hot path |
| --- | --- | --- |
| `spec-decode` | Draft + verify loop (SSD / draft-model / Medusa) | Engine, verify, Metal placeholders |
| `concurrent-exec` | Execution-plan DAG + agent adapters | LatentMAS, TiDAR, SSD, JetSpec |
| `turbo-quant` | CPU SIMD pack/unpack + Lloyd-Max centroids | Bit-packing kernel |
| `tree-attention` | Tree causal mask + token verification | Mask builder, verify |
| `fleet-proto` | JSON-RPC peer protocol + registry | Envelope, dispatch |

The Python FFI (`python/ffi/src/lib.rs`) uses pyo3 to expose the Rust surface as the `_phenotype_omlx_core` module. The FFI crate is optional — the Python surface falls back to pure-Python implementations if the `.so` isn't built.

### 5. Backends (compiled engines)

| Engine | Platform | Notes |
| --- | --- | --- |
| **MLX** | Apple Silicon | Primary path. TurboQuant+ inject is in `mlx.nn.layers.turbo_kv_cache`. |
| **Metal** | Apple Silicon | Direct Metal kernel dispatch. Used by `turbo_mlx.ssd` and `turbo_mlx.jetspec`. |
| **vLLM** | Linux + NVIDIA / ROCm | High-throughput serving. |
| **TensorRT-LLM** | Linux + NVIDIA | Max-throughput inference. |
| **SGLang** | Linux + NVIDIA | RadixAttention + structured generation. *Planned — adapter stub in place.* |
| **llama.cpp** | Any | GGUF + CPU/Metal/CUDA. Broadest model coverage. |

The `HybridDispatch` policy decides per-request which backend to use:

- `auto` — pick the first available in the priority list (MLX > Metal > vLLM > TensorRT > SGLang > llama.cpp)
- `mlx` / `metal` / `vllm` / `tensorrt` / `sglang` / `llamacpp` — force a specific backend
- `lowest-latency` — pick the lowest-p50 backend
- `highest-throughput` — pick the highest tokens/sec

### 6. Hardware

- **Apple Silicon (M1 / M2 / M3 / M4)** — primary. MLX + Metal + AMX + Neural Engine.
- **NVIDIA / ROCm / CPU** — Linux / cloud. PyTorch + TensorRT / vLLM / llama.cpp.

## Data flow: a single inference call

```
user → omlx-research inference --prompt "Hello" --policy auto
  ↓
python -m omlx_research.cli inference
  ↓
engines/hybrid_dispatch.py::HybridDispatch.generate(policy=AUTO)
  ↓
  Probe each backend in priority order (MLX, Metal, vLLM, TensorRT, SGLang, llama.cpp)
  → pick first one that .is_available() and has the requested model
  ↓
backends/mlx_backend.py::MlxBackend.generate(req)
  → rust perf-core (optional) for tokenization + sampling
  → mlx_lm.generate (or mlx_vlm for vision)
  → stream tokens back to caller
  ↓
Rust perf-core: speculative decoding (if enabled)
  → draft γ tokens with the draft model (or self-speculative / Medusa head)
  → verify with the target model
  → commit accepted prefix, resample rejected tail
  → repeat until EOS or max_tokens
  ↓
streamed response → CLI stdout / web admin / GUI extension
```

## Data flow: a multi-agent concurrent call

```
user → omlx-research latentmas --prompt "Plan a trip" --n-agents 4
  ↓
python -m omlx_research.cli latentmas
  ↓
agents/scheduler.py::ConcurrentScheduler
  ↓
  Build DAG: prompt → [agent_0, agent_1, agent_2, agent_3] (fan-out) → reduce
  ↓
  asyncio.gather(*[agent.step(prompt, state) for agent in agents])
  ↓
  Each agent wraps a third-party model:
    - LatentMasRunner → LatentMAS methods (latent_mas, latent_cot, latent_tot)
    - TidarRunner    → TiDAR hybrid AR+diffusion
    - SsdRunner      → SSD self-speculative decoding
    - JetSpecRunner  → JetSpec draft-head tree
  ↓
  Reduce (concatenate / vote / rerank) → final answer
  ↓
streamed response → CLI / web / GUI
```

## Concurrency model

Three layers of concurrency, each appropriate for its tier:

| Layer | Model | Why |
| --- | --- | --- |
| **HTTP admin** | ThreadingMixIn + daemon threads | Per-request isolation; `omlx-research web` is a low-volume admin surface |
| **Python agents** | `asyncio.gather` + `asyncio.Queue` | I/O-bound fan-out; minimal overhead |
| **Rust perf-core** | `Arc<Mutex<…>>` + work-stealing | CPU-bound; threads pinned to performance cores where possible |

The `ConcurrentScheduler` exposes `Strategy::Sequential`, `Strategy::FanOut`,
`Strategy::Reduce`, `Strategy::DAG` for the four common multi-agent shapes.

## Why these languages?

| Language | Role | Why |
| --- | --- | --- |
| **Python** | Glue / surface | The OMLX framework, the upstream CLI, the GUI, and the venv are all Python. |
| **Rust** | Hot path | CPU-bound inner loops (spec-decode, pack/unpack) need native speed. pyo3 FFI lets us keep the Python surface idiomatic. |
| **Mojo** | (planned) | When Mojo's Python interop stabilizes, port the pack/unpack kernels for a free 2-3× on Apple Silicon. |
| **Zig** | (planned) | When we need direct Metal/MSL calls without going through MLX, Zig is the leanest option. |
| **Go** | (planned) | For the fleet-proto peer protocol when we want a standalone server binary. |
| **C++** | (existing) | The upstream OMLX framework + MLX core. We don't touch it. |

## What changes when an upstream OMLX release lands

`scripts/phenotype-omlx-env.sh` is read on every `omlx-research` invocation.
If `/Applications/oMLX.app` is replaced by a new version, the script
automatically picks up the new framework path. The TurboQuant+ module lives
in `~/.omlx/turboquant-plus/mlx/nn/layers/turbo_kv_cache.py` (a persistent
copy) **and** in the OMLX framework's site-packages. The PYTHONPATH order
in `phenotype-omlx-env.sh` prefers the framework, so a fresh OMLX install
inherits our inject automatically.

## What we modify in the OMLX app bundle

Two files. Both are reproducible from this repo:

1. `/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages/mlx/nn/layers/turbo_kv_cache.py` — copy of `~/.omlx/turboquant-plus/mlx/nn/layers/turbo_kv_cache.py`. Re-copy on every OMLX update via `./scripts/phenotype-omlx-ready` (or a one-liner: `cp ~/.omlx/turboquant-plus/mlx/nn/layers/turbo_kv_cache.py "/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages/mlx/nn/layers/"`).
2. **Nothing else.** The `.app` binary, the Electron bundle, the JS assets, the FastAPI server — all untouched.

## Future work

- Mojo port of `turbo-quant` (planned for Q3 2026 — see `docs/adr/`).
- Tauri-based desktop shell (replaces Electron when the upstream OMLX app adopts it).
- GPU fleet-proto with libp2p for the Windows / Linux clients.
- Direct Metal shader dispatch from `perf-core/spec-decode/src/metal.rs` (currently placeholder).
