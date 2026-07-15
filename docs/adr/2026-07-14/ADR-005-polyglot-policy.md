# ADR-005 — Polyglot language policy (null the "Rust only" marker)

**Status:** Accepted
**Date:** 2026-07-14
**Supersedes:** the implicit "Rust only" implication in ADR-002 §1
**Related:** ADR-002 (tiered runtime), ADR-003 (multi-engine dispatch), ADR-004 (concurrent agents)

## Context

ADR-002 established Rust as the perf-core tier by reasoning through the language
landscape available at the time (Mojo pre-1.0, Zig lacking MLX/Metal bindings,
Go/C++ integrated via pyo3 where they exist). The decision was correct for
the *default tier allocation* but was informally interpreted downstream as
"Rust only" — which was never the intent.

Concretely, the `phenotype-omlx/AGENTS.md` and the worktree copy at
`.worktrees/ffi-correctness/AGENTS.md` both contain the line:

> Treat `perf-core/` as the Rust workspace and `python/` as its Python
> integration surface.

…which has been read by downstream agents as an outright ban on other
languages in `perf-core/`. This is wrong on two counts:

1. **Rust is the default, not the only choice.** Mojo (when stable), Zig (when
   its MLX/Metal story lands), CUDA kernels for NVIDIA hot paths, C++ for
   vendor BLAS/MLX-C interop, Swift for Apple-specific Metal compute, and even
   hand-tuned assembly for inner loops are all legitimate choices when they
   win on a measured path.
2. **Bindings are first-class.** pyo3, UniFFI, ctypes/cffi, JNI, Swift↔ObjC
   bridges — whatever is the canonical binding for the runtime. The choice of
   binding should follow the source language, not the other way around.

## Decision

The "Rust only" / "Python integration surface" markers in `AGENTS.md` are
**nulled**. The replacement policy is:

### Default tier allocation (kept from ADR-002 as the *default*, not as a *restriction*)

| Tier                 | Default language | Override condition                                       |
|----------------------|------------------|----------------------------------------------------------|
| Perf-cores           | Rust             | MLX/Metal native → MLX-C or Swift; NVIDIA hot paths → CUDA; SIMD-dense → Rust/C++; ergonomics-on-Apple → Mojo (when ≥ 1.0) |
| Orchestration        | Python           | Zig/C++ if pyo3 binding friction is measured-blocking     |
| CLI / wrappers       | Bash             | Zsh if macOS-specific ergonomics matter                   |
| GUI / web            | TypeScript/Svelte | SwiftUI on macOS where it pulls weight                  |
| Config               | TOML             | JSON if a downstream tool requires it strictly           |

### Polyglot rules

1. **Choose by measured perf, not by default.** If a profiling run shows Rust
   losing to Mojo/Zig/CUDA/Swift on a specific path, switch that path. No
   paperwork required for the switch — just a note in `docs/sessions/<id>/`.
2. **All bindings are first-class.** pyo3, UniFFI, ctypes, cffi, JNI, Swift↔ObjC
   bridges. The binding is selected by the source language's canonical
   idiomatic FFI, not by what the existing tier happens to use.
3. **Multi-engine is the default, not the exception.** MLX/Metal + SGLang +
   vLLM + TensorRT-LLM + llama.cpp running concurrently is wired into
   `python/omlx_research/nanovm/` and `python/omlx_research/hybrid/`. New
   engines (CUDA, ROCm, Mojo-native, Nim/Jai if they get a backend) plug in
   via a new `plugin.toml` manifest, no code change.
4. **Experimental languages get sandboxed crates, not blocked.** Mojo, Zig,
   Nim, Jai, Vale, Carbon → each can get its own `perf-core/<lang>/` crate
   with its own `Cargo.toml` workspace entry (or equivalent build system).
   Nothing is forbidden at the agent level; only the workspace manifest decides
   what ships in CI.
5. **Default allocation still applies in the absence of measurement.** If
   there is no profiling evidence that another language wins, Rust is the
   right default for new perf-core work. The shift from "restriction" to
   "default" is the operative change here.

### What this does NOT change

- ADR-002's tier model stays intact: perf-cores vs orchestration vs CLI vs GUI.
  What changes is *which languages may occupy a tier*.
- The Cargo workspace at `perf-core/Cargo.toml` still defines the canonical
  perf-core build graph for Rust crates. A non-Rust crate (e.g. a future
  `perf-core/mojo-kernels/`) plugs in via its own build system and is
  invoked from `Cargo.toml` only if Rust needs to call it.
- MLX remains the primary inference backend on Apple Silicon. That is a
  hardware-availability decision, not a language decision.

## Consequences

- Agents can now propose Mojo/Zig/CUDA/Swift for a specific perf-core path
  without needing to first amend an ADR. The ADR amendment happens *after* the
  measurement, not before.
- A future `perf-core/mojo-kernels/` sandbox crate is allowed. It would carry
  a `mojo` runtime dependency and a `mojo.toml` (or equivalent) build manifest,
  separate from the Cargo workspace.
- `AGENTS.md` gets a one-paragraph header (§2) that records the polyglot
  policy at the level an agent reads first. ADR-005 stays as the formal record.