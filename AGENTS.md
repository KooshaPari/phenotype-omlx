# phenotype-omlx agent contract

- This directory is an independent Git repository. Never rely on a parent Git worktree.
- Treat `perf-core/` as the Rust workspace and `python/` as its Python integration surface.

## §2 — Polyglot language policy (max-optimal, supersedes any "Rust only" markers)

**No language is forbidden.** Pick by measured perf, not by default.

| Tier | Default language | Override when it wins |
|---|---|---|
| Perf-cores (SIMD / fusion / kernels) | Rust | Mojo (MLX-native), CUDA (NVIDIA hot paths), C++ (C-API), Zig (FFI ergonomics), Swift (Metal) |
| Orchestration | Python | Zig/C++ if pyo3 friction; Swift if orchestrating MLX/Metal natively |
| CLI / wrappers | Bash | Zsh on macOS, PowerShell on Windows |
| GUI / web | TypeScript / Svelte | SwiftUI on macOS where appropriate |
| Config | TOML | JSON if downstream tool requires it |

- **All bindings are first-class.** pyo3, UniFFI, ctypes, cffi, JNI, Swift↔ObjC bridges, Mojo↔Python — whatever is canonical for the runtime.
- **Multi-engine is the default.** MLX/Metal + SGLang + vLLM + TRT-LLM + llama.cpp running concurrently is wired into the NanoVM plugin layer (`python/omlx_research/nanovm/plugins/`); treat this as the normal path.
- **Experimental languages get sandboxed crates, not blocked.** Mojo, Zig, Nim, Jai, Vale, Carbon → each can get its own `perf-core/<lang>/` crate with its own workspace entry. Nothing is blocked at the agent level.
- **Do not reintroduce "Rust only" / "use only [...]" markers anywhere in this repo** — the prior restriction has been nulled.

See `docs/adr/2026-07-14/ADR-005-polyglot-policy.md` for the full decision rationale.
- Do not expose Python bindings for Rust APIs that do not exist in the checked-in crates.
- Add failing tests before fixing correctness defects, then run focused and workspace checks.
- Keep generated targets, environments, model weights, and extension artifacts out of Git.
- Prefer pure, explicit FFI payloads over process-global mutable state.
- Keep modules at or below 500 lines and target 350 lines; split by coherent responsibility.
- Record session research and decisions under `docs/sessions/<session-id>/`.
