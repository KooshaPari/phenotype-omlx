# phenotype-omlx agent contract

- This directory is an independent Git repository. Never rely on a parent Git worktree.
- Treat `perf-core/` as the Rust workspace and `python/` as its Python integration surface.
- Do not expose Python bindings for Rust APIs that do not exist in the checked-in crates.
- Add failing tests before fixing correctness defects, then run focused and workspace checks.
- Keep generated targets, environments, model weights, and extension artifacts out of Git.
- Prefer pure, explicit FFI payloads over process-global mutable state.
- Keep modules at or below 500 lines and target 350 lines; split by coherent responsibility.
- Record session research and decisions under `docs/sessions/<session-id>/`.
