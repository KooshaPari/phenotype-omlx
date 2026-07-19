# Research: complete-polyglot-vpu-stack (WP01)
**Date**: 2026-07-19 | **Mode**: inventory + compile gate

## Spec Summary
7 functional requirements (FR-1..FR-7) for eval loaders, AArch64 NEON min/max,
Mojo TurboQuant smoke, Nim/Go FFI, Qwen3.5 NIAH, stock harness reuse, vPU dashboard.

## FR → Path Matrix

| FR | Primary paths | Compile / test status | Gaps |
|----|---------------|----------------------|------|
| **FR-1** eval loaders | `perf-core/eval-harness/src/{mmlu,gpqa,terminal_bench,perplexity,dataset,provenance,report,runner,backend,error}.rs`; tests `tests/{evaluation,loaders,scoring}.rs` | **PASS** — `cargo test -p eval-harness` → 90 tests | No live-model backend wired; runners need a `Backend` impl for end-to-end eval |
| **FR-2** AArch64 NEON min/max | `perf-core/turbo-quant/src/minmax.rs` | **PASS** on arm64 — scalar/NEON parity + tail tests; microbench ignored | Portable fallback exists via `scalar_min_max`; no separate x86 SIMD path (spec only asks NEON + fallback) |
| **FR-3** Mojo TurboQuant smoke | `perf-core/turbo-quant-mojo/` (`mojo-src/turbo_quant.mojo`, feature `mojo`) | **PARTIAL** — Rust crate tests pass (placeholder); Mojo build gated on `mojo` feature + SDK in PATH | Need `cargo test -p turbo-quant-mojo --features mojo` + SDK smoke script in CI/local gate |
| **FR-4** Nim/Go FFI | `perf-core/turbo-quant-{nim,go,c,zig}/`, worktree `../phenotype-omlx.worktrees/ffi-validation` | **PARTIAL** — Rust placeholders pass; Zig/C ABI tests exist; Go/Nim have WIP in ffi-validation worktree | End-to-end cgo/nimlink against `turbo-quant-c` not green in canonical tree |
| **FR-5** Qwen3.5 NIAH | `scripts/niah_benchmark.py` | **SCRIPT ONLY** — requires MLX model + GPU; linear-attention caveat in spec | No checked-in regression artifact; path still points at legacy `phenotype-omlx/python` |
| **FR-6** stock harness | `python/omlx_research/cli/`, `cli/bin/omlx-research`, `kernel-registry` governance | **PASS** — Python CLI tests (89+) + Rust governance tests | Must not add ForgeCode-style loop; wire eval-harness outputs into existing promote/gates flow |
| **FR-7** vPU dashboard | `gui/admin-extensions/{api/research_panel.py,static/,templates/}`, `python/omlx_research/web.py` | **UNVERIFIED** — Flask/HTTP panel exists; no automated serve test in perf-core | Need smoke: `omlx-research web` + panel API contract test |

## Workspace compile gates (2026-07-19)
- `cargo test -p eval-harness` — **green**
- `cargo test -p kernel-registry --test governance_fuzz` — **green after float_roundtrip fix**
- `cargo test` (full perf-core) — **red** on `regress-baseline/tests/dispatch_buckets.rs` (environment energy envelope; pre-existing, not FR-scoped)

## WP01 finding
Spec cited broken evaluation tests; eval-harness already compiles and passes. The highest-leverage blocker found was `kernel-registry` governance fuzz: `PromotionRecord` content hash failed `verify_content_hash()` after JSON round-trip due to f64 ULP drift.

## Recommended approach (split Initial Implementation WP)
See `plan.md` for WP02–WP07 decomposition.
