# Plan: complete-polyglot-vpu-stack
**Date**: 2026-07-19 | **WPs**: 7 (split from monolithic WP01)

## Work Packages

### WP01: Compile gate + FR inventory (this session)
**ID**: WP01 | **Dependencies**: none | **State**: in_progress

**Acceptance Criteria:**
- FR→path matrix documented
- Highest-leverage compile/test blocker fixed with regression test
- Verification commands recorded

**Delivered:** `PromotionRecord` JSON float round-trip fix (`serde_json/float_roundtrip` + unit test).

---

### WP02: FR-1 — eval harness backend + live eval wiring
**ID**: WP02 | **Dependencies**: WP01

Wire a minimal `Backend` (stub or metal-runtime adapter) and prove `run_multiple_choice_suite` on fixture datasets.

---

### WP03: FR-2 — NEON min/max hardening
**ID**: WP03 | **Dependencies**: WP01

Document arm64 gate; optional CI cfg for non-macOS skip; keep scalar oracle as SSOT.

---

### WP04: FR-3 + FR-4 — polyglot FFI smoke (Mojo, Nim, Go)
**ID**: WP04 | **Dependencies**: WP01

Merge/consolidate `ffi-validation` worktree; green `turbo-quant-c` ABI tests + Nim/Go e2e against C ABI.

---

### WP05: FR-5 — Qwen3.5 NIAH + quality eval
**ID**: WP05 | **Dependencies**: WP02

Fix `niah_benchmark.py` paths for absorbed crate layout; run locally; record linear-attention limitations explicitly.

---

### WP06: FR-6 — stock harness integration
**ID**: WP06 | **Dependencies**: WP02, WP05

Export eval reports into `omlx-research promote/gates` without custom agent loops.

---

### WP07: FR-7 — vPU dashboard serve smoke
**ID**: WP07 | **Dependencies**: WP06

Automated smoke for `omlx-research web` + `research_panel` API.

## Execution Waves
- **Wave 0** (done): WP01 inventory + governance hash fix
- **Wave 1** (parallel): WP02, WP03, WP04
- **Wave 2**: WP05 → WP06 → WP07
