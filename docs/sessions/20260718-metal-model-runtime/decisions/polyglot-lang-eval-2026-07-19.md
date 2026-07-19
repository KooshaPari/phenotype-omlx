# Polyglot language evaluation — phenotype-omlx turbo-quant / vPU stack

**Date**: 2026-07-19  
**Mandate**: No Cargo feature optionality for lang backends; toolchains required or build fails.  
**Feature**: `complete-polyglot-vpu-stack`

## Already in stack (keep, ungate)

| Lang | Role | Toolchain (host) | Gate status |
|------|------|------------------|-------------|
| Rust | primary | cargo | required |
| C | ABI SSOT | clang | required |
| Go | e2e vs C | go 1.26.5 | required (tests invoke `go test`) |
| Nim | e2e vs C | nim 2.2.10 | required |
| Zig | kernel twin | zig 0.16 | **was optional `--features zig` → REMOVE** |
| Mojo | ML/kernel twin | mojo 1.0.0b3 | **was optional `--features mojo` → REMOVE** |

## Evaluated candidates (~10)

| # | Language | Fit for turbo-quant / vPU | Verdict | Why |
|---|----------|---------------------------|---------|-----|
| 1 | **Odin** | Excellent C ABI twin; data-oriented; mature 2026 | **BRING IN** | Best next kernel-parity language; simple FFI; commercial systems use |
| 2 | **Pony** | Strong for multi-device agent/orchestration, not raw kernels | **BRING IN (orchestration)** | ponyc 0.67 installed; 2026 C-shim + embedded linker; capabilities model fits FR-6/7 harnesses |
| 3 | **Swift** | Apple Silicon / Metal adjacency | **BRING IN (Apple path)** | swiftc 6.4 present; natural for Metal runtime glue, not portable CI |
| 4 | **Crystal** | Nice Ruby-like + LLVM; GC | **EVALUATE / thin binding** | crystal 1.20 installed; good for tooling/CLIs around eval, not hot kernels |
| 5 | **Julia** | ML eval / NIAH / scientific | **BRING IN (eval path)** | Not installed yet — install + wire FR-5 scripts; high value for model eval |
| 6 | **Hare** | Minimal C-like systems | **DEFER** | Not installed; qbe backend; less SIMD/ML ecosystem vs Zig/Odin |
| 7 | **Vale (lang)** | Memory-safe systems research | **DEFER** | `vale` on PATH is **prose linter**, not Vale-lang; immature compiler packaging |
| 8 | **Austral** | Linear types / high assurance | **DEFER** | Niche; weak ML/SIMD ecosystem; high learning cost |
| 9 | **Carbon** | C++ interop experiment | **SKIP now** | Not production; Google experiment status |
| 10 | **V** | Simple C-like | **DEFER** | Not installed; churny compiler history; Zig/Odin already cover niche |
| 11 | **Chapel** | Parallel HPC | **DEFER** | Heavy runtime; overkill for kernel ABI twins |
| 12 | **Fortran** | Numeric kernels | **SKIP** | No incremental value over Mojo/Rust/C for this stack |

## Decision order (forward-only)

1. **Ungate Mojo + Zig** — always build; panic if SDK missing.  
2. **Install Julia** — required for FR-5 eval scripts (no optional path).  
3. **Add `turbo-quant-odin`** — C ABI parity tests (mirror Go/Nim).  
4. **Add Pony harness crate/scripts** — multi-device vPU orchestration (not duplicate kernels).  
5. **Swift Metal glue** only where Metal runtime needs it (required on darwin-arm64 builds that touch Metal).  
6. Crystal thin CLI wrappers only if a concrete FR needs them.

## Anti-patterns (banned)

- `#[cfg(feature = "mojo")]` / `zig` stubs that “succeed” without the language  
- Silent fallbacks to Rust when foreign kernel missing  
- Documenting “optional SDK” as acceptable CI green
