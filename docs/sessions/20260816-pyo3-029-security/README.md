# PyO3 0.29 security and CPython 3.14t validation

## Outcome

- Upgraded the isolated Python FFI crate from PyO3 0.22.6 to 0.29.2.
- Cleared `RUSTSEC-2025-0020` and `RUSTSEC-2026-0177` in the crate lockfile.
- Migrated removed PyO3 APIs to the 0.29 `Bound` / `Py<PyAny>` API.
- Retained optimized release builds with thin LTO.

## Root-cause experiment

The original `lto = true` release artifact returned an uninitialized object
when imported by CPython 3.14t. The same source imported in debug, and a
release artifact with either `lto = false` or `lto = "thin"` imported with
the GIL disabled. A minimal PyO3 0.29 release probe with fat LTO also
imported, so the fault is the interaction between fat LTO and this FFI
crate's linked dependency surface rather than PyO3 or CPython 3.14t alone.

`lto = "thin"` is the narrowest working release setting tested here.

## Validation

All commands were run without a model, Harbor, Metal, or live evaluation:

```text
cargo check --locked --features extension-module      PASS
cargo test --locked -q                                PASS (6)
cargo audit --file Cargo.lock                         PASS (90 dependencies)
cargo fmt --check                                     PASS
git diff --check                                      PASS
CPython 3.14t release wheel import with -X gil=0      PASS
CPython 3.14 release abi3 wheel import                PASS
```

The 3.14t wheel is intentionally version-specific because PyO3 reports that
the requested `abi3-py311` configuration cannot provide an abi3 artifact for
that interpreter yet.

## Non-claims

This is a binding/API and artifact-import validation only. It does not prove
MLX model loading, Qwen3.5 inference, Metal execution, Harbor evidence, or
promotion readiness.
