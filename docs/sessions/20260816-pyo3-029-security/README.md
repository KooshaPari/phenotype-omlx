# PyO3 0.29 security and CPython 3.14t validation

## Outcome

- Upgraded the isolated Python FFI crate from PyO3 0.22.6 to 0.29.2.
- Cleared `RUSTSEC-2025-0020` and `RUSTSEC-2026-0177` in the crate lockfile.
- Migrated removed PyO3 APIs to the 0.29 `Bound` / `Py<PyAny>` API.
- Retained optimized release builds with fat LTO.
- Added a CPython 3.14t release-wheel smoke gate to FFI CI.

## Root-cause experiment

`python/ffi/.cargo/config.toml` force-set `PYO3_PYTHON` to the local
GIL-enabled framework build. Cargo therefore ignored Maturin's explicit
CPython 3.14t interpreter when building a `cp314t` wheel, leaving the
extension compiled for the wrong ABI and causing its initializer to return an
uninitialized object at import time.

The configuration now provides the framework interpreter only as a default;
an explicit Maturin target wins. A fresh, installed CPython 3.14t wheel
imports with `-X gil=0` under both thin and fat LTO. Fat LTO is retained.

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
that interpreter yet. The CI workflow now repeats the installed-wheel smoke
test under CPython 3.14t with the GIL disabled.

## Non-claims

This is a binding/API and artifact-import validation only. It does not prove
MLX model loading, Qwen3.5 inference, Metal execution, Harbor evidence, or
promotion readiness.
