//! Recurrent-hybrid selector coverage for the SOTA operator families
//! listed in `docs/sessions/20260718-metal-model-runtime/02_SPECIFICATIONS.md`:
//!
//! - (a) Mamba selective scan — `OperatorKind::Scan`, head_dim=8, chunk_size=16
//! - (i) RWKV-7             — `OperatorKind::Recurrent`, state_channels=4
//! - (j) dispatch_buckets_recurrent — per-shape envelope for the *batched*
//!   DeltaNet, Mamba, and RWKV selectors.
//!
//! The submodule split mirrors the operator-family boundaries so each
//! per-topic file stays under the project's 350-line target and 500-line
//! hard cap. `super::recurrent::mamba_key()` and `super::recurrent::mamba_registry()`
//! continue to satisfy the parent module's `use` sites unchanged.

mod dispatch_envelope;
mod mamba_extended;
mod mamba_scan;
mod rwkv7;
mod rwkv_extended;

// Re-export the parent module's shared builder helpers (`build_record`,
// `fresh_capabilities`, `make_candidate`, `samples_with_p95`, `shape`,
// `NOW_UNIX_MS`, `TEST_FINGERPRINT`, `build_record_with_dispatches`) so
// each per-topic submodule can keep using the `super::{...}` form for
// these utilities. The parent (`main.rs`) already exposes them as
// `pub(crate)`, so `pub(crate) use` here preserves that visibility.
pub(crate) use super::{
    build_record, build_record_with_dispatches, fresh_capabilities, make_candidate,
    samples_with_p95, shape, NOW_UNIX_MS, TEST_FINGERPRINT,
};

// Re-export the `pub` helpers from the per-topic submodules so the parent
// `main.rs` test module can keep its existing call sites:
// `super::recurrent::mamba_key()` / `super::recurrent::mamba_registry()`.
// This is purely an integration test crate.
pub use mamba_scan::{mamba_key, mamba_registry};
