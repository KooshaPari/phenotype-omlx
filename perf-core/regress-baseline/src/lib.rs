//! `regress-baseline` — deterministic regression baselines for model-runtime
//! kernels.
//!
//! A *baseline* is a known-good `(inputs, outputs)` pair for a named
//! kernel. The crate gives callers a [`BaselineRecorder`] that:
//!
//! - hashes the inputs (so a stale baseline for a different shape is
//!   detected as a mismatch),
//! - stores the outputs as a JSON object under `baselines.json`,
//! - verifies subsequent runs against the recorded baseline,
//! - surfaces a structured [`VerifyResult`] on mismatch so the caller
//!   knows which field drifted.
//!
//! ## File format
//!
//! One JSON file `baselines.json` per [`BaselineRecorder::output_dir`].
//! Schema:
//!
//! ```json
//! {
//!   "schema_version": 1,
//!   "baselines": {
//!     "<kernel_name>": {
//!       "input_hash": "<lowercase hex sha256>",
//!       "output": <arbitrary JSON object>
//!     }
//!   }
//! }
//! ```
//!
//! `schema_version` is `1` for this revision; bumping it forces a manual
//! review of every checked-in baseline.
//!
//! ## Determinism
//!
//! [`BaselineRecorder`] is pure: it does no I/O outside the configured
//! directory, holds no global state, and serializes inputs with a stable
//! JSON representation so the same input vector always hashes the same.
//!
//! ## Shape-bucketed dispatch and energy budgets
//!
//! The [`dispatch_budget`] / [`energy_budget_j`] helpers return the
//! ceiling this crate expects for a single timed invocation of a kernel
//! with a given [`ShapeKey`]. They are deliberately stubbed (returning
//! `u64::MAX` / `f64::INFINITY`) so the regression test in
//! `tests/dispatch_buckets.rs` can be wired up before any real budget
//! numbers are committed; the test prints observed numbers and forces a
//! follow-up commit to populate these helpers with measured ceilings.
//!
//! ## Module layout
//!
//! - [`types`] — pure data: `BaselineEntry`, `BaselinesFile`,
//!   `VerifyResult`, `BaselineError`, `SCHEMA_VERSION`.
//! - [`recorder`] — `BaselineRecorder` (the I/O surface).
//! - [`json_diff`] — `canonicalize` + `find_first_diff` (private).
//! - [`budget`] — `ShapeKey`, `dispatch_budget`, `energy_budget_j`.

#![deny(unsafe_code)]

pub mod budget;
pub mod recorder;
pub mod types;

mod json_diff;

// Re-export every public symbol so external callers do not need to
// change. The internal modules are also exposed as `pub mod` above so
// callers can opt into per-module paths if they prefer.
pub use crate::budget::{dispatch_budget, energy_budget_j, ShapeKey};
pub use crate::recorder::BaselineRecorder;
pub use crate::types::{BaselineEntry, BaselineError, BaselinesFile, VerifyResult, SCHEMA_VERSION};

#[cfg(test)]
mod shape_key_tests {
    use super::*;

    #[test]
    fn output_cells_matches_manual() {
        let k = ShapeKey::new(4, 7, 9);
        assert_eq!(k.output_cells(), 28);
    }

    #[test]
    fn flops_matches_manual() {
        // 2 * M * N * K
        let k = ShapeKey::new(4, 7, 9);
        assert_eq!(k.flops(), 2 * 4 * 7 * 9);
    }

    #[test]
    fn dispatch_budget_stub_returns_max() {
        let k = ShapeKey::new(8, 16, 32);
        assert_eq!(dispatch_budget(&k), u64::MAX);
    }

    #[test]
    fn energy_budget_stub_returns_infinity() {
        let k = ShapeKey::new(8, 16, 32);
        assert_eq!(energy_budget_j(&k), f64::INFINITY);
    }
}
