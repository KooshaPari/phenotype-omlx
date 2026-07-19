//! Kernel registry: candidate metadata, capability matching, deterministic
//! selection policy, tuning records, and execution traces.
//!
//! See `docs/sessions/20260718-metal-model-runtime/02_SPECIFICATIONS.md`
//! ("Kernel Selection Contract") and `04_IMPLEMENTATION_STRATEGY.md`
//! ("Selection Process") for the contractual requirements implemented here.
//!
//! ## Module layout
//! - [`key`]: [`KernelKey`](key::KernelKey), [`ShapeSignature`](key::ShapeSignature),
//!   and `fast_hash` for stable identity.
//! - [`candidate`]: [`Candidate`](candidate::Candidate), [`CandidateId`](candidate::CandidateId),
//!   [`BackendKind`](candidate::BackendKind), [`Capability`](candidate::Capability).
//! - [`record`]: [`TuningRecord`](record::TuningRecord), [`Measurement`](record::Measurement).
//! - [`trace`]: [`ExecutionTrace`](trace::ExecutionTrace) emitted on every selection.
//! - [`selector`]: [`SelectionPolicy`](selector::SelectionPolicy) and
//!   [`SelectionDecision`](selector::SelectionDecision).
//! - [`registry`]: [`KernelRegistry`](registry::KernelRegistry) storage and entry points.
//! - [`tuner`]: [`BoundedTuner`](tuner::BoundedTuner) with warmup + budget enforcement.
//! - [`error`]: [`Error`](error::Error) and [`Result`](error::Result).
//!
//! ## Determinism
//! All selector tie-breaks are on a `Vec` that is sorted by `(metric, candidate_id)`
//! ascending. HashMap iteration is never used to determine selection order.
//!
//! ## compat module
//! Until the `model-plan` crate is created in parallel (Task 2), this crate
//! ships a minimal [`compat`] mirror of the subset of model-plan types it
//! consumes (`OperatorKind`, `AttentionKind`, `QuantizationPolicy`, `DType`).
//! When `model-plan` is stable, the next pass should:
//!
//! 1. Add `model-plan` as a `kernel-registry` dependency.
//! 2. Replace every `compat::TypeName` with `model_plan::TypeName`.
//! 3. Delete this module.

#![deny(unsafe_code)]

pub mod candidate;
pub mod compat;
pub mod error;
pub mod key;
pub mod record;
pub mod registry;
pub mod selector;
pub mod trace;
pub mod tuner;

pub use candidate::{BackendKind, Candidate, CandidateId, Capability};
pub use compat::{AttentionKind, DType, OperatorKind, QuantizationPolicy};
pub use error::{Error, Result};
pub use key::{fast_hash_bytes, KernelKey, ShapeSignature, ATTENTION_NONE};
pub use record::{Measurement, TuningRecord};
pub use registry::{DeviceCaps, KernelRegistry};
pub use selector::{
    Metric, RejectionReason, RejectionRecord, SelectionDecision, SelectionPolicy,
};
pub use trace::{ExecutionTrace, TraceRejection};
pub use tuner::{BoundedTuner, TunerError};