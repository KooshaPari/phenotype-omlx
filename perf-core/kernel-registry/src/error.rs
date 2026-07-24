//! Crate-wide error types for `kernel-registry`.
//!
//! Two distinct error categories are modelled:
//!
//! - [`Error`]: structural failures of the registry itself (duplicate
//!   registration, malformed shapes). These are uncommon and indicate a
//!   programming or environment fault.
//! - [`crate::tuner::TunerError`]: failures of the bounded tuner, including
//!   budget exhaustion. These are runtime signals that the caller should
//!   handle by recording an [`crate::ExecutionTrace`] and falling back.
//!
//! The split keeps the hot path (selection and execution) free of
//! non-essential error machinery.

use serde::{Deserialize, Serialize};

/// Structural registry errors.
#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Error {
    #[error("candidate id {id} already registered with a different name (existing={existing}, new={new})")]
    DuplicateCandidateId {
        id: u64,
        existing: String,
        new: String,
    },
    #[error("tuning record references unknown candidate {0}")]
    UnknownCandidate(u64),
    #[error("invalid shape: {message}")]
    InvalidShape { message: String },
    #[error("invalid quantization policy: {message}")]
    InvalidQuantization { message: String },
}

pub type Result<T> = std::result::Result<T, Error>;
