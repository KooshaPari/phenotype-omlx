//! Core public types for `regress-baseline`.
//!
//! These types are pure data + enum carriers; they have no I/O or
//! hashing logic of their own. The hashing + diffing helpers live in
//! [`crate::json_diff`]; the recorder that uses them lives in
//! [`crate::recorder`]; the shape-bucketed budget helpers live in
//! [`crate::budget`].

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Current on-disk schema version. Bumping this forces every consumer to
/// acknowledge the format change.
pub const SCHEMA_VERSION: u32 = 1;

/// One recorded baseline entry: the hash of the inputs that produced the
/// outputs, and the outputs themselves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineEntry {
    /// Lowercase hex SHA-256 of the stable-JSON encoding of the inputs.
    pub input_hash: String,
    /// Captured output payload, stored as an opaque JSON object so the
    /// recorder does not impose a schema on each kernel.
    pub output: Value,
}

/// On-disk envelope for `baselines.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselinesFile {
    /// Always [`SCHEMA_VERSION`] for this crate revision.
    pub schema_version: u32,
    /// Map from kernel name to its recorded baseline.
    pub baselines: BTreeMap<String, BaselineEntry>,
}

/// Result of a [`crate::BaselineRecorder::verify`] call.
#[derive(Debug, Clone, PartialEq)]
pub enum VerifyResult {
    /// Inputs and outputs match the recorded baseline.
    Ok,
    /// The recorded baseline was for a different input shape/hash. The
    /// caller should re-record before comparing outputs.
    InputHashMismatch {
        /// Hash the baseline was recorded under.
        expected: String,
        /// Hash of the inputs supplied to `verify`.
        actual: String,
    },
    /// The input hashes match but the output field drifted. `field` is a
    /// dotted JSON path (e.g. `"values.0"`); `expected` and `actual` are
    /// the recorded and supplied values respectively.
    Mismatch {
        /// Dotted JSON path to the first drifted field.
        field: String,
        /// Recorded value at `field`.
        expected: Value,
        /// Value supplied to `verify`.
        actual: Value,
    },
}

/// Errors produced by [`crate::BaselineRecorder`].
#[derive(Debug, Error)]
pub enum BaselineError {
    /// Underlying I/O failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// `baselines.json` could not be parsed.
    #[error("malformed baselines.json: {0}")]
    Parse(#[from] serde_json::Error),
    /// The on-disk schema version is not [`SCHEMA_VERSION`].
    #[error("unsupported baselines schema version {got} (expected {expected})")]
    SchemaVersion { got: u32, expected: u32 },
}
