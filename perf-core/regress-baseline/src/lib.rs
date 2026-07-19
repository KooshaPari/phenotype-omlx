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

#![deny(unsafe_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
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

/// Result of a [`BaselineRecorder::verify`] call.
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

/// Errors produced by [`BaselineRecorder`].
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

/// Recorder that owns one `baselines.json` file.
#[derive(Debug, Clone)]
pub struct BaselineRecorder {
    output_dir: PathBuf,
}

impl BaselineRecorder {
    /// Create a recorder rooted at `output_dir`. The directory is created
    /// lazily on the first write; reading from a missing directory
    /// returns an empty baseline set.
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
        }
    }

    /// The directory this recorder writes to.
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    /// Stable-JSON hash of an inputs payload. The serialization sorts map
    /// keys so structurally-equivalent inputs hash the same regardless
    /// of declaration order.
    pub fn hash_inputs(inputs: &Value) -> String {
        let canonical = canonicalize(inputs);
        let bytes = serde_json::to_vec(&canonical).expect("regress-baseline: Value is JSON-safe");
        let mut h = Sha256::new();
        h.update(&bytes);
        let digest = h.finalize();
        let mut out = String::with_capacity(64);
        for b in digest {
            out.push_str(&format!("{:02x}", b));
        }
        out
    }

    /// Read the on-disk `baselines.json`, returning an empty envelope if
    /// the file does not exist.
    pub fn load(&self) -> Result<BaselinesFile, BaselineError> {
        let path = self.baselines_path();
        if !path.exists() {
            return Ok(BaselinesFile {
                schema_version: SCHEMA_VERSION,
                baselines: BTreeMap::new(),
            });
        }
        let raw = std::fs::read_to_string(&path)?;
        let parsed: BaselinesFile = serde_json::from_str(&raw)?;
        if parsed.schema_version != SCHEMA_VERSION {
            return Err(BaselineError::SchemaVersion {
                got: parsed.schema_version,
                expected: SCHEMA_VERSION,
            });
        }
        Ok(parsed)
    }

    /// Record or overwrite `kernel_name`'s baseline. Persists immediately.
    pub fn record(
        &self,
        kernel_name: &str,
        inputs: &Value,
        outputs: Value,
    ) -> Result<BaselineEntry, BaselineError> {
        let mut file = self.load()?;
        let entry = BaselineEntry {
            input_hash: Self::hash_inputs(inputs),
            output: outputs,
        };
        file.baselines.insert(kernel_name.to_string(), entry.clone());
        self.write(&file)?;
        Ok(entry)
    }

    /// Verify `kernel_name` against the recorded baseline.
    pub fn verify(
        &self,
        kernel_name: &str,
        inputs: &Value,
        outputs: &Value,
    ) -> Result<VerifyResult, BaselineError> {
        let file = self.load()?;
        let entry = match file.baselines.get(kernel_name) {
            Some(e) => e,
            None => {
                return Ok(VerifyResult::Mismatch {
                    field: "<entry>".to_string(),
                    expected: Value::Null,
                    actual: Value::String("<missing>".to_string()),
                })
            }
        };
        let actual_hash = Self::hash_inputs(inputs);
        if actual_hash != entry.input_hash {
            return Ok(VerifyResult::InputHashMismatch {
                expected: entry.input_hash.clone(),
                actual: actual_hash,
            });
        }
        match find_first_diff(&entry.output, outputs, "") {
            Some((field, expected, actual)) => Ok(VerifyResult::Mismatch {
                field,
                expected,
                actual,
            }),
            None => Ok(VerifyResult::Ok),
        }
    }

    fn baselines_path(&self) -> PathBuf {
        self.output_dir.join("baselines.json")
    }

    fn write(&self, file: &BaselinesFile) -> Result<(), BaselineError> {
        std::fs::create_dir_all(&self.output_dir)?;
        let json = serde_json::to_string_pretty(file)?;
        let path = self.baselines_path();
        std::fs::write(path, json)?;
        Ok(())
    }
}

/// Recursively sort map keys so canonical-JSON hash is order-independent.
fn canonicalize(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut sorted: BTreeMap<String, Value> = BTreeMap::new();
            for (k, vv) in map {
                sorted.insert(k.clone(), canonicalize(vv));
            }
            let mut out = serde_json::Map::new();
            for (k, vv) in sorted {
                out.insert(k, vv);
            }
            Value::Object(out)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(canonicalize).collect()),
        other => other.clone(),
    }
}

/// Walk two JSON values in parallel and return the dotted path of the
/// first mismatch (or `None` if they are equal). Path is built from map
/// keys and array indices (`values.0`, `expert_buckets.2.0`, ...).
fn find_first_diff(expected: &Value, actual: &Value, path: &str) -> Option<(String, Value, Value)> {
    if expected == actual {
        return None;
    }
    match (expected, actual) {
        (Value::Object(e), Value::Object(a)) => {
            // Check all expected keys in stable order.
            let mut keys: Vec<&String> = e.keys().collect();
            keys.sort();
            for k in keys {
                let exp_v = &e[k];
                let act_v = a.get(k).unwrap_or(&Value::Null);
                let child = format!("{}.{}", if path.is_empty() { k.clone() } else { format!("{path}.{k}") }, "");
                if let Some(diff) = find_first_diff(exp_v, act_v, &child[..child.len() - 1]) {
                    return Some(diff);
                }
            }
            // Surplus keys on `actual` count as a mismatch.
            for k in a.keys() {
                if !e.contains_key(k) {
                    let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                    return Some((p, Value::Null, a[k].clone()));
                }
            }
            None
        }
        (Value::Array(e), Value::Array(a)) => {
            let n = e.len().max(a.len());
            for i in 0..n {
                let exp_v = e.get(i).unwrap_or(&Value::Null);
                let act_v = a.get(i).unwrap_or(&Value::Null);
                let p = if path.is_empty() {
                    format!("{i}")
                } else {
                    format!("{path}.{i}")
                };
                if let Some(diff) = find_first_diff(exp_v, act_v, &p) {
                    return Some(diff);
                }
            }
            None
        }
        _ => Some((path.to_string(), expected.clone(), actual.clone())),
    }
}
