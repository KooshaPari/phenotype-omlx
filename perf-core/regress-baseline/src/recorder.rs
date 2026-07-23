//! `BaselineRecorder` — owns one `baselines.json` file.
//!
//! This is the I/O surface of the crate. The recorder:
//!
//! - hashes the inputs (so a stale baseline for a different shape is
//!   detected as a mismatch),
//! - stores the outputs as a JSON object under `baselines.json`,
//! - verifies subsequent runs against the recorded baseline,
//! - surfaces a structured [`VerifyResult`] on mismatch so the caller
//!   knows which field drifted.
//!
//! Pure-data types (the envelope, the result enum, the error type) live
//! in [`crate::types`]; the hashing primitives live in
//! [`crate::json_diff`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::json_diff::{canonicalize, find_first_diff};
use crate::types::{BaselineEntry, BaselineError, BaselinesFile, VerifyResult, SCHEMA_VERSION};

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
        file.baselines
            .insert(kernel_name.to_string(), entry.clone());
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
