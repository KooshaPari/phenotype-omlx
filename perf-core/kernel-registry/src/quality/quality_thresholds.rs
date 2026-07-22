//! Threshold definitions: gate direction and quality gate.
//!
//! Split from `quality.rs` to isolate threshold-related types from scoring
//! and promotion logic.

use serde::{Deserialize, Serialize};

/// How a [`QualityGate`] compares its score against its threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateDirection {
    /// Higher is better. `score >= threshold` to pass.
    AtLeast,
    /// Lower is better. `score <= threshold` to pass. Used for
    /// perplexity, calibration error, and rejection rates.
    AtMost,
}

/// One named quality threshold. A gate is *passing* when
/// `evidence.score >= gate.threshold`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityGate {
    /// Stable short id used in rejection traces. Convention: lowercase
    /// hyphenated, e.g. `"mmlu-pro"`, `"gpqa-diamond"`, `"bfcl-ast"`.
    pub id: String,
    /// Threshold the gate enforces. The interpretation depends on
    /// `direction`: for `AtLeast` the score must be `>= threshold`, for
    /// `AtMost` the score must be `<= threshold`.
    pub threshold: f64,
    /// How to interpret `threshold`.
    pub direction: GateDirection,
    /// Optional human-readable note (provenance, dataset revision).
    #[serde(default)]
    pub note: String,
}

impl QualityGate {
    /// Convenience constructor for "score >= threshold".
    pub fn at_least(id: impl Into<String>, threshold: f64) -> Self {
        Self {
            id: id.into(),
            threshold,
            direction: GateDirection::AtLeast,
            note: String::new(),
        }
    }

    /// Convenience constructor for "score <= threshold" (e.g. perplexity).
    pub fn at_most(id: impl Into<String>, threshold: f64) -> Self {
        Self {
            id: id.into(),
            threshold,
            direction: GateDirection::AtMost,
            note: String::new(),
        }
    }

    /// `true` when `score` satisfies the gate under its direction.
    pub fn passes(&self, score: f64) -> bool {
        match self.direction {
            GateDirection::AtLeast => score >= self.threshold,
            GateDirection::AtMost => score <= self.threshold,
        }
    }

    /// Short stable tag for use in rejection strings.
    pub fn tag(&self) -> &str {
        &self.id
    }
}
