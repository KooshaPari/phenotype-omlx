//! spec-decode — High-performance speculative decoding engine.
//!
//! Rust rewrite of the turbo_mlx.ssd Python speculative decoder, targeting
//! Apple Silicon performance cores via Metal compute when available and
//! falling back to portable CPU paths otherwise.
//!
//! Supports three draft strategies (mirroring the Python reference):
//!   - SameModel: n-gram / prompt lookup matching against the live KV cache
//!   - DraftModel: a smaller companion model proposes tokens
//!   - Medusa: multiple parallel draft heads on the target model
//!
//! The public surface is engine-agnostic. Concrete backends (MLX, Metal,
//! CUDA, vLLM, SGLang, llama.cpp) are pluggable via the [`TargetBackend`]
//! and [`DraftBackend`] traits.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::Mutex;

#[cfg(feature = "metal")]
mod metal;

pub mod backend;
pub mod engine;
pub mod proposal;
pub mod proposal_state;
pub mod state;
pub mod verify;

pub use backend::{BackendInfo, DraftBackend, NullDraftBackend, TargetBackend, TargetOutput};
pub use engine::{DraftCandidate, SpecDecodeEngine, SpecStats};
pub use proposal::{MedusaHead, MedusaProposal, MockMedusaHead, TreeTopology};
pub use proposal_state::ProposalState;
pub use state::{EngineState, HISTORY_CAP};
pub use verify::{verify as verify_draft, VerifyResult};

/// Draft strategy selection — mirrors `turbo_mlx.ssd.DraftMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DraftMode {
    /// Prompt-lookup / n-gram matching against the running KV cache.
    /// Zero-cost drafting — reuses tokens already seen.
    SameModel,
    /// A smaller companion model proposes continuation tokens.
    DraftModel,
    /// Multiple parallel draft heads attached to the target model.
    Medusa,
}

impl DraftMode {
    pub const ALL: [DraftMode; 3] = [
        DraftMode::SameModel,
        DraftMode::DraftModel,
        DraftMode::Medusa,
    ];
}

impl std::fmt::Display for DraftMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DraftMode::SameModel => write!(f, "same_model"),
            DraftMode::DraftModel => write!(f, "draft_model"),
            DraftMode::Medusa => write!(f, "medusa"),
        }
    }
}

/// Configuration for the speculative decoding engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecDecodeConfig {
    /// Draft strategy.
    pub mode: DraftMode,
    /// Maximum number of tokens the drafter may propose per step.
    pub max_draft_tokens: usize,
    /// Number of top-k tokens to sample for tree expansion (Medusa).
    #[serde(default = "default_tree_width")]
    pub tree_width: usize,
    /// Maximum tree depth for Medusa verification.
    #[serde(default = "default_tree_depth")]
    pub tree_depth: usize,
    /// Temperature for draft sampling (0 = greedy).
    #[serde(default)]
    pub temperature: f32,
    /// Whether to fall back to vanilla autoregressive decode on rejection.
    #[serde(default = "default_true")]
    pub fallback_on_reject: bool,
}

fn default_tree_width() -> usize {
    4
}
fn default_tree_depth() -> usize {
    1
}
fn default_true() -> bool {
    true
}

impl Default for SpecDecodeConfig {
    fn default() -> Self {
        Self {
            mode: DraftMode::SameModel,
            max_draft_tokens: 4,
            tree_width: default_tree_width(),
            tree_depth: default_tree_depth(),
            temperature: 0.0,
            fallback_on_reject: true,
        }
    }
}

/// Errors emitted by the speculative decoding engine.
#[derive(Debug, Error)]
pub enum SpecError {
    #[error("backend error: {0}")]
    Backend(String),
    #[error("draft model not loaded")]
    DraftNotLoaded,
    #[error("verification failed: all {n} draft tokens rejected")]
    AllRejected { n: usize },
    #[error("configuration error: {0}")]
    Config(String),
    #[error("cancelled by caller")]
    Cancelled,
}

/// One accepted token plus optional debug metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptedToken {
    pub token_id: u32,
    pub was_drafted: bool,
}

/// Result of a single speculative step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub accepted: Vec<AcceptedToken>,
    pub drafted: usize,
    pub finished: bool,
}

/// Thread-safe handle returned to Python / FFI callers.
pub type SharedEngine = Arc<Mutex<SpecDecodeEngine>>;

/// Build a ready-to-use engine handle from a config + backends.
pub fn build_engine(
    config: SpecDecodeConfig,
    target: Box<dyn TargetBackend>,
    draft: Option<Box<dyn DraftBackend>>,
) -> SharedEngine {
    Arc::new(Mutex::new(SpecDecodeEngine::new(config, target, draft)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default() {
        let c = SpecDecodeConfig::default();
        assert_eq!(c.mode, DraftMode::SameModel);
        assert_eq!(c.max_draft_tokens, 4);
    }

    #[test]
    fn draft_mode_serde() {
        let json = serde_json::to_string(&DraftMode::Medusa).unwrap();
        assert_eq!(json, "\"medusa\"");
        let back: DraftMode = serde_json::from_str("\"draft_model\"").unwrap();
        assert_eq!(back, DraftMode::DraftModel);
    }
}
