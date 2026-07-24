//! Integration / contract tests for the spec-decode engine.
//!
//! These tests cover three contracts:
//!   * Engine state ownership — counters and history are observable.
//!   * Medusa proposal pipeline — multi-head drafts produce candidates.
//!   * Verification determinism — verify() is reproducible and rejects
//!     malformed inputs cleanly.

#[path = "contracts/engine_tests.rs"]
mod engine_tests;
#[path = "contracts/proposal_tests.rs"]
mod proposal_tests;

pub(crate) use async_trait::async_trait;
pub(crate) use spec_decode::{
    build_engine, dedup_preserve, verify_draft, AcceptedToken, BackendInfo, DraftMode, EngineState,
    MedusaHead, MedusaProposal, MockMedusaHead, SharedEngine, SpecDecodeConfig, SpecDecodeEngine,
    SpecError, StepResult, TargetBackend, TargetOutput, TreeTopology, VerifyResult, HISTORY_CAP,
};

// -----------------------------------------------------------------------------
// Test backends
// -----------------------------------------------------------------------------

pub(crate) struct ScriptedTarget {
    logits: Vec<f32>,
}

impl ScriptedTarget {
    pub(crate) fn with_argmax(vocab: usize, idx: u32) -> Self {
        let mut logits = vec![-10.0_f32; vocab];
        logits[idx as usize] = 10.0;
        Self { logits }
    }
}

#[async_trait]
impl TargetBackend for ScriptedTarget {
    async fn forward(&self, _token_ids: &[u32]) -> Result<TargetOutput, String> {
        Ok(TargetOutput {
            logits: self.logits.clone(),
            hidden: None,
            finished: false,
        })
    }
    async fn verify_tree(
        &self,
        _prefix: &[u32],
        candidates: &[Vec<u32>],
    ) -> Result<Vec<bool>, String> {
        Ok(candidates.iter().map(|c| !c.is_empty()).collect())
    }
    fn info(&self) -> BackendInfo {
        BackendInfo {
            engine: "scripted".into(),
            model_id: "test-scripted".into(),
            device: "cpu".into(),
            dtype: "f32".into(),
            kv_cache_type: None,
        }
    }
}

pub(crate) struct NullDraft;
#[async_trait]
impl spec_decode::DraftBackend for NullDraft {
    async fn draft(&self, _prefix: &[u32], _max: usize) -> Result<Vec<u32>, String> {
        Ok(Vec::new())
    }
    fn info(&self) -> BackendInfo {
        BackendInfo {
            engine: "null".into(),
            model_id: "test-null".into(),
            device: "n/a".into(),
            dtype: "n/a".into(),
            kv_cache_type: None,
        }
    }
}

pub(crate) fn engine_with(config: SpecDecodeConfig) -> SpecDecodeEngine {
    SpecDecodeEngine::new(config, Box::new(ScriptedTarget::with_argmax(64, 5)), None)
}

// -----------------------------------------------------------------------------
// Smoke checks for re-exports and handle type
// -----------------------------------------------------------------------------

#[test]
fn shared_engine_handle_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SharedEngine>();
}

#[test]
fn step_result_serialization_smoke() {
    let r = StepResult {
        accepted: vec![AcceptedToken {
            token_id: 7,
            was_drafted: true,
        }],
        drafted: 3,
        finished: false,
    };
    let j = serde_json::to_string(&r).unwrap();
    assert!(j.contains("\"accepted\""));
    assert!(j.contains("\"drafted\":3"));
}

#[test]
fn verify_result_serialization_smoke() {
    let v = VerifyResult {
        accepted_prefix: vec![1, 2],
        first_reject_idx: Some(2),
        bonus_token: Some(9),
        seed: Some(0xDEAD_BEEF),
    };
    let j = serde_json::to_string(&v).unwrap();
    assert!(j.contains("\"accepted_prefix\":[1,2]"));
}

#[test]
fn build_engine_returns_shared_handle() {
    let cfg = SpecDecodeConfig::default();
    let h: SharedEngine = build_engine(cfg, Box::new(ScriptedTarget::with_argmax(8, 1)), None);
    let lock = h.try_lock();
    assert!(lock.is_ok());
}

#[test]
fn medusa_proposal_tree_topology_serializes() {
    let t = TreeTopology { width: 4, depth: 2 };
    let j = serde_json::to_string(&t).unwrap();
    let back: TreeTopology = serde_json::from_str(&j).unwrap();
    assert_eq!(back.width, 4);
    assert_eq!(back.depth, 2);
}

#[test]
fn spec_error_display_strings_are_stable() {
    let _ = format!("{}", SpecError::DraftNotLoaded);
    let _ = format!("{}", SpecError::AllRejected { n: 4 });
    let _ = format!("{}", SpecError::Config("bad".into()));
}
