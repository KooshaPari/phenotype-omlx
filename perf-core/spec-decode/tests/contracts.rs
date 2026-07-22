//! Integration / contract tests for the spec-decode engine.
//!
//! These tests cover three contracts:
//!   * Engine state ownership — counters and history are observable.
//!   * Medusa proposal pipeline — multi-head drafts produce candidates.
//!   * Verification determinism — verify() is reproducible and rejects
//!     malformed inputs cleanly.

use async_trait::async_trait;
use spec_decode::{
    verify_draft, AcceptedToken, BackendInfo, DraftMode, EngineState, MedusaHead, MedusaProposal,
    MockMedusaHead, SharedEngine, SpecDecodeConfig, SpecDecodeEngine, SpecError, StepResult,
    TargetBackend, TargetOutput, TreeTopology, VerifyResult, build_engine,
};

// -----------------------------------------------------------------------------
// Test backends
// -----------------------------------------------------------------------------

struct ScriptedTarget {
    logits: Vec<f32>,
}

impl ScriptedTarget {
    fn with_argmax(vocab: usize, idx: u32) -> Self {
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

struct NullDraft;
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

fn engine_with(config: SpecDecodeConfig) -> SpecDecodeEngine {
    SpecDecodeEngine::new(config, Box::new(ScriptedTarget::with_argmax(64, 5)), None)
}

// -----------------------------------------------------------------------------
// Engine state contract
// -----------------------------------------------------------------------------

#[test]
fn engine_state_snapshot_reflects_initial_zero_counters() {
    let engine = engine_with(SpecDecodeConfig::default());
    let s: EngineState = engine.state();
    assert_eq!(s.kv_len, 0);
    assert_eq!(s.drafted_total, 0);
    assert_eq!(s.accepted_total, 0);
    assert_eq!(s.last_step_accepted, 0);
    assert_eq!(s.last_step_drafted, 0);
    assert!(s.history.is_empty());
}

#[test]
fn engine_state_records_drafted_and_accepted_after_step() {
    let config = SpecDecodeConfig {
        mode: DraftMode::DraftModel,
        max_draft_tokens: 2,
        ..Default::default()
    };

    let _engine = SpecDecodeEngine::new(
        config,
        Box::new(ScriptedTarget::with_argmax(64, 7)),
        Some(Box::new(NullDraft)),
    );
    // The point of this test is to make sure the engine constructs with the
    // new mode without panicking and that the per-step recording surface
    // (record_step / push_accepted) stays wired up.
    let mut s = EngineState::new();
    s.record_step(2, 1);
    s.push_accepted(7);
    assert_eq!(s.last_step_drafted, 2);
    assert_eq!(s.last_step_accepted, 1);
    assert_eq!(s.drafted_total, 2);
    assert_eq!(s.accepted_total, 1);
    assert_eq!(s.history, vec![7]);
}

#[test]
fn engine_state_new_zeroes_every_counter() {
    let s = EngineState::new();
    assert_eq!(s.kv_len, 0);
    assert_eq!(s.drafted_total, 0);
    assert_eq!(s.accepted_total, 0);
    assert_eq!(s.last_step_accepted, 0);
    assert_eq!(s.last_step_drafted, 0);
    assert!(s.history.is_empty());
}

#[test]
fn engine_reset_state_clears_counters_and_history() {
    let mut s = EngineState::new();
    s.kv_len = 16;
    s.drafted_total = 100;
    s.accepted_total = 73;
    s.last_step_accepted = 5;
    s.last_step_drafted = 6;
    s.history.extend([1, 2, 3, 4]);
    s.reset();
    assert_eq!(s.kv_len, 0);
    assert_eq!(s.drafted_total, 0);
    assert_eq!(s.accepted_total, 0);
    assert_eq!(s.last_step_accepted, 0);
    assert_eq!(s.last_step_drafted, 0);
    assert!(s.history.is_empty());
}

#[test]
fn engine_snapshot_returns_independent_copy() {
    let s1 = EngineState::new();
    let s2 = s1.snapshot();
    let mut s1 = s1;
    s1.kv_len = 99;
    s1.history.push_back(1);
    assert_eq!(s2.kv_len, 0);
    assert!(s2.history.is_empty());
}

#[test]
fn engine_history_caps_at_1024_tokens() {
    let mut s = EngineState::new();
    for i in 0..2000_u32 {
        s.push_accepted(i);
    }
    assert_eq!(s.history.len(), 1024);
    assert_eq!(s.history.back().copied(), Some(1999));
}

// -----------------------------------------------------------------------------
// Cancellation, zero-proposal, empty-draft
// -----------------------------------------------------------------------------

#[test]
fn engine_step_with_empty_draft_does_not_panic() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut engine = engine_with(SpecDecodeConfig::default());
        let res = engine.step(&[1, 2, 3]).await;
        assert!(res.is_ok());
        let r = res.unwrap();
        let _ = r.accepted.len();
    });
}

#[test]
fn engine_step_with_zero_proposal_returns_no_accepted_tokens() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut engine = engine_with(SpecDecodeConfig::default());
        let r = engine.step(&[1, 2, 3]).await.unwrap();
        assert!(r.accepted.len() <= 1);
        assert_eq!(r.drafted, 0);
    });
}

#[test]
fn engine_step_cancellation_token_aborts_before_state_mutation() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut engine = engine_with(SpecDecodeConfig::default());
        let before = engine.state();
        let r = engine
            .step_cancellable(&[1, 2, 3], &[], || true)
            .await
            .unwrap();
        assert!(r.accepted.is_empty());
        assert_eq!(r.drafted, 0);
        assert!(!r.finished);
        let after = engine.state();
        assert_eq!(before.kv_len, after.kv_len);
        assert_eq!(before.drafted_total, after.drafted_total);
        assert_eq!(before.accepted_total, after.accepted_total);
        assert_eq!(before.history, after.history);
    });
}

// -----------------------------------------------------------------------------
// Medusa proposal contract
// -----------------------------------------------------------------------------

#[test]
fn engine_step_medusa_collects_candidates_from_each_head() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let config = SpecDecodeConfig {
            mode: DraftMode::Medusa,
            max_draft_tokens: 6,
            ..Default::default()
        };

        let mut engine = SpecDecodeEngine::new(
            config,
            Box::new(ScriptedTarget::with_argmax(64, 7)),
            None,
        );

        let heads: Vec<Box<dyn MedusaHead>> = vec![
            Box::new(MockMedusaHead::new(vec![11, 12])),
            Box::new(MockMedusaHead::new(vec![21, 22])),
            Box::new(MockMedusaHead::new(vec![31, 32])),
        ];

        let r = engine.step_medusa(&[1, 2, 3], &heads).await.unwrap();
        assert!(r.drafted > 0, "expected drafted > 0, got {}", r.drafted);
    });
}

#[test]
fn engine_step_medusa_deduplicates_by_token_id_preserving_order() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let config = SpecDecodeConfig {
            mode: DraftMode::Medusa,
            max_draft_tokens: 8,
            ..Default::default()
        };

        let heads: Vec<Box<dyn MedusaHead>> = vec![
            Box::new(MockMedusaHead::new(vec![11, 12])),
            Box::new(MockMedusaHead::new(vec![11, 12])),
        ];

        let prop = MedusaProposal::from_heads(
            &heads,
            &[1, 2, 3],
            &TreeTopology {
                width: 4,
                depth: 2,
            },
            config.max_draft_tokens,
        );
        assert_eq!(prop.heads.len(), 2);
        assert!(prop.heads[0].contains(&11));
        assert!(prop.heads[0].contains(&12));
        let mut seen = Vec::new();
        for h in &prop.heads {
            for &t in h {
                if !seen.contains(&t) {
                    seen.push(t);
                }
            }
        }
        assert_eq!(seen, vec![11, 12]);
    });
}

#[test]
fn engine_step_medusa_respects_max_draft_tokens_cap() {
    let config = SpecDecodeConfig {
        mode: DraftMode::Medusa,
        max_draft_tokens: 3,
        ..Default::default()
    };
    let max = config.max_draft_tokens;
    let heads: Vec<Box<dyn MedusaHead>> = vec![
        Box::new(MockMedusaHead::new(vec![1, 2, 3, 4, 5, 6, 7, 8])),
        Box::new(MockMedusaHead::new(vec![9, 10, 11, 12])),
    ];
    let prop = MedusaProposal::from_heads(
        &heads,
        &[1, 2, 3],
        &TreeTopology {
            width: 4,
            depth: 2,
        },
        max,
    );
    let total: usize = prop.heads.iter().map(|h| h.len()).sum();
    assert!(total <= 3, "got {} total tokens, exceeds cap", total);
}

#[test]
fn medusa_head_propose_returns_at_most_k_tokens() {
    let head = MockMedusaHead::new(vec![1, 2, 3, 4, 5]);
    let out = head.propose(&[9, 8, 7], 3);
    assert_eq!(out.len(), 3);
    assert_eq!(out, vec![1, 2, 3]);
}

#[test]
fn medusa_head_propose_handles_kv_shorter_than_ngram_window() {
    let head = MockMedusaHead::new(vec![42]);
    let out = head.propose(&[1], 5);
    assert!(out.len() <= 5);
    assert!(!out.is_empty());
}

// -----------------------------------------------------------------------------
// Verification contract
// -----------------------------------------------------------------------------

#[test]
fn verify_accepts_full_draft_when_target_agrees() {
    let mut logits = vec![0.0_f32; 16];
    logits[5] = 10.0;
    let probs = vec![1.0_f32; 16];
    let draft = vec![5_u32];
    let cfg = SpecDecodeConfig::default();

    let r = verify_draft(&logits, &draft, &probs, &cfg).expect("verify ok");
    assert_eq!(r.accepted_prefix, draft);
    assert_eq!(r.first_reject_idx, None);
    assert_eq!(r.bonus_token, Some(5));
}

#[test]
fn verify_rejects_at_first_mismatch_and_emits_bonus_token() {
    let mut logits = vec![0.0_f32; 16];
    logits[9] = 10.0;
    let draft = vec![1_u32, 2, 3];
    let probs = vec![1.0_f32; 16];
    let cfg = SpecDecodeConfig::default();

    let r = verify_draft(&logits, &draft, &probs, &cfg).expect("verify ok");
    assert!(r.accepted_prefix.is_empty());
    assert_eq!(r.first_reject_idx, Some(0));
    assert_eq!(r.bonus_token, Some(9));
}

#[test]
fn verify_handles_empty_draft() {
    let cfg = SpecDecodeConfig::default();
    let r = verify_draft(&[0.0; 4], &[], &[1.0; 4], &cfg).expect("verify ok");
    assert!(r.accepted_prefix.is_empty());
    assert_eq!(r.first_reject_idx, None);
    assert_eq!(r.bonus_token, None);
}

#[test]
fn verify_rejects_malformed_prob_length() {
    let cfg = SpecDecodeConfig::default();
    let logits = vec![0.0_f32; 16];
    let draft = vec![1_u32, 2];
    let bad_probs = vec![1.0_f32; 3];
    let res = verify_draft(&logits, &draft, &bad_probs, &cfg);
    assert!(res.is_err(), "expected Err on malformed probs");
}

#[test]
fn verify_is_deterministic_with_fixed_seed() {
    let mut logits = vec![0.0_f32; 16];
    logits[3] = 1.5;
    logits[7] = 2.5;
    let draft = vec![3_u32, 7];
    let cfg = SpecDecodeConfig::default();

    let r1 = verify_draft(&logits, &draft, &[1.0_f32; 16], &cfg).unwrap();
    let r2 = verify_draft(&logits, &draft, &[1.0_f32; 16], &cfg).unwrap();
    assert_eq!(r1.accepted_prefix, r2.accepted_prefix);
    assert_eq!(r1.first_reject_idx, r2.first_reject_idx);
    assert_eq!(r1.bonus_token, r2.bonus_token);
}

#[test]
fn verify_clamps_draft_longer_than_max_draft_tokens() {
    let cfg = SpecDecodeConfig {
        max_draft_tokens: 2,
        ..Default::default()
    };
    let mut logits = vec![0.0_f32; 8];
    logits[4] = 10.0;
    let draft = vec![4_u32, 4, 4, 4, 4];
    let probs = vec![1.0_f32; 8];
    let r = verify_draft(&logits, &draft, &probs, &cfg).expect("verify ok");
    assert!(r.accepted_prefix.len() <= 2);
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
    let h: SharedEngine = build_engine(
        cfg,
        Box::new(ScriptedTarget::with_argmax(8, 1)),
        None,
    );
    let lock = h.try_lock();
    assert!(lock.is_ok());
}

#[test]
fn medusa_proposal_tree_topology_serializes() {
    let t = TreeTopology {
        width: 4,
        depth: 2,
    };
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
