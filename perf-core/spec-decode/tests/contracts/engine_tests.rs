use super::*;

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

// ---------------------------------------------------------------------------
// Edge-case tests — history capacity boundary
// ---------------------------------------------------------------------------

#[test]
fn engine_state_push_accepted_evicts_oldest_at_exact_capacity_boundary() {
    let mut s = EngineState::new();
    for i in 0..HISTORY_CAP as u32 {
        s.push_accepted(i);
    }
    assert_eq!(s.history.len(), HISTORY_CAP);
    assert_eq!(s.history.front(), Some(&0));
    assert_eq!(s.history.back(), Some(&(HISTORY_CAP as u32 - 1)));

    s.push_accepted(9999);
    assert_eq!(s.history.len(), HISTORY_CAP);
    assert_eq!(s.history.front(), Some(&1), "oldest entry must be evicted");
    assert_eq!(s.history.back(), Some(&9999));
}

#[test]
fn engine_state_push_accepted_evicts_correctly_over_multiple_overflows() {
    let mut s = EngineState::new();
    for i in 0..(HISTORY_CAP + 50) as u32 {
        s.push_accepted(i);
    }
    assert_eq!(s.history.len(), HISTORY_CAP);
    assert_eq!(
        s.history.front(),
        Some(&50),
        "first 50 entries must have been evicted"
    );
    assert_eq!(s.history.back(), Some(&((HISTORY_CAP + 49) as u32)));
}

// ---------------------------------------------------------------------------
// push_accepted edge cases — capacity boundary
// ---------------------------------------------------------------------------

#[test]
fn push_accepted_capacity_one_evicts_old_entry() {
    let mut s = EngineState::new();
    for i in 0..HISTORY_CAP as u32 {
        s.push_accepted(i);
    }
    assert_eq!(s.history.len(), HISTORY_CAP);

    s.push_accepted(9999);
    assert_eq!(
        s.history.len(),
        HISTORY_CAP,
        "queue must stay at cap after eviction"
    );
    assert_eq!(s.history[0], 1, "oldest entry (0) must be evicted");
    assert_eq!(
        s.history[HISTORY_CAP - 1],
        9999,
        "newest entry must be at back"
    );
}

#[test]
fn push_accepted_capacity_zero_does_not_crash() {
    let mut s = EngineState::new();
    for i in 0..HISTORY_CAP as u32 {
        s.push_accepted(i);
    }
    s.push_accepted(0);
    assert_eq!(s.history.len(), HISTORY_CAP);
}
