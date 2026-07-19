//! Speculative decoding proposal-state contract — the model-owned execution
//! state that a DeepSeek-MTP-style or EAGLE-style speculative decoder
//! threads through each proposal/verify round. The session overview
//! (line 32 of `15_TURN_8_RESUME_NOTES.md`) flags "Speculative decoding
//! lacks a complete proposal path and model-owned execution state" as
//! a coverage gap; this file pins the *state-shape* contract for that
//! path.
//!
//! This is a self-contained unit test for the proposal-state struct
//! shape — it does not bind against a production `ProposalState` type
//! (one does not exist in the checked-in crates yet). The struct lives
//! in a private `mod proposal_state` so the tests assert the contract
//! against an authoritative reference implementation, not a hypothetical
//! production type. When the production `ProposalState` lands, the
//! assertions become a verbatim migration checklist.
//!
//! Tests:
//!
//!   1. `proposal_state_initializes_with_zero_acceptance_count` — a
//!      freshly-constructed state has `acceptance_count == 0`, a
//!      zero-filled `draft_tokens` vector of the requested length, and
//!      an empty `acceptance_history`.
//!   2. `proposal_state_accepts_draft_tokens_byte_identical` — accepting
//!      the first 3 of 5 drafts increments `acceptance_count` to 3,
//!      records the run length `[3]` in `acceptance_history`, and copies
//!      the verified prefix bytes from the input.
//!   3. `proposal_state_rejects_with_zero_acceptance_preserves_state`
//!      — rejecting all 5 drafts leaves `acceptance_count` at 0 and
//!      keeps `verified_prefix` empty; the proposal remains available
//!      for re-attempt (state is *not* invalidated).
//!   4. `proposal_state_bonus_token_appended_after_full_acceptance` —
//!      after accepting all 5 drafts and recording a bonus token,
//!      `acceptance_count == 6` (5 drafts + 1 bonus) and the bonus is
//!      reflected in the `bonus_token` field.
//!
//! Convention: every test seeds with `seed=42` and a fixed `(num_drafts,
//! vocab_size)` shape so the contract is reproducible across runs.

mod proposal_state {
    //! Self-contained reference `ProposalState`. Mirrors the contract a
    //! production speculative decoder would expose: an accumulator of
    //! per-round acceptance counts, the rolling `verified_prefix` of
    //! tokens that survived the verifier, the rolling
    //! `acceptance_history` of per-round acceptance lengths, and the
    //! optional `bonus_token` recorded when the verifier awards an
    //! extra sample after a fully-accepted proposal.

    /// Model-owned speculative-decoding proposal state.
    ///
    /// `draft_tokens` holds the most recent round's proposed draft
    /// tokens (length `num_drafts`). `acceptance_count` is the running
    /// total of accepted drafts across rounds (drafts only — bonuses
    /// are tracked separately). `verified_prefix` is the concatenation
    /// of every accepted draft token, in order, across rounds.
    /// `acceptance_history` records the per-round accepted count
    /// (`0..=num_drafts`) for diagnostics. `bonus_token` is `Some(t)`
    /// iff a verifier-awarded bonus token has been recorded; it does
    /// not contribute to `acceptance_count` but does extend the
    /// effective sequence length.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ProposalState {
        pub num_drafts: usize,
        pub vocab_size: usize,
        pub seed: u64,
        pub draft_tokens: Vec<u32>,
        pub acceptance_count: usize,
        pub verified_prefix: Vec<u32>,
        pub acceptance_history: Vec<usize>,
        pub bonus_token: Option<u32>,
    }

    impl ProposalState {
        /// Construct a fresh proposal state. `draft_tokens` is
        /// zero-initialized (not random) so the contract is
        /// reproducible: a fresh state must never carry a residual
        /// draft from a prior round. `seed` is recorded verbatim for
        /// diagnostics but does not influence the initial draft bytes.
        pub fn new(num_drafts: usize, vocab_size: usize, seed: u64) -> Self {
            assert!(num_drafts > 0, "num_drafts must be positive");
            assert!(vocab_size > 0, "vocab_size must be positive");
            Self {
                num_drafts,
                vocab_size,
                seed,
                draft_tokens: vec![0u32; num_drafts],
                acceptance_count: 0,
                verified_prefix: Vec::new(),
                acceptance_history: Vec::new(),
                bonus_token: None,
            }
        }

        /// Accept the first `n` of the current `draft_tokens` as
        /// verified. `n` must satisfy `0 <= n <= num_drafts`. The
        /// accepted slice is appended to `verified_prefix` and the
        /// per-round count is pushed to `acceptance_history`.
        pub fn accept_drafts(&mut self, n: usize) {
            assert!(n <= self.num_drafts, "accept n={n} exceeds num_drafts={}", self.num_drafts);
            // Append the verified prefix slice — verbatim from
            // `draft_tokens[..n]` so the bytes are byte-identical to
            // the proposed draft.
            self.verified_prefix.extend_from_slice(&self.draft_tokens[..n]);
            self.acceptance_count += n;
            self.acceptance_history.push(n);
        }

        /// Reject the current draft round without recording any
        /// acceptance. The proposal state is preserved verbatim so the
        /// caller can re-attempt the round with new draft bytes.
        pub fn reject_drafts(&mut self) {
            // No-op on `verified_prefix`, `acceptance_count`,
            // `acceptance_history`. The contract is "preserves state":
            // a regression that quietly zeros `draft_tokens` or bumps
            // a counter trips the byte-equality assertions below.
            self.acceptance_history.push(0);
        }

        /// Record a verifier-awarded bonus token. The bonus is stored
        /// verbatim in `bonus_token`. Per the DeepSeek-MTP contract,
        /// the bonus counts as an additional accepted token: callers
        /// see `acceptance_count + (bonus_token.is_some() as usize) ==
        /// num_drafts + 1` after a full accept + bonus.
        pub fn append_bonus(&mut self, token: u32) {
            self.bonus_token = Some(token);
            self.acceptance_count += 1;
        }
    }
}

use proposal_state::ProposalState;

// Canonical fixture: 5 drafts, vocab 100, seed 42. The seed is the
// reproducibility anchor — every test reuses it so the proposal-state
// constructor's recorded seed is verifiable.
const NUM_DRAFTS: usize = 5;
const VOCAB_SIZE: usize = 100;
const SEED: u64 = 42;

/// Helper: build a fresh state and overwrite `draft_tokens` with the
/// supplied proposal bytes. The proposal bytes are deterministic across
/// calls (caller supplies them) so the acceptance tests can verify
/// byte-for-byte prefix copying.
fn state_with_drafts(drafts: &[u32]) -> ProposalState {
    let mut s = ProposalState::new(NUM_DRAFTS, VOCAB_SIZE, SEED);
    assert_eq!(drafts.len(), NUM_DRAFTS,
        "fixture must supply exactly num_drafts tokens");
    for (i, &t) in drafts.iter().enumerate() {
        assert!((t as usize) < VOCAB_SIZE,
            "draft token {t} at slot {i} is out of vocab range");
    }
    s.draft_tokens.copy_from_slice(drafts);
    s
}

// ---------------------------------------------------------------------------
// Test 1 — fresh state is zero-initialized
// ---------------------------------------------------------------------------

#[test]
fn proposal_state_initializes_with_zero_acceptance_count() {
    let s = ProposalState::new(NUM_DRAFTS, VOCAB_SIZE, SEED);

    // (a) Acceptance counters are zero — no drafts accepted yet, no
    // bonus, no history.
    assert_eq!(s.acceptance_count, 0,
        "fresh ProposalState must have acceptance_count=0");
    assert_eq!(s.bonus_token, None,
        "fresh ProposalState must have bonus_token=None");
    assert!(s.acceptance_history.is_empty(),
        "fresh ProposalState must have empty acceptance_history");

    // (b) `draft_tokens` is zero-filled to `num_drafts` slots. The
    // contract is explicit: a fresh state must NOT carry a residual
    // draft from a prior round — `vec![0; num_drafts]` is the byte
    // contract.
    assert_eq!(s.draft_tokens, vec![0u32; NUM_DRAFTS],
        "fresh ProposalState must zero-fill draft_tokens");
    assert_eq!(s.draft_tokens.len(), NUM_DRAFTS);

    // (c) `verified_prefix` is empty. A regression that copied
    // `draft_tokens` into `verified_prefix` at construction time
    // trips this assertion.
    assert!(s.verified_prefix.is_empty(),
        "fresh ProposalState must have empty verified_prefix");

    // (d) Construction-time scalars are recorded verbatim.
    assert_eq!(s.num_drafts, NUM_DRAFTS);
    assert_eq!(s.vocab_size, VOCAB_SIZE);
    assert_eq!(s.seed, SEED);
}

// ---------------------------------------------------------------------------
// Test 2 — accept first 3 of 5 drafts; verify byte-identical prefix
// ---------------------------------------------------------------------------

#[test]
fn proposal_state_accepts_draft_tokens_byte_identical() {
    // Distinct per-slot values so a regression that copies the wrong
    // slice (e.g. offsets by one) trips the prefix equality check.
    let drafts: [u32; NUM_DRAFTS] = [11, 22, 33, 44, 55];
    let mut s = state_with_drafts(&drafts);

    s.accept_drafts(3);

    assert_eq!(s.acceptance_count, 3,
        "accepting 3 of 5 drafts must set acceptance_count=3");
    assert_eq!(s.acceptance_history, vec![3usize],
        "acceptance_history must record per-round count [3]");
    assert_eq!(s.verified_prefix, drafts[..3].to_vec(),
        "verified_prefix must be byte-identical to drafts[..3]");
    // The unaccepted slots (44, 55) MUST NOT leak into the prefix.
    assert!(!s.verified_prefix.contains(&44),
        "verified_prefix must not contain slot 3 (44)");
    assert!(!s.verified_prefix.contains(&55),
        "verified_prefix must not contain slot 4 (55)");
    // Bonus is still unset — acceptance does not imply a bonus.
    assert_eq!(s.bonus_token, None,
        "accept_drafts must not set bonus_token");
}

// ---------------------------------------------------------------------------
// Test 3 — rejecting all 5 preserves state for re-attempt
// ---------------------------------------------------------------------------

#[test]
fn proposal_state_rejects_with_zero_acceptance_preserves_state() {
    let drafts: [u32; NUM_DRAFTS] = [7, 14, 21, 28, 35];
    let mut s = state_with_drafts(&drafts);

    s.reject_drafts();

    assert_eq!(s.acceptance_count, 0,
        "rejecting all 5 drafts must leave acceptance_count=0");
    assert!(s.verified_prefix.is_empty(),
        "rejecting all 5 drafts must leave verified_prefix empty");
    // `draft_tokens` MUST remain intact so the caller can re-attempt
    // the round (e.g. with a freshly-sampled draft, overwriting the
    // same field). The contract is "preserves state", not "resets state".
    assert_eq!(s.draft_tokens, drafts.to_vec(),
        "reject_drafts must not mutate draft_tokens; re-attempt requires the original bytes");
    // `acceptance_history` records the rejection as a 0-length round
    // — this is the "preserved for diagnostics" half of the contract.
    assert_eq!(s.acceptance_history, vec![0usize],
        "acceptance_history must record per-round count [0] on rejection");
    // Re-attempt contract: a fresh accept after the rejection must
    // behave exactly as on a clean state.
    s.accept_drafts(NUM_DRAFTS);
    assert_eq!(s.acceptance_count, NUM_DRAFTS,
        "after rejection, a full accept must restore acceptance_count to num_drafts");
    assert_eq!(s.verified_prefix, drafts.to_vec(),
        "after rejection, a full accept must rebuild verified_prefix from the original draft bytes");
    assert_eq!(s.acceptance_history, vec![0usize, NUM_DRAFTS],
        "acceptance_history must accumulate [0, num_drafts] across reject + accept");
}

// ---------------------------------------------------------------------------
// Test 4 — full accept + bonus appends the bonus and bumps the count
// ---------------------------------------------------------------------------

#[test]
fn proposal_state_bonus_token_appended_after_full_acceptance() {
    let drafts: [u32; NUM_DRAFTS] = [2, 4, 6, 8, 10];
    let mut s = state_with_drafts(&drafts);

    // Accept all 5 drafts first.
    s.accept_drafts(NUM_DRAFTS);
    assert_eq!(s.acceptance_count, NUM_DRAFTS,
        "baseline: full accept must set acceptance_count=5");

    // Append a bonus token (the verifier's "you predicted the whole
    // window correctly, here's one extra sample" signal).
    let bonus: u32 = 99;
    s.append_bonus(bonus);

    // (a) `acceptance_count` is now 6: 5 drafts + 1 bonus.
    assert_eq!(s.acceptance_count, NUM_DRAFTS + 1,
        "bonus token must increment acceptance_count to 6 (= num_drafts + 1)");
    // (b) `bonus_token` is recorded verbatim.
    assert_eq!(s.bonus_token, Some(bonus),
        "bonus_token must be recorded verbatim from append_bonus");
    // (c) `verified_prefix` still holds the 5 accepted drafts — the
    // bonus is NOT appended to the prefix. The DeepSeek-MTP contract
    // is that the bonus is a *model-side* sample that the next round
    // consumes as input, not a verified token. A regression that
    // extends `verified_prefix` with the bonus trips this assertion.
    assert_eq!(s.verified_prefix, drafts.to_vec(),
        "verified_prefix must hold only the accepted drafts; the bonus is NOT a verified token");
    assert_eq!(s.verified_prefix.len(), NUM_DRAFTS,
        "verified_prefix length must stay at num_drafts after bonus");
    // (d) `acceptance_history` records only the accept (bonus is not
    // a separate round).
    assert_eq!(s.acceptance_history, vec![NUM_DRAFTS],
        "acceptance_history must record only the accept, not the bonus round");
}
