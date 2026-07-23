//! Per-round proposal state — the model-owned execution state threaded
//! through each proposal/verify round.
//!
//! This is the contract a DeepSeek-MTP-style or EAGLE-style speculative
//! decoder depends on. The accumulator holds the current round's draft
//! bytes, the rolling `verified_prefix` of accepted tokens across rounds,
//! the per-round acceptance history, and the optional bonus token
//! recorded after a fully-accepted proposal.
//!
//! The kernel-registry coverage pin (turn-9 commit `a0fba0f`) originated
//! this state in a private test shim; this module promotes it to a real
//! crate-level type so production callers can depend on the same contract.

use serde::{Deserialize, Serialize};

/// Model-owned speculative-decoding proposal state.
///
/// `draft_tokens` holds the most recent round's proposed draft tokens
/// (length `num_drafts`). `acceptance_count` is the running total of
/// accepted drafts across rounds (drafts only — bonuses are tracked
/// separately). `verified_prefix` is the concatenation of every
/// accepted draft token, in order, across rounds.
/// `acceptance_history` records the per-round accepted count
/// (`0..=num_drafts`) for diagnostics. `bonus_token` is `Some(t)` iff a
/// verifier-awarded bonus token has been recorded; it does not
/// contribute to `acceptance_count` but does extend the effective
/// sequence length.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Construct a fresh proposal state. `draft_tokens` is zero-initialized
    /// (not random) so the contract is reproducible: a fresh state must
    /// never carry a residual draft from a prior round. `seed` is recorded
    /// verbatim for diagnostics but does not influence the initial draft
    /// bytes.
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

    /// Accept the first `n` of the current `draft_tokens` as verified.
    /// `n` must satisfy `0 <= n <= num_drafts`. The accepted slice is
    /// appended verbatim to `verified_prefix` and the per-round count is
    /// pushed to `acceptance_history`.
    pub fn accept_drafts(&mut self, n: usize) {
        assert!(
            n <= self.num_drafts,
            "accept n={n} exceeds num_drafts={}",
            self.num_drafts
        );
        // Append the verified prefix slice — verbatim from
        // `draft_tokens[..n]` so the bytes are byte-identical to the
        // proposed draft.
        self.verified_prefix
            .extend_from_slice(&self.draft_tokens[..n]);
        self.acceptance_count += n;
        self.acceptance_history.push(n);
    }

    /// Reject the current draft round without recording any acceptance.
    /// The proposal state is preserved verbatim so the caller can
    /// re-attempt the round with new draft bytes.
    pub fn reject_drafts(&mut self) {
        // No-op on `verified_prefix`, `acceptance_count`,
        // `draft_tokens`. The contract is "preserves state":
        // `acceptance_history` records the rejection as a 0-length
        // round for diagnostics.
        self.acceptance_history.push(0);
    }

    /// Record a verifier-awarded bonus token. The bonus is stored
    /// verbatim in `bonus_token` and increments `acceptance_count` by
    /// one. The bonus does NOT extend `verified_prefix` — the
    /// DeepSeek-MTP contract treats it as a model-side sample consumed
    /// by the next round, not a verified token.
    pub fn append_bonus(&mut self, token: u32) {
        self.bonus_token = Some(token);
        self.acceptance_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NUM_DRAFTS: usize = 5;
    const VOCAB_SIZE: usize = 100;
    const SEED: u64 = 42;

    fn state_with_drafts(drafts: &[u32]) -> ProposalState {
        let mut s = ProposalState::new(NUM_DRAFTS, VOCAB_SIZE, SEED);
        assert_eq!(
            drafts.len(),
            NUM_DRAFTS,
            "fixture must supply exactly num_drafts tokens"
        );
        for (i, &t) in drafts.iter().enumerate() {
            assert!(
                (t as usize) < VOCAB_SIZE,
                "draft token {t} at slot {i} is out of vocab range"
            );
        }
        s.draft_tokens.copy_from_slice(drafts);
        s
    }

    #[test]
    fn fresh_state_zero_initializes_all_fields() {
        let s = ProposalState::new(NUM_DRAFTS, VOCAB_SIZE, SEED);

        assert_eq!(s.acceptance_count, 0);
        assert_eq!(s.bonus_token, None);
        assert!(s.acceptance_history.is_empty());

        assert_eq!(s.draft_tokens, vec![0u32; NUM_DRAFTS]);
        assert_eq!(s.draft_tokens.len(), NUM_DRAFTS);

        assert!(s.verified_prefix.is_empty());

        assert_eq!(s.num_drafts, NUM_DRAFTS);
        assert_eq!(s.vocab_size, VOCAB_SIZE);
        assert_eq!(s.seed, SEED);
    }

    #[test]
    fn accept_drafts_copies_prefix_verbatim() {
        let drafts: [u32; NUM_DRAFTS] = [11, 22, 33, 44, 55];
        let mut s = state_with_drafts(&drafts);

        s.accept_drafts(3);

        assert_eq!(s.acceptance_count, 3);
        assert_eq!(s.acceptance_history, vec![3usize]);
        assert_eq!(s.verified_prefix, drafts[..3].to_vec());
        assert!(!s.verified_prefix.contains(&44));
        assert!(!s.verified_prefix.contains(&55));
        assert_eq!(s.bonus_token, None);
    }

    #[test]
    fn reject_drafts_preserves_state_for_re_attempt() {
        let drafts: [u32; NUM_DRAFTS] = [7, 14, 21, 28, 35];
        let mut s = state_with_drafts(&drafts);

        s.reject_drafts();

        assert_eq!(s.acceptance_count, 0);
        assert!(s.verified_prefix.is_empty());
        assert_eq!(s.draft_tokens, drafts.to_vec());
        assert_eq!(s.acceptance_history, vec![0usize]);

        // Re-attempt contract.
        s.accept_drafts(NUM_DRAFTS);
        assert_eq!(s.acceptance_count, NUM_DRAFTS);
        assert_eq!(s.verified_prefix, drafts.to_vec());
        assert_eq!(s.acceptance_history, vec![0usize, NUM_DRAFTS]);
    }

    #[test]
    fn full_accept_plus_bonus_increments_count_and_records_bonus() {
        let drafts: [u32; NUM_DRAFTS] = [2, 4, 6, 8, 10];
        let mut s = state_with_drafts(&drafts);

        s.accept_drafts(NUM_DRAFTS);
        assert_eq!(s.acceptance_count, NUM_DRAFTS);

        let bonus: u32 = 99;
        s.append_bonus(bonus);

        assert_eq!(s.acceptance_count, NUM_DRAFTS + 1);
        assert_eq!(s.bonus_token, Some(bonus));
        // The bonus is NOT appended to verified_prefix — it is a
        // model-side sample consumed by the next round.
        assert_eq!(s.verified_prefix, drafts.to_vec());
        assert_eq!(s.verified_prefix.len(), NUM_DRAFTS);
        // `acceptance_history` records only the accept, not the bonus.
        assert_eq!(s.acceptance_history, vec![NUM_DRAFTS]);
    }

    #[test]
    fn panics_when_accept_n_exceeds_num_drafts() {
        let mut s = ProposalState::new(NUM_DRAFTS, VOCAB_SIZE, SEED);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            s.accept_drafts(NUM_DRAFTS + 1);
        }));
        assert!(result.is_err(), "accepting n > num_drafts must panic");
    }

    #[test]
    fn panics_on_zero_num_drafts() {
        let result = std::panic::catch_unwind(|| ProposalState::new(0, VOCAB_SIZE, SEED));
        assert!(result.is_err(), "num_drafts=0 must panic");
    }

    #[test]
    fn panics_on_zero_vocab_size() {
        let result = std::panic::catch_unwind(|| ProposalState::new(NUM_DRAFTS, 0, SEED));
        assert!(result.is_err(), "vocab_size=0 must panic");
    }

    #[test]
    fn serde_json_round_trip_preserves_state() {
        let drafts: [u32; NUM_DRAFTS] = [3, 6, 9, 12, 15];
        let mut s = state_with_drafts(&drafts);
        s.accept_drafts(2);
        s.append_bonus(77);

        let json = serde_json::to_string(&s).expect("serialize");
        let back: ProposalState = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(s, back);
    }
}
