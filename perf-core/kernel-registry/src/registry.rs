//! In-memory [`KernelRegistry`] and the [`KernelRegistry::select_with_caps`]
//! entry point.
//!
//! Storage is two `HashMap`s:
//!
//! - `candidates: HashMap<CandidateId, Candidate>` — candidate metadata.
//! - `tuning: HashMap<KernelKey, Vec<TuningRecord>>` — fresh and stale
//!   evidence for each key.
//!
//! `select_with_caps` is the only public selector. It performs capability,
//! shape, dtype, freshness, and policy filtering deterministically and
//! returns either a [`SelectionDecision::Chosen`] (with the chosen
//! [`TuningRecord`]) or [`SelectionDecision::Rejected`] (with every
//! candidate id that was considered and the rejection reasons).
//!
//! ## Determinism
//!
//! All ordering happens on `Vec`s sorted by `(metric, candidate_id)`. The
//! selector never depends on HashMap iteration order, so two `select`
//! calls against the same registry state return identical decisions.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::candidate::{Candidate, CandidateId, Capability};
use crate::key::KernelKey;
use crate::record::TuningRecord;
use crate::selector::{
    Metric, RejectionReason, RejectionRecord, SelectionDecision, SelectionPolicy,
};
use crate::trace::ExecutionTrace;

/// Capability advertisement for the device that will execute kernels. The
/// selector requires `candidate.requires ⊆ caps.capabilities` to deem a
/// candidate eligible.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceCaps {
    pub capabilities: Vec<Capability>,
}

impl DeviceCaps {
    pub fn new(capabilities: Vec<Capability>) -> Self {
        Self { capabilities }
    }
}

/// In-memory registry of candidates and tuning evidence.
#[derive(Debug, Default)]
pub struct KernelRegistry {
    pub(crate) candidates: HashMap<CandidateId, Candidate>,
    pub(crate) tuning: HashMap<KernelKey, Vec<TuningRecord>>,
}

impl KernelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a candidate. If a candidate with the same id already
    /// exists it is overwritten — callers that need to detect collisions
    /// can use [`KernelRegistry::register_candidate_checked`].
    pub fn register_candidate(&mut self, candidate: Candidate) {
        self.candidates.insert(candidate.id, candidate);
    }

    /// Register a candidate only if no candidate with the same id exists
    /// yet. Returns the existing candidate on collision so callers can log
    /// provenance.
    pub fn register_candidate_checked(
        &mut self,
        candidate: Candidate,
    ) -> Option<Candidate> {
        self.candidates.insert(candidate.id, candidate)
    }

    /// Attach a tuning record for `key`. Records are appended in the order
    /// they arrive; the selector sorts them deterministically.
    pub fn attach_tuning_record(&mut self, key: KernelKey, record: TuningRecord) {
        self.tuning.entry(key).or_default().push(record);
    }

    /// All candidates, sorted by id for deterministic callers.
    pub fn list_candidates(&self) -> Vec<Candidate> {
        let mut v: Vec<Candidate> = self.candidates.values().cloned().collect();
        v.sort_by_key(|c| c.id);
        v
    }

    /// Candidates whose id is present in `keys`. The result is sorted by id
    /// ascending.
    pub fn list_candidates_for_ids(&self, keys: &[CandidateId]) -> Vec<Candidate> {
        let mut out: Vec<Candidate> = keys
            .iter()
            .filter_map(|id| self.candidates.get(id).cloned())
            .collect();
        out.sort_by_key(|c| c.id);
        out
    }

    /// Candidates whose `supports_shape(key.shape_signature)` is true. The
    /// result is sorted by id ascending.
    pub fn list_candidates_for_key(&self, key: &KernelKey) -> Vec<Candidate> {
        let mut v: Vec<Candidate> = self
            .candidates
            .values()
            .filter(|c| c.supports_shape(&key.shape_signature))
            .cloned()
            .collect();
        v.sort_by_key(|c| c.id);
        v
    }

    /// Tuning records attached to `key`, in insertion order. Callers that
    /// care about freshness must filter by `expires_at_unix_ms` themselves.
    pub fn tuning_records_for(&self, key: &KernelKey) -> Vec<TuningRecord> {
        self.tuning.get(key).cloned().unwrap_or_default()
    }

    /// Build an [`ExecutionTrace`] for a prior [`SelectionDecision`].
    ///
    /// The decision's rejections carry their own `candidate` ids, so the
    /// trace is constructed directly without re-running the selector.
    pub fn explain(&self, decision: &SelectionDecision) -> ExecutionTrace {
        ExecutionTrace::from_decision(decision)
    }

    /// The selector. Returns a [`SelectionDecision`] that either points at
    /// the chosen candidate + tuning record or at the rejection reasons.
    pub fn select_with_caps(
        &self,
        key: &KernelKey,
        policy: SelectionPolicy,
        caps: &DeviceCaps,
        now_unix_ms: u64,
    ) -> SelectionDecision {
        // 1. Collect every candidate id, sorted ascending. We use id order
        //    as the canonical iteration order so deterministic ties and
        //    considered-list order are reproducible.
        let mut all_ids: Vec<CandidateId> = self.candidates.keys().copied().collect();
        all_ids.sort();

        // 2. Filter and accumulate rejections.
        let mut rejections: Vec<RejectionRecord> = Vec::new();
        let mut considered: Vec<CandidateId> = Vec::new();
        let mut tuned: Vec<(Candidate, TuningRecord)> = Vec::new();
        let mut reference_fallback: Option<Candidate> = None;

        for id in &all_ids {
            let cand = match self.candidates.get(id) {
                Some(c) => c.clone(),
                None => continue,
            };
            considered.push(*id);

            // 2a. Capability filter.
            if !cand.capabilities_satisfied(caps) {
                // Record the missing capability with the highest-priority
                // missing name so traces are informative.
                let missing = cand
                    .requires
                    .iter()
                    .find(|req| !caps.capabilities.contains(req))
                    .map(|c| c.as_str().to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                rejections
                    .push(RejectionRecord::new(*id, RejectionReason::MissingCapability(missing)));
                continue;
            }

            // 2b. Dtype filter.
            if !cand.supports_dtype(key.dtype) {
                rejections.push(RejectionRecord::new(
                    *id,
                    RejectionReason::UnsupportedDtype(format!("{:?}", key.dtype)),
                ));
                continue;
            }

            // 2c. Shape filter.
            if !cand.supports_shape(&key.shape_signature) {
                rejections.push(RejectionRecord::new(*id, RejectionReason::ShapeOutOfRange));
                continue;
            }

            // 2d. Backend filter for ExperimentalOnly.
            if matches!(policy, SelectionPolicy::ExperimentalOnly) && !cand.tunable {
                rejections.push(RejectionRecord::new(
                    *id,
                    RejectionReason::PolicyExcluded(
                        "experimental policy requires tunable candidate".to_string(),
                    ),
                ));
                continue;
            }

            // 2e. Tuning evidence.
            let rec = self.tuning.get(key).and_then(|records| {
                records
                    .iter()
                    .find(|r| r.candidate_id == *id && !r.is_stale(now_unix_ms))
                    .cloned()
            });

            if let Some(r) = rec {
                tuned.push((cand, r));
                continue;
            }

            // Stale-only evidence: record a StaleTuning rejection so
            // traces explain why the candidate was rejected.
            let stale = self.tuning.get(key).and_then(|records| {
                records
                    .iter()
                    .find(|r| r.candidate_id == *id && r.is_stale(now_unix_ms))
                    .cloned()
            });
            if let Some(s) = stale {
                let expires = s.expires_at_unix_ms.unwrap_or(0);
                rejections.push(RejectionRecord::new(
                    *id,
                    RejectionReason::StaleTuning {
                        expires_at_unix_ms: expires,
                        now_unix_ms,
                    },
                ));
                continue;
            }

            // No evidence at all.
            if matches!(policy, SelectionPolicy::ExperimentalOnly) {
                rejections
                    .push(RejectionRecord::new(*id, RejectionReason::NoTuningEvidence));
                continue;
            }

            if cand.backend.is_reference() {
                // Hold the reference candidate for fallback after the
                // tuned short-circuit.
                reference_fallback.get_or_insert(cand);
            } else {
                rejections
                    .push(RejectionRecord::new(*id, RejectionReason::NoTuningEvidence));
            }
        }

        // 3. If we have tuned candidates, rank and pick.
        if !tuned.is_empty() {
            let metric = policy.metric();
            let chosen = pick_winner(&tuned, metric);
            return SelectionDecision::Chosen {
                candidate: chosen.0,
                tuning: chosen.1,
            };
        }

        // 4. Reference fallback.
        if let Some(refc) = reference_fallback {
            // Synthesize a "no-op" tuning record so callers can rely on
            // the Chosen variant always carrying a record.
            let placeholder = TuningRecord::from_samples(
                refc.id,
                key.clone(),
                &[0],
                /*warmup_discarded*/ 0,
                "ref-oracle",
                "0.0.0",
                now_unix_ms,
                "ref",
                None,
            );
            return SelectionDecision::Chosen {
                candidate: refc,
                tuning: placeholder,
            };
        }

        // 5. Nothing left.
        SelectionDecision::Rejected { rejections, considered }
    }
}

/// Pick the winning (candidate, tuning) pair. Sort by `(metric, id)` so
/// the result is independent of HashMap iteration order.
fn pick_winner(
    tuned: &[(Candidate, TuningRecord)],
    metric: Metric,
) -> (Candidate, TuningRecord) {
    debug_assert!(!tuned.is_empty());
    let mut sorted: Vec<(Candidate, TuningRecord)> = tuned.to_vec();
    sorted.sort_by(|a, b| {
        let ma = metric.extract(&a.1);
        let mb = metric.extract(&b.1);
        ma.cmp(&mb).then(a.0.id.cmp(&b.0.id))
    });
    sorted.into_iter().next().expect("non-empty")
}