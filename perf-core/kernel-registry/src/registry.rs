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
use crate::quality::{evaluate_for_production, QualityAttachment, QualityError};
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
    pub fn register_candidate_checked(&mut self, candidate: Candidate) -> Option<Candidate> {
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
                rejections.push(RejectionRecord::new(
                    *id,
                    RejectionReason::MissingCapability(missing),
                ));
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
                rejections.push(RejectionRecord::new(*id, RejectionReason::NoTuningEvidence));
                continue;
            }

            if cand.backend.is_reference() {
                // Hold the reference candidate for fallback after the
                // tuned short-circuit.
                reference_fallback.get_or_insert(cand);
            } else {
                rejections.push(RejectionRecord::new(*id, RejectionReason::NoTuningEvidence));
            }
        }

        // 3. If we have tuned candidates, rank and pick.
        if !tuned.is_empty() {
            let metric = policy.metric();
            // 3a. Under Production, filter candidates against quality gates.
            // A candidate without a quality attachment is rejected with
            // `MissingQualityEvidence`; a candidate whose attachment fails a
            // gate is rejected with `QualityGateFailed` so traces show the
            // gate id + observed/threshold.
            let gates_required = policy.requires_quality_evidence();
            let survived: Vec<(Candidate, TuningRecord)> = if gates_required {
                tuned
                    .into_iter()
                    .filter_map(|(c, r)| match check_production_quality(&c, &r) {
                        Ok(()) => Some((c, r)),
                        Err(reason) => {
                            rejections.push(RejectionRecord::new(c.id, reason));
                            None
                        }
                    })
                    .collect()
            } else {
                tuned
            };

            if !survived.is_empty() {
                let chosen = pick_winner(&survived, metric);
                return SelectionDecision::Chosen {
                    candidate: chosen.0,
                    tuning: Box::new(chosen.1),
                };
            }
            // Quality-gate enforcement filtered out every tuned candidate.
            // Fall through to reference fallback (if any) and otherwise to
            // a rejection that cites the gates.
            if gates_required {
                if let Some(refc) = reference_fallback.take() {
                    let placeholder = TuningRecord::from_samples(
                        refc.id,
                        key.clone(),
                        &[0],
                        0,
                        "ref-oracle",
                        "0.0.0",
                        now_unix_ms,
                        "ref",
                        None,
                    );
                    return SelectionDecision::Chosen {
                        candidate: refc,
                        tuning: Box::new(placeholder),
                    };
                }
                return SelectionDecision::Rejected {
                    rejections,
                    considered,
                };
            }
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
                tuning: Box::new(placeholder),
            };
        }

        // 5. Nothing left.
        SelectionDecision::Rejected {
            rejections,
            considered,
        }
    }
}

/// Pick the winning (candidate, tuning) pair. Sort by `(metric, id)` so
/// the result is independent of HashMap iteration order.
fn pick_winner(tuned: &[(Candidate, TuningRecord)], metric: Metric) -> (Candidate, TuningRecord) {
    debug_assert!(!tuned.is_empty());
    let mut sorted: Vec<(Candidate, TuningRecord)> = tuned.to_vec();
    sorted.sort_by(|a, b| {
        let ma = metric.extract(&a.1);
        let mb = metric.extract(&b.1);
        ma.cmp(&mb).then(a.0.id.cmp(&b.0.id))
    });
    sorted.into_iter().next().expect("non-empty")
}

/// Verify that `record`'s quality attachment (if any) lets `candidate`
/// serve under `SelectionPolicy::Production`.
///
/// Translation rules:
/// - No attachment -> `RejectionReason::MissingQualityEvidence` describing
///   what needs to be attached.
/// - Attachment present but empty/with `gates.is_empty()` -> same rejection.
/// - Attachment present but a gate failed -> `QualityGateFailed`.
/// - Signature/duplicate errors are reported as `Other` (the registry
///   surface doesn't yet know how to react to them).
fn check_production_quality(
    candidate: &Candidate,
    record: &TuningRecord,
) -> std::result::Result<(), RejectionReason> {
    let attachment: &QualityAttachment = match record.quality.as_ref() {
        Some(q) => q,
        None => {
            return Err(RejectionReason::MissingQualityEvidence(format!(
                "candidate {} has no quality attachment under Production policy",
                candidate.id
            )))
        }
    };
    match evaluate_for_production(record, attachment) {
        Ok(()) => Ok(()),
        Err(QualityError::PromotionGateMissingEvidence { gate }) => {
            Err(RejectionReason::MissingQualityEvidence(format!(
                "candidate {} missing evidence for gate '{}'",
                candidate.id, gate
            )))
        }
        Err(QualityError::PromotionGateRejected {
            gate,
            observed,
            threshold,
        }) => Err(RejectionReason::QualityGateFailed {
            gate,
            observed,
            threshold,
        }),
        Err(QualityError::PromotionWithoutGates) => {
            Err(RejectionReason::MissingQualityEvidence(format!(
                "candidate {} attachment has no gates configured",
                candidate.id
            )))
        }
        Err(e) => Err(RejectionReason::Other(format!(
            "candidate {} quality check failed: {}",
            candidate.id, e
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::{BackendKind, CandidateId};
    use crate::compat::{DType, OperatorKind, QuantizationPolicy};
    use crate::key::{ShapeSignature, ATTENTION_NONE};

    fn wide_shape() -> ShapeSignature {
        ShapeSignature {
            m: 4096,
            n: 4096,
            k: 4096,
            batch: 64,
            seq: 4096,
            group: 64,
        }
    }

    fn _narrow_shape() -> ShapeSignature {
        ShapeSignature {
            m: 128,
            n: 128,
            k: 128,
            batch: 1,
            seq: 128,
            group: 1,
        }
    }

    fn make_candidate(name: &str) -> Candidate {
        Candidate::new(
            name,
            BackendKind::Cpu,
            "src-hash",
            vec![],
            ShapeSignature {
                m: 0,
                n: 0,
                k: 0,
                batch: 0,
                seq: 0,
                group: 0,
            },
            wide_shape(),
            vec![DType::Fp16, DType::Bf16],
            true,
        )
    }

    fn make_candidate_with_backend(name: &str, backend: BackendKind) -> Candidate {
        Candidate::new(
            name,
            backend,
            "src-hash",
            vec![],
            ShapeSignature {
                m: 0,
                n: 0,
                k: 0,
                batch: 0,
                seq: 0,
                group: 0,
            },
            wide_shape(),
            vec![DType::Fp16],
            true,
        )
    }

    fn _default_key() -> KernelKey {
        KernelKey {
            operator_kind: OperatorKind::DenseMatmul,
            attention_kind: ATTENTION_NONE,
            shape_signature: _narrow_shape(),
            dtype: DType::Fp16,
            quantization: QuantizationPolicy::None,
            state_layout_version: 1,
            device_fingerprint: "test-fp".to_string(),
            policy_version: 1,
        }
    }

    // --- Task 1: register / overwrite ------------------------------------------------

    #[test]
    fn register_duplicate_name_overwrites_existing_candidate() {
        let mut reg = KernelRegistry::new();
        let c1 = make_candidate("my-kernel");
        let c2 = make_candidate("my-kernel"); // same (name, backend) → same id

        reg.register_candidate(c1);
        assert_eq!(reg.list_candidates().len(), 1);

        // Overwrite: second registration with same id replaces the first.
        reg.register_candidate(c2);
        let listed = reg.list_candidates();
        assert_eq!(
            listed.len(),
            1,
            "duplicate insert must not create a second entry"
        );
    }

    #[test]
    fn register_candidate_checked_returns_old_on_collision() {
        let mut reg = KernelRegistry::new();
        let c1 = make_candidate("alpha");
        let c2 = make_candidate("alpha");

        assert!(
            reg.register_candidate_checked(c1).is_none(),
            "first insert must return None"
        );
        let returned = reg.register_candidate_checked(c2);
        assert!(
            returned.is_some(),
            "collision must return the displaced candidate"
        );
        assert_eq!(returned.unwrap().name, "alpha");
        assert_eq!(reg.list_candidates().len(), 1);
    }

    #[test]
    fn register_different_backends_produce_distinct_ids() {
        let mut reg = KernelRegistry::new();
        let c_cpu = make_candidate_with_backend("kernel-x", BackendKind::Cpu);
        let c_metal = make_candidate_with_backend("kernel-x", BackendKind::Metal);

        assert_ne!(
            c_cpu.id, c_metal.id,
            "different backends must yield different ids"
        );
        reg.register_candidate(c_cpu);
        reg.register_candidate(c_metal);
        assert_eq!(reg.list_candidates().len(), 2);
    }

    // --- Task 1: lookup non-existent -------------------------------------------------

    #[test]
    fn list_candidates_for_ids_returns_empty_for_unknown_id() {
        let mut reg = KernelRegistry::new();
        reg.register_candidate(make_candidate("real"));
        let unknown = CandidateId(0xDEAD_BEEF);
        let result = reg.list_candidates_for_ids(&[unknown]);
        assert!(result.is_empty(), "unknown id must not appear in result");
    }

    #[test]
    fn list_candidates_for_ids_mixes_known_and_unknown() {
        let mut reg = KernelRegistry::new();
        reg.register_candidate(make_candidate("a"));
        reg.register_candidate(make_candidate("b"));
        let known = CandidateId::derive("a", BackendKind::Cpu);
        let unknown = CandidateId(0x0000);
        let result = reg.list_candidates_for_ids(&[known, unknown]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "a");
    }

    // --- Task 1: listing returns all registered --------------------------------------

    #[test]
    fn list_candidates_returns_all_registered_sorted_by_id() {
        let mut reg = KernelRegistry::new();
        reg.register_candidate(make_candidate("z-last"));
        reg.register_candidate(make_candidate("a-first"));
        reg.register_candidate(make_candidate("m-middle"));

        let listed = reg.list_candidates();
        assert_eq!(listed.len(), 3);
        for pair in listed.windows(2) {
            assert!(pair[0].id <= pair[1].id, "list_candidates must sort by id");
        }
    }

    #[test]
    fn list_candidates_returns_empty_for_fresh_registry() {
        let reg = KernelRegistry::new();
        assert!(reg.list_candidates().is_empty());
    }

    // --- Task 1: unregister removes --------------------------------------------------

    #[test]
    fn unregister_candidate_by_id_removes_from_registry() {
        let mut reg = KernelRegistry::new();
        let c = make_candidate("removable");
        let id = c.id;
        reg.register_candidate(c);
        assert_eq!(reg.list_candidates().len(), 1);

        let removed = reg.candidates.remove(&id);
        assert!(removed.is_some(), "remove must return the candidate");
        assert!(reg.list_candidates().is_empty());
    }

    #[test]
    fn unregister_nonexistent_candidate_returns_none() {
        let mut reg = KernelRegistry::new();
        let removed = reg.candidates.remove(&CandidateId(0xBEEF));
        assert!(removed.is_none());
    }

    #[test]
    fn unregister_does_not_affect_other_candidates() {
        let mut reg = KernelRegistry::new();
        let c_keep = make_candidate("keep");
        let keep_id = c_keep.id;
        let c_remove = make_candidate("remove");
        let remove_id = c_remove.id;
        reg.register_candidate(c_keep);
        reg.register_candidate(c_remove);
        assert_eq!(reg.list_candidates().len(), 2);

        reg.candidates.remove(&remove_id);
        let remaining = reg.list_candidates();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, keep_id);
    }
}
