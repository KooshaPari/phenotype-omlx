use crate::candidate::{Candidate, CandidateId};
use crate::key::KernelKey;
use crate::record::TuningRecord;
use crate::selector::{
    Metric, RejectionReason, RejectionRecord, SelectionDecision, SelectionPolicy,
};

use super::registry_validate::check_production_quality;
use super::{DeviceCaps, KernelRegistry};

impl KernelRegistry {
    /// The selector. Returns a [`SelectionDecision`] that either points at
    /// the chosen candidate + tuning record or at the rejection reasons.
    pub fn select_with_caps(
        &self,
        key: &KernelKey,
        policy: SelectionPolicy,
        caps: &DeviceCaps,
        now_unix_ms: u64,
    ) -> SelectionDecision {
        // 1. Collect every candidate, sorted by (id, index) for deterministic
        //    iteration order across multi-candidate ids.
        let mut all_candidates: Vec<(CandidateId, Candidate)> = self
            .candidates
            .iter()
            .flat_map(|(id, vec)| vec.iter().cloned().map(move |c| (*id, c)))
            .collect();
        all_candidates.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.id.cmp(&b.1.id)));

        // 2. Filter and accumulate rejections.
        let mut rejections: Vec<RejectionRecord> = Vec::new();
        let mut considered: Vec<CandidateId> = Vec::new();
        let mut tuned: Vec<(Candidate, TuningRecord)> = Vec::new();
        let mut reference_fallback: Option<Candidate> = None;

        for (id, cand) in &all_candidates {
            considered.push(*id);

            // 2a. Capability filter.
            if !cand.capabilities_satisfied(caps) {
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
                tuned.push((cand.clone(), r));
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
                reference_fallback.get_or_insert(cand.clone());
            } else {
                rejections.push(RejectionRecord::new(*id, RejectionReason::NoTuningEvidence));
            }
        }

        // 3. If we have tuned candidates, rank and pick.
        if !tuned.is_empty() {
            let metric = policy.metric();
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

    /// Select the best candidate for `task` considering KV cache state.
    ///
    /// When KV utilization (`kv_cache_size / kv_max_size`) exceeds 80 %,
    /// candidates that advertise `dynamic_eviction = "true"` in their
    /// [`properties`](Candidate::properties) map are preferred because
    /// they integrate with [`turbo_quant::echokv::EchoKVCache`] for
    /// on-the-fly eviction. For lower utilization the first (default)
    /// candidate is returned.
    ///
    /// Returns `None` when `task` has no registered candidates.
    pub fn select_with_kv_state(
        &self,
        task: &CandidateId,
        kv_cache_size: usize,
        kv_max_size: usize,
    ) -> Option<&Candidate> {
        let candidates = self.candidates.get(task)?;

        if kv_max_size == 0 {
            return candidates.first();
        }

        let utilization = kv_cache_size as f32 / kv_max_size as f32;
        if utilization > 0.8 {
            candidates
                .iter()
                .find(|c| c.has_property("dynamic_eviction"))
                .or_else(|| candidates.first())
        } else {
            candidates.first()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::{BackendKind, CandidateId};
    use crate::compat::DType;
    use crate::key::ShapeSignature;
    use crate::registry::KernelRegistry;

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

    fn make_candidate_with_property(name: &str, prop_key: &str, prop_val: &str) -> Candidate {
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
            vec![DType::Fp16],
            true,
        )
        .with_property(prop_key, prop_val)
    }

    #[test]
    fn select_with_kv_state_returns_evictable_when_utilization_high() {
        let mut reg = KernelRegistry::new();

        let shared_id = CandidateId::derive("multi-cand", BackendKind::Cpu);
        reg.candidates.insert(
            shared_id,
            vec![
                make_candidate("standard"),
                make_candidate_with_property("echokv", "dynamic_eviction", "true"),
            ],
        );

        let chosen = reg.select_with_kv_state(&shared_id, 90, 100);
        assert!(chosen.is_some(), "must return a candidate");
        assert!(
            chosen.unwrap().has_property("dynamic_eviction"),
            "must prefer evictable candidate at high utilization"
        );
    }

    #[test]
    fn select_with_kv_state_returns_first_when_utilization_low() {
        let mut reg = KernelRegistry::new();
        let shared_id = CandidateId::derive("low-util", BackendKind::Cpu);
        reg.candidates.insert(
            shared_id,
            vec![
                make_candidate("first"),
                make_candidate_with_property("echokv", "dynamic_eviction", "true"),
            ],
        );

        let chosen = reg.select_with_kv_state(&shared_id, 50, 100);
        assert!(chosen.is_some(), "must return a candidate");
        assert_eq!(
            chosen.unwrap().name,
            "first",
            "must return first candidate at low utilization"
        );
    }

    #[test]
    fn select_with_kv_state_returns_none_for_unknown_task() {
        let reg = KernelRegistry::new();
        let unknown = CandidateId(0xDEAD_BEEF);
        assert!(
            reg.select_with_kv_state(&unknown, 50, 100).is_none(),
            "unknown task must return None"
        );
    }

    #[test]
    fn select_with_kv_state_handles_zero_max_size() {
        let mut reg = KernelRegistry::new();
        let id = CandidateId::derive("zero-max", BackendKind::Cpu);
        reg.candidates.insert(id, vec![make_candidate("only-one")]);

        let chosen = reg.select_with_kv_state(&id, 0, 0);
        assert!(chosen.is_some(), "zero max_size must not panic");
    }

    #[test]
    fn select_with_kv_state_falls_back_when_no_evictable_candidate() {
        let mut reg = KernelRegistry::new();
        let id = CandidateId::derive("no-evict", BackendKind::Cpu);
        reg.candidates
            .insert(id, vec![make_candidate("no-eviction-prop")]);

        let chosen = reg.select_with_kv_state(&id, 95, 100);
        assert!(chosen.is_some());
        assert_eq!(chosen.unwrap().name, "no-eviction-prop");
    }

    #[test]
    fn select_with_kv_state_boundary_exactly_80_percent() {
        let mut reg = KernelRegistry::new();
        let id = CandidateId::derive("boundary", BackendKind::Cpu);
        reg.candidates.insert(
            id,
            vec![
                make_candidate("first"),
                make_candidate_with_property("echokv", "dynamic_eviction", "true"),
            ],
        );

        let chosen = reg.select_with_kv_state(&id, 80, 100);
        assert!(chosen.is_some());
        assert_eq!(chosen.unwrap().name, "first");
    }

    #[test]
    fn select_with_kv_state_boundary_just_over_80_percent() {
        let mut reg = KernelRegistry::new();
        let id = CandidateId::derive("boundary-over", BackendKind::Cpu);
        reg.candidates.insert(
            id,
            vec![
                make_candidate("first"),
                make_candidate_with_property("echokv", "dynamic_eviction", "true"),
            ],
        );

        let chosen = reg.select_with_kv_state(&id, 81, 100);
        assert!(chosen.is_some());
        assert!(
            chosen.unwrap().has_property("dynamic_eviction"),
            "just over 80% must prefer evictable"
        );
    }
}
