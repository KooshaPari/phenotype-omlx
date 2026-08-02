mod registry_select;
mod registry_validate;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::candidate::{Candidate, CandidateId, Capability};
use crate::key::KernelKey;
use crate::record::TuningRecord;
use crate::selector::SelectionDecision;
use crate::trace::ExecutionTrace;

/// Capability advertisement for the device that will execute kernels. The
/// selector requires `candidate.requires ⊆ caps.capabilities` to deem a
/// candidate eligible.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceCaps {
    pub capabilities: Vec<Capability>,
}

impl DeviceCaps {
    /// Create a new device capability set from the given capabilities.
    pub fn new(capabilities: Vec<Capability>) -> Self {
        Self { capabilities }
    }
}

/// In-memory registry of candidates and tuning evidence.
#[derive(Debug, Default)]
pub struct KernelRegistry {
    pub(crate) candidates: HashMap<CandidateId, Vec<Candidate>>,
    pub(crate) tuning: HashMap<KernelKey, Vec<TuningRecord>>,
}

impl KernelRegistry {
    /// Create an empty registry with no candidates or tuning records.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a candidate. If a candidate with the same id already
    /// exists it is overwritten — callers that need to detect collisions
    /// can use [`KernelRegistry::register_candidate_checked`].
    pub fn register_candidate(&mut self, candidate: Candidate) {
        self.candidates
            .entry(candidate.id)
            .or_default()
            .push(candidate);
    }

    /// Register a candidate only if no candidate with the same id exists
    /// yet. Returns the existing candidate on collision so callers can log
    /// provenance.
    pub fn register_candidate_checked(&mut self, candidate: Candidate) -> Option<Candidate> {
        match self.candidates.entry(candidate.id) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut().first().cloned(),
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(vec![candidate]);
                None
            }
        }
    }

    /// Attach a tuning record for `key`. Records are appended in the order
    /// they arrive; the selector sorts them deterministically.
    pub fn attach_tuning_record(&mut self, key: KernelKey, record: TuningRecord) {
        self.tuning.entry(key).or_default().push(record);
    }

    /// All candidates, sorted by id for deterministic callers.
    pub fn list_candidates(&self) -> Vec<Candidate> {
        let mut v: Vec<Candidate> = self
            .candidates
            .values()
            .flat_map(|v| v.iter().cloned())
            .collect();
        v.sort_by_key(|c| c.id);
        v
    }

    /// Candidates whose id is present in `keys`. The result is sorted by id
    /// ascending.
    pub fn list_candidates_for_ids(&self, keys: &[CandidateId]) -> Vec<Candidate> {
        let mut out: Vec<Candidate> = keys
            .iter()
            .filter_map(|id| self.candidates.get(id))
            .flat_map(|v| v.iter().cloned())
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
            .flat_map(|v| v.iter().cloned())
            .filter(|c| c.supports_shape(&key.shape_signature))
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::{BackendKind, CandidateId};
    use crate::compat::DType;
    use crate::key::ShapeSignature;

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

    // --- register / overwrite ------------------------------------------------

    #[test]
    fn register_duplicate_name_appends_candidate() {
        let mut reg = KernelRegistry::new();
        let c1 = make_candidate("my-kernel");
        let c2 = make_candidate("my-kernel");

        reg.register_candidate(c1);
        assert_eq!(reg.list_candidates().len(), 1);

        reg.register_candidate(c2);
        let listed = reg.list_candidates();
        assert_eq!(
            listed.len(),
            2,
            "duplicate insert appends to the candidate vec"
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
            "collision must return the existing candidate"
        );
        assert_eq!(returned.unwrap().name, "alpha");
        assert_eq!(
            reg.list_candidates().len(),
            1,
            "checked insert must not add duplicate"
        );
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

    // --- lookup non-existent -------------------------------------------------

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

    // --- listing returns all registered --------------------------------------

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

    // --- unregister removes --------------------------------------------------

    #[test]
    fn unregister_candidate_by_id_removes_from_registry() {
        let mut reg = KernelRegistry::new();
        let c = make_candidate("removable");
        let id = c.id;
        reg.register_candidate(c);
        assert_eq!(reg.list_candidates().len(), 1);

        let removed = reg.candidates.remove(&id);
        assert!(removed.is_some(), "remove must return the candidate vec");
        assert!(
            !removed.unwrap().is_empty(),
            "removed vec must be non-empty"
        );
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

    #[test]
    fn new_registry_is_empty() {
        let reg = KernelRegistry::new();
        assert!(reg.candidates.is_empty());
        assert!(reg.tuning.is_empty());
        assert!(reg.list_candidates().is_empty());
    }

    #[test]
    fn register_and_get_round_trip() {
        let mut reg = KernelRegistry::new();
        let c = make_candidate("round-trip");
        let id = c.id;
        reg.register_candidate(c);

        let got = reg
            .candidates
            .get(&id)
            .expect("candidate must be retrievable by id")
            .first()
            .expect("candidate vec must be non-empty");
        assert_eq!(got.name, "round-trip");
        assert_eq!(got.id, id);
    }

    #[test]
    fn get_nonexistent_candidate_returns_none() {
        let reg = KernelRegistry::new();
        let missing = CandidateId(0xFFFF_FFFF);
        assert!(!reg.candidates.contains_key(&missing));
    }

    #[test]
    fn list_returns_all_registered() {
        let mut reg = KernelRegistry::new();
        for name in &["alpha", "bravo", "charlie", "delta"] {
            reg.register_candidate(make_candidate(name));
        }
        let all = reg.list_candidates();
        assert_eq!(all.len(), 4, "list must return every registered candidate");
        let names: Vec<&str> = all.iter().map(|c| c.name.as_str()).collect();
        assert!(
            names.contains(&"alpha") && names.contains(&"delta"),
            "all names must be present"
        );
    }

    #[test]
    fn unregister_removes_candidate() {
        let mut reg = KernelRegistry::new();
        let c = make_candidate("delete-me");
        let id = c.id;
        reg.register_candidate(c);
        assert_eq!(reg.list_candidates().len(), 1);

        let removed = reg.candidates.remove(&id);
        assert!(removed.is_some(), "removal must return the candidate vec");
        assert!(
            !removed.unwrap().is_empty(),
            "removed vec must be non-empty"
        );
        assert_eq!(
            reg.list_candidates().len(),
            0,
            "registry must be empty after removal"
        );
    }
}
