//! `SelectionPolicy::ExperimentalOnly` contract test.
//!
//! Guards the experimental-policy path so it can't silently regress.

use kernel_registry::compat::OperatorKind;
use kernel_registry::selector::{RejectionReason, SelectionDecision};
use kernel_registry::{BackendKind, Capability, DeviceCaps, KernelRegistry, SelectionPolicy};

use super::{candidate_from, key_with, NOW_UNIX_MS, TEST_DEVICE_FINGERPRINT};

#[test]
fn selector_experimental_only_skips_tunable_candidates_without_evidence() {
    let now = NOW_UNIX_MS;
    let mut reg = KernelRegistry::new();
    let tunable_no_evidence = candidate_from(
        "experimental",
        BackendKind::Metal,
        vec![Capability::MetalGpu],
    );
    let id_tunable = tunable_no_evidence.id;
    reg.register_candidate(tunable_no_evidence);
    let key = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);
    let caps = DeviceCaps { capabilities: vec![Capability::MetalGpu] };
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::ExperimentalOnly,
        &caps,
        now,
    );
    match decision {
        SelectionDecision::Rejected { rejections, considered } => {
            assert!(considered.contains(&id_tunable));
            assert!(rejections.iter().any(|r| matches!(r.reason, RejectionReason::NoTuningEvidence)),
                "tunable candidates without tuning must be excluded under ExperimentalOnly");
        }
        other => panic!("expected Rejected under ExperimentalOnly, got {other:?}"),
    }
}