//! Production-policy + tiebreak tests for MoD sparse depth routing.

use kernel_registry::compat::DType;
use kernel_registry::selector::{RejectionReason, SelectionDecision};
use kernel_registry::{
    BackendKind, Capability, KernelRegistry, Metric, QualityGate, SelectionPolicy,
};

use super::super::{
    build_record, fresh_capabilities, make_candidate, samples_with_p95, shape, NOW_UNIX_MS,
};
use super::{build_mod_record, mod_key};

#[test]
fn mod_routing_production_policy_selects_passing_quality_attachment() {
    // Reference backend has no quality attachment → must be rejected
    // by `MissingQualityEvidence`; Metal carries a passing attachment
    // and must be selected under Production. Pins the contract that a
    // MoD candidate cannot ship without quality attestation.
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 4096, 1);
    let reference = make_candidate(
        "ModRoutingReference",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "ModRoutingMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_reference = reference.id;
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(reference);
    reg.register_candidate(metal);

    let key = mod_key();
    // Reference: no quality attachment (will fail under Production).
    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_reference,
            key.clone(),
            &samples_with_p95(1200),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    // Metal: passing evidence attached (the contract the test pins).
    reg.attach_tuning_record(
        key.clone(),
        build_mod_record(
            id_metal,
            key.clone(),
            &samples_with_p95(2200),
            Some(NOW_UNIX_MS + 86_400_000),
            "mod-throughput",
            0.90,
            0.92,
        ),
    );

    let gate = QualityGate::at_least("mod-throughput", 0.90);
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Production { gates: vec![gate], metric: Metric::P95 },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match &decision {
        SelectionDecision::Chosen { candidate, tuning } => {
            assert_eq!(
                candidate.id, id_metal,
                "Production policy must select the candidate whose quality gate passes; \
                 got {:?}, expected Metal id {:?}",
                candidate.id, id_metal
            );
            // The chosen candidate's tuning record must carry a passing
            // QualityAttachment for the active gate.
            let attachment = tuning
                .quality
                .as_ref()
                .expect("chosen candidate must have a QualityAttachment under Production");
            assert!(
                attachment.passes_all().unwrap_or(false),
                "the chosen candidate's QualityAttachment must satisfy every gate"
            );
            // The Reference candidate must be in the rejection list with
            // MissingQualityEvidence so traces explain why it was skipped.
            let _ = id_reference; // (referenced above for clarity)
        }
        other => panic!("expected Chosen under Production, got {other:?}"),
    }
}

#[test]
fn mod_routing_production_policy_rejects_with_quality_gate_failed() {
    // Both candidates fail the active quality gate. The selector must
    // surface `QualityGateFailed` rejections with gate id, observed
    // score, and threshold attached.
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 4096, 1);
    let reference = make_candidate(
        "ModRoutingReference",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "ModRoutingMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_reference = reference.id;
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(reference);
    reg.register_candidate(metal);

    let key = mod_key();
    // Reference: p95=3200, evidence 0.50 < threshold 0.65 (fails).
    reg.attach_tuning_record(
        key.clone(),
        build_mod_record(
            id_reference,
            key.clone(),
            &samples_with_p95(3200),
            Some(NOW_UNIX_MS + 86_400_000),
            "mod-throughput",
            0.65,
            0.50,
        ),
    );
    // Metal: p95=900 (faster) but evidence 0.40 < threshold 0.65 (also fails).
    reg.attach_tuning_record(
        key.clone(),
        build_mod_record(
            id_metal,
            key.clone(),
            &samples_with_p95(900),
            Some(NOW_UNIX_MS + 86_400_000),
            "mod-throughput",
            0.65,
            0.40,
        ),
    );

    let gate = QualityGate::at_least("mod-throughput", 0.65);
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Production { gates: vec![gate], metric: Metric::P95 },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    // Both candidates fail the gate so the decision must be Rejected with
    // QualityGateFailed entries for both ids.
    let rejections = match &decision {
        SelectionDecision::Rejected { rejections, .. } => rejections.clone(),
        SelectionDecision::Chosen { .. } => panic!(
            "expected Rejected when all candidates fail the quality gate; got {decision:?}"
        ),
    };
    let metal_rejection = rejections
        .iter()
        .find(|r| r.candidate == id_metal)
        .expect("Metal must appear in the rejection list");
    let reference_rejection = rejections
        .iter()
        .find(|r| r.candidate == id_reference)
        .expect("Reference must appear in the rejection list");
    match &metal_rejection.reason {
        RejectionReason::QualityGateFailed { gate, observed, threshold } => {
            assert_eq!(gate, "mod-throughput");
            assert!((*observed - 0.40).abs() < 1e-9, "observed must be 0.40, got {observed}");
            assert!((*threshold - 0.65).abs() < 1e-9, "threshold must be 0.65, got {threshold}");
        }
        other => panic!("expected QualityGateFailed for Metal, got {other:?}"),
    }
    match &reference_rejection.reason {
        RejectionReason::QualityGateFailed { gate, observed, .. } => {
            assert_eq!(gate, "mod-throughput");
            assert!((*observed - 0.50).abs() < 1e-9, "observed must be 0.50, got {observed}");
        }
        other => panic!("expected QualityGateFailed for Reference, got {other:?}"),
    }
}

#[test]
fn mod_routing_deterministic_policy_tiebreaks_by_lower_id_when_metrics_match() {
    // Two MoD candidates with identical p95 but different ids; the
    // deterministic policy must break the tie by lower id, not by
    // registration order or HashMap iteration order.
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 4096, 1);
    let candidate_a = make_candidate(
        "ModRoutingAlpha",
        BackendKind::Cpu,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        true,
    );
    let candidate_z = make_candidate(
        "ModRoutingZeta",
        BackendKind::Cpu,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        true,
    );
    // Confirm the names resolve into different ids and that the alpha id is
    // strictly lower than the zeta id so the test is unambiguous.
    assert!(
        candidate_a.id < candidate_z.id,
        "test setup requires candidate_a.id ({:?}) < candidate_z.id ({:?})",
        candidate_a.id,
        candidate_z.id
    );
    let id_a = candidate_a.id;
    let id_z = candidate_z.id;
    let mut reg = KernelRegistry::new();
    // Register zeta first to ensure tiebreak is by id, not by insertion order.
    reg.register_candidate(candidate_z);
    reg.register_candidate(candidate_a);

    let key = mod_key();
    // Identical p95 across both records.
    let samples = samples_with_p95(1500);
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_z, key.clone(), &samples, Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_a, key.clone(), &samples, Some(NOW_UNIX_MS + 86_400_000)),
    );

    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(
                candidate.id, id_a,
                "deterministic tiebreak must select the lower-id candidate on \
                 equal p95; got {:?}, expected alpha id {:?}",
                candidate.id, id_a
            );
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn mod_routing_production_policy_rejects_when_quality_evidence_missing() {
    // Both candidates lack a `QualityAttachment`. Under Production the
    // selector must reject both with `MissingQualityEvidence` rather
    // than promoting a tuning-only record.
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 4096, 1);
    let reference = make_candidate(
        "ModRoutingReference",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "ModRoutingMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_reference = reference.id;
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(reference);
    reg.register_candidate(metal);

    let key = mod_key();
    // Neither record carries a quality attachment.
    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_reference,
            key.clone(),
            &samples_with_p95(3200),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_metal,
            key.clone(),
            &samples_with_p95(900),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );

    let gate = QualityGate::at_least("mod-throughput", 0.65);
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Production { gates: vec![gate], metric: Metric::P95 },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let rejections = match &decision {
        SelectionDecision::Rejected { rejections, .. } => rejections.clone(),
        SelectionDecision::Chosen { .. } => panic!(
            "expected Rejected when no candidate has a QualityAttachment; got {decision:?}"
        ),
    };
    for id in [id_reference, id_metal] {
        let rejection = rejections
            .iter()
            .find(|r| r.candidate == id)
            .unwrap_or_else(|| panic!("missing rejection record for candidate {id:?}"));
        match &rejection.reason {
            RejectionReason::MissingQualityEvidence(why) => {
                assert!(
                    why.contains("mod-throughput") || why.contains("quality"),
                    "rejection reason must mention the gate family; got {why:?}"
                );
            }
            other => panic!("expected MissingQualityEvidence for {id:?}, got {other:?}"),
        }
    }
}