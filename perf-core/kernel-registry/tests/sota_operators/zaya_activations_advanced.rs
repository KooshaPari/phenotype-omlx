//! ZAYA 1-bit activation selector coverage — advanced surface.
//!
//! This file is the **advanced** half of the `zaya_activations` test
//! family, split from the prior `zaya_activations.rs` (476L) in
//! turn-9's module-size sweep. The basic half
//! (`zaya_activations_basic.rs`) owns the shared helpers
//! (`zaya_binary_act_key`, `B`/`C` constants, the LCG input fixture,
//! the scalar / packed-bits matmul references). This file picks them
//! up via `use super::zaya_activations_basic::{...}` so the LCG and
//! the pack/unpack routines live in exactly one place.
//!
//! One test covers the advanced surface:
//!
//! 1. `zaya_binary_act_metal_capability_required` — a Metal
//!    binary-activation candidate that does *not* declare
//!    `Capability::MetalMs3` must be rejected with
//!    `RejectionReason::MissingCapability("metal-ms3")`. This is the
//!    capability-floor contract the runtime relies on when it falls
//!    back to scalar under iGPUs that lack MSL 3.0 atomics. The test
//!    pins both halves of the contract in a single `#[test]`: (a) a
//!    device without `MetalMs3` must reject the Metal candidate with
//!    `MissingCapability("metal-ms3")`; (b) a device with the cap (the
//!    production-path `fresh_capabilities()`) must accept and pick it.

use kernel_registry::selector::SelectionDecision;
use kernel_registry::{BackendKind, Capability, KernelRegistry, SelectionPolicy};

use super::{
    build_record, fresh_capabilities, make_candidate, samples_with_p95, shape, NOW_UNIX_MS,
};

use super::zaya_activations_basic::{zaya_binary_act_key, B, C};

// ---------------------------------------------------------------------------
// Test 4 — MetalMs3 capability is required for the Metal backend
// ---------------------------------------------------------------------------

#[test]
fn zaya_binary_act_metal_capability_required() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(B, C, 16, 4, 1, 1);
    // Metal binary-activation candidate declares `Capability::MetalMs3`
    // in its `requires` list — the ZAYA pack/unpack kernels rely on
    // MSL 3.0+ atomics, so this cap is non-negotiable for the Metal
    // backend. On a device that lacks `MetalMs3` (legacy iGPU, certain
    // virtualized Metal stacks) the selector must reject this
    // candidate with `MissingCapability("metal-ms3")`. This pins the
    // capability-floor contract the runtime relies on for the ZAYA
    // fallback path.
    //
    // To force a `SelectionDecision::Rejected` (which surfaces the
    // rejection list) we register ONLY the Metal candidate. The
    // control assertion at the bottom of the test verifies the
    // positive case (Metal wins with fresh caps) so both halves of
    // the contract are covered in a single #[test].
    let metal_requires_ms3 = make_candidate(
        "BinaryActMetalMs3",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::MetalMs3],
        min,
        max,
        vec![kernel_registry::compat::DType::Int8],
        true,
    );
    let id_metal_requires_ms3 = metal_requires_ms3.id;
    let key = zaya_binary_act_key();

    // (1) Device lacks MetalMs3 → metal candidate must be rejected
    //     with `MissingCapability("metal-ms3")`.
    let mut reg = KernelRegistry::new();
    reg.register_candidate(metal_requires_ms3.clone());
    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_metal_requires_ms3,
            key.clone(),
            &samples_with_p95(1700),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    let caps_no_ms3 = kernel_registry::DeviceCaps {
        capabilities: vec![
            kernel_registry::Capability::MetalGpu,
            kernel_registry::Capability::Bf16,
        ],
    };
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &caps_no_ms3,
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Rejected { rejections, considered } => {
            assert!(considered.contains(&id_metal_requires_ms3),
                "the Metal (requires-Ms3) candidate must be considered before rejection");
            assert!(rejections.iter().any(|r| {
                matches!(
                    &r.reason,
                    kernel_registry::selector::RejectionReason::MissingCapability(s)
                        if s == "metal-ms3"
                )
            }),
                "expected MissingCapability(\"metal-ms3\") rejection, got {rejections:?}");
        }
        other => panic!("expected Rejected, got {other:?}"),
    }

    // (2) Device has MetalMs3 (the production-path caps from
    //     `fresh_capabilities()`) → the same Metal candidate wins.
    //     This is the positive-control half: the two assertions
    //     together pin the symmetry "missing cap ⇒ reject; present
    //     cap ⇒ choose" without ambiguity.
    let fresh = fresh_capabilities();
    assert!(fresh.capabilities.contains(&Capability::MetalMs3),
        "fresh_capabilities() must include MetalMs3 — test 1's contract relies on it");
    let mut reg2 = KernelRegistry::new();
    let scalar2 = make_candidate(
        "BinaryActScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![kernel_registry::compat::DType::Int8, kernel_registry::compat::DType::Fp32],
        false,
    );
    let metal2 = make_candidate(
        "BinaryActMetalMs3",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::MetalMs3],
        min,
        max,
        vec![kernel_registry::compat::DType::Int8],
        true,
    );
    let id_metal2 = metal2.id;
    reg2.register_candidate(scalar2);
    reg2.register_candidate(metal2);
    reg2.attach_tuning_record(
        key.clone(),
        build_record(id_metal2, key.clone(), &samples_with_p95(1700), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let chosen = reg2.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh,
        NOW_UNIX_MS,
    );
    match chosen {
        SelectionDecision::Chosen { candidate, .. } => assert_eq!(candidate.id, id_metal2),
        other => panic!("expected Chosen (Metal wins with fresh caps), got {other:?}"),
    }
}
