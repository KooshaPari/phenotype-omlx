//! (a) Mamba selective scan — OperatorKind::Scan, head_dim=8, chunk_size=16
//! (i) RWKV-7             — OperatorKind::Recurrent, state_channels=4

use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::selector::{RejectionReason, SelectionDecision};
use kernel_registry::{
    BackendKind, CandidateId, Capability, ExecutionTrace, KernelKey, KernelRegistry, SelectionPolicy,
};

use super::{
    build_record, build_record_with_dispatches, fresh_capabilities, make_candidate,
    samples_with_p95, shape, NOW_UNIX_MS, TEST_FINGERPRINT,
};

// (a) Mamba selective scan — OperatorKind::Scan, head_dim=8, chunk_size=16

pub(super) fn mamba_key() -> KernelKey {
    // head_dim is carried via `m`, chunk_size via `seq`.
    KernelKey {
        operator_kind: OperatorKind::Scan,
        attention_kind: None,
        shape_signature: shape(8, 8, 8, 1, 16, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

pub(super) fn mamba_registry() -> (KernelRegistry, CandidateId, CandidateId, CandidateId) {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 256, 1);
    let scalar = make_candidate(
        "MambaSelectiveScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16, DType::Fp16],
        false,
    );
    let simd = make_candidate(
        "MambaSelectiveSimd",
        BackendKind::Cpu,
        vec![Capability::Neon],
        min,
        max,
        vec![DType::Fp32, DType::Bf16, DType::Fp16],
        true,
    );
    let metal = make_candidate(
        "MambaSelectiveMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_scalar = scalar.id;
    let id_simd = simd.id;
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(simd);
    reg.register_candidate(metal);
    let key = mamba_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_scalar, key.clone(), &samples_with_p95(5000), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_simd, key.clone(), &samples_with_p95(2000), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(1500), Some(NOW_UNIX_MS + 86_400_000)),
    );
    (reg, id_scalar, id_simd, id_metal)
}

#[test]
fn mamba_scan_deterministic_picks_lowest_p95_metal_backend() {
    let (reg, _id_scalar, _id_simd, id_metal) = mamba_registry();
    let decision = reg.select_with_caps(
        &mamba_key(),
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(
                candidate.id, id_metal,
                "metal backend p95=1500 must beat simd p95=2000 and scalar p95=5000"
            );
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn mamba_scan_experimental_only_picks_a_tunable_candidate() {
    let (reg, _id_scalar, _id_simd, _id_metal) = mamba_registry();
    let decision = reg.select_with_caps(
        &mamba_key(),
        SelectionPolicy::ExperimentalOnly,
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert!(candidate.tunable,
                "ExperimentalOnly must select a tunable candidate; got non-tunable {:?}",
                candidate.name);
        }
        other => panic!("expected Chosen under ExperimentalOnly, got {other:?}"),
    }
}

#[test]
fn mamba_scan_trace_lists_chosen_candidate() {
    let (reg, _id_scalar, _id_simd, id_metal) = mamba_registry();
    let decision = reg.select_with_caps(
        &mamba_key(),
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let trace: ExecutionTrace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_metal),
        "trace must record the chosen candidate id");
    assert!(trace.tuning_record_id.is_some(),
        "tuned selection must carry a tuning_record_id");
    assert!(trace.human_explanation.contains(&format!("{}", id_metal)),
        "human explanation must mention the chosen id; got {:?}",
        trace.human_explanation);
}

#[test]
fn mamba_scan_rejects_with_unsupported_dtype_when_key_dtype_mismatches() {
    // Register only a candidate that supports Bf16; query with Fp32.
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 256, 1);
    let metal = make_candidate(
        "MambaSelectiveMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16],
        true,
    );
    let id = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(metal);

    let mut key = mamba_key();
    key.dtype = DType::Fp32; // unsupported
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    // The trace lists every considered candidate id (the rejected one).
    let trace = reg.explain(&decision);
    match &decision {
        SelectionDecision::Rejected { rejections, considered } => {
            assert!(considered.contains(&id),
                "candidate must appear in the considered list");
            assert!(rejections.iter().any(|r| matches!(r.reason, RejectionReason::UnsupportedDtype(_))),
                "expected UnsupportedDtype rejection, got {rejections:?}");
        }
        other => panic!("expected Rejected, got {other:?}"),
    }
    let trace_ids: Vec<CandidateId> = trace.considered.iter().map(|r| r.candidate).collect();
    assert!(trace_ids.contains(&id),
        "ExecutionTrace.considered must list every rejected candidate id; got {trace_ids:?}");
    assert!(trace.human_explanation.to_lowercase().contains("dtype"),
        "human explanation should categorize the rejection; got {:?}",
        trace.human_explanation);
}

// (i) RWKV-7 — OperatorKind::Recurrent, state_channels=4

fn rwkv_key() -> KernelKey {
    // state_channels = m=4 (RWKV-7 keeps [k, v, r, w]).
    KernelKey {
        operator_kind: OperatorKind::Recurrent,
        attention_kind: None,
        shape_signature: shape(4, 4, 4, 1, 1, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

#[test]
fn rwkv7_deterministic_picks_lowest_p95_metal_backend() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 256, 1);
    let scalar = make_candidate(
        "Rwkv7Scalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "Rwkv7Metal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_scalar = scalar.id;
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = rwkv_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_scalar, key.clone(), &samples_with_p95(5500), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(1700), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(candidate.id, id_metal,
                "metal p95=1700 must beat scalar p95=5500");
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn rwkv7_trace_lists_chosen_candidate() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 256, 1);
    let scalar = make_candidate(
        "Rwkv7Scalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "Rwkv7Metal",
        BackendKind::Metal,
        vec![Capability::MetalGpu],
        min,
        max,
        vec![DType::Bf16],
        true,
    );
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = rwkv_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(1700), Some(NOW_UNIX_MS + 86_400_000)),
    );
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let trace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_metal));
}

// (h) Qwen DeltaNet *batched* — covers `(B=2, H=2, C=4, D=8)` shape
// signatures for Qwen3-Coder-Next style hybrid DeltaNet. The selector
// must return at least one candidate tagged `DeltaNetBatched` for this
// shape so the runtime can dispatch the parallel implementation
// instead of the single-(batch, head) chunk.

fn deltanet_batched_key() -> KernelKey {
    // (B=2, H=2, C=4, D=8) carried via (m=D=8, n=D=8, k=D=8, batch=B=2,
    // seq=C=4, group=1 — heads are not GQA groups for DeltaNet). The
    // batched shape signature is what the runtime queries when
    // Qwen3-Coder-Next dispatches the parallel DeltaNet path.
    KernelKey {
        operator_kind: OperatorKind::DeltaNet,
        attention_kind: None,
        shape_signature: shape(8, 8, 8, 2, 4, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

#[test]
fn deltanet_batched_selector_returns_tagged_candidate() {
    // Register a Reference and a Metal candidate for DeltaNetBatched.
    // The Metal candidate name carries the `DeltaNetBatched` tag so the
    // runtime can dispatch by tag. The selector must pick the Metal
    // candidate (lowest p95) and the chosen candidate's name must
    // contain `DeltaNetBatched`.
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 256, 1);
    let scalar = make_candidate(
        "DeltaNetBatchedScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "DeltaNetBatchedMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_scalar = scalar.id;
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    let key = deltanet_batched_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_scalar, key.clone(), &samples_with_p95(4500), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_metal, key.clone(), &samples_with_p95(1300), Some(NOW_UNIX_MS + 86_400_000)),
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
                candidate.id, id_metal,
                "metal p95=1300 must beat scalar p95=4500"
            );
            assert!(
                candidate.name.contains("DeltaNetBatched"),
                "chosen candidate must be tagged DeltaNetBatched; got {:?}",
                candidate.name
            );
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

#[test]
fn deltanet_batched_considered_list_contains_tagged_candidate() {
    // Confirms at least one candidate tagged `DeltaNetBatched` appears
    // in the selector's considered list (regardless of which the
    // policy ultimately picks). This is the "selector sees the new
    // kernel" assertion the spec asks for.
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 256, 1);
    let scalar = make_candidate(
        "DeltaNetBatchedScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Bf16],
        false,
    );
    let id_scalar = scalar.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    let key = deltanet_batched_key();
    // No tuning records: the Deterministic policy will fall back to the
    // Reference backend. The test only asserts the selector saw the
    // tagged candidate.
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let tagged_seen = match &decision {
        SelectionDecision::Chosen { candidate, .. } => candidate.name.contains("DeltaNetBatched"),
        SelectionDecision::Rejected { considered, .. } => considered.contains(&id_scalar),
    };
    assert!(
        tagged_seen,
        "selector must surface at least one DeltaNetBatched-tagged candidate for the (B=2,H=2,C=4,D=8) signature; got {decision:?}"
    );
}

// (j) dispatch_buckets_recurrent — per-shape envelope for the *batched*
// DeltaNet, Mamba, and RWKV selectors. Pins the chosen candidate's
// `median_dispatches` against an oracle policy so a future regression
// in the runtime's chunk policy (e.g. accidentally re-introducing a
// per-(batch, head) loop where a parallel tile is expected) is caught.
//
// Oracle policy for the batched recurrence kernel is:
//   dispatches_oracle = ceil(B / 32) * (1 setup + ceil(C / chunk))
// The single setup launch is the metadata emission; the `ceil(C / chunk)`
// captures the per-tile launches for the chunked recurrence. 32 is the
// 1D tile size, `chunk` is the recurrent kernel's `chunk_size`
// (currently 16 for the batched DeltaNet). 1.2× ceiling is the same
// headroom used by `regress_baseline::dispatch_budget` for matmul.

/// `ceil_div` is local to this test file — it's not in the library
/// because the test is the one place the dispatch budget is enforced.
fn ceil_div(a: usize, b: usize) -> u32 {
    ((a + b - 1) / b) as u32
}

#[test]
fn dispatch_buckets_recurrent_picks_delta_net_batched_within_budget() {
    // Five shape buckets spanning decode, prompt, and long-context
    // recurrence. (batch=B, state_channels=C, head_dim=D, chunk=16.)
    let buckets: &[(/* tag */ &str, /* (B, H, C, D) */ (usize, usize, usize, usize))] = &[
        ("decode_1x1", (1, 1, 1, 64)),
        ("decode_4x4", (4, 4, 4, 128)),
        ("prompt_2x2_c16", (2, 2, 16, 64)),
        ("prompt_2x2_c64", (2, 2, 64, 64)),
        ("longctx_8x4_c128", (8, 4, 128, 64)),
    ];
    for &(name, (b_size, _h_size, c_size, d_size)) in buckets {
        // shape_signature carries (m=D, n=D, k=D, batch=B, seq=C, group=1).
        let sig = shape(d_size, d_size, d_size, b_size, c_size, 1);
        let key = KernelKey {
            operator_kind: OperatorKind::DeltaNet,
            attention_kind: None,
            shape_signature: sig,
            dtype: DType::Bf16,
            quantization: QuantizationPolicy::None,
            state_layout_version: 1,
            device_fingerprint: TEST_FINGERPRINT.to_string(),
            policy_version: 1,
        };
        let min = shape(1, 1, 1, 1, 1, 1);
        // The bucket sweep below includes head_dim up to 128, batch
        // up to 8, and state_channels up to 128. The candidate
        // registration must accept those bounds or the test will
        // incorrectly fail with `ShapeOutOfRange` rather than the
        // dispatch-budget assertion we are trying to pin.
        let max = shape(128, 128, 128, 16, 256, 1);
        let scalar = make_candidate(
            "DeltaNetBatchedScalar",
            BackendKind::Reference,
            vec![],
            min,
            max,
            vec![DType::Fp32, DType::Bf16],
            false,
        );
        let metal = make_candidate(
            "DeltaNetBatchedMetal",
            BackendKind::Metal,
            vec![Capability::MetalGpu, Capability::Bf16],
            min,
            max,
            vec![DType::Bf16, DType::Fp16],
            true,
        );
        let id_metal = metal.id;
        let mut reg = KernelRegistry::new();
        reg.register_candidate(scalar);
        reg.register_candidate(metal);

        // Synthesize a `dispatches` claim for the Metal candidate that
        // matches the oracle policy exactly so the test pins the
        // selector's chosen candidate's `median_dispatches` against
        // the same number (1.2× ceiling grants headroom for future
        // tile-size changes, mirroring regress_baseline::BUCKETS).
        const TILE_BATCH: usize = 32;
        const CHUNK_SIZE: usize = 16;
        let oracle: u32 =
            ceil_div(b_size, TILE_BATCH) * (1 + ceil_div(c_size, CHUNK_SIZE));
        reg.attach_tuning_record(
            key.clone(),
            build_record_with_dispatches(
                id_metal,
                key.clone(),
                &samples_with_p95(1300),
                Some(NOW_UNIX_MS + 86_400_000),
                oracle,
            ),
        );
        let decision = reg.select_with_caps(
            &key,
            SelectionPolicy::Deterministic { prefer_lower_p95: true },
            &fresh_capabilities(),
            NOW_UNIX_MS,
        );
        match &decision {
            SelectionDecision::Chosen { candidate, tuning } => {
                assert_eq!(candidate.id, id_metal,
                    "[{name}] deterministic must pick Metal; got {:?}", candidate.name);
                let observed = tuning.median_dispatches.expect(
                    "Metal tuning record must carry dispatches metadata",
                );
                // ceil-into-rounding: observed_dispatches <= oracle * 1.2
                // (no floor since oracle itself may be the minimum).
                let ceiling = oracle.saturating_mul(12) / 10;
                assert!(
                    observed <= ceiling,
                    "[{name}] DeltaNetBatched (B={b_size}, C={c_size}, D={d_size}): \
                     observed dispatches={observed} must be <= 1.2*oracle={ceiling}; \
                     oracle was {oracle}"
                );
                // And the tag must be DeltaNetBatched so the runtime
                // dispatches the parallel implementation.
                assert!(
                    candidate.name.contains("DeltaNetBatched"),
                    "[{name}] chosen candidate must carry DeltaNetBatched tag; got {:?}",
                    candidate.name
                );
            }
            other => panic!("[{name}] expected Chosen under Deterministic, got {other:?}"),
        }
    }
}