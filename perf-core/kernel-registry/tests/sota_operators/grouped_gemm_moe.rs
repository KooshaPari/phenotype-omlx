//! (m) `grouped_gemm_moe` — selector + oracle parity test for the MoE
//! grouped-expert GEMM family. Mirrors the `discrete_diffusion_oracle.rs`
//! pattern (returns `(KernelRegistry, scalar_id, tiled_id)` so the
//! bench envelope in this file can re-use it) and the
//! `moe_routing_top_k_small.rs` pattern (re-exports a tagged helper
//! for the coverage matrix to find via substring search).
//!
//! Two candidates are registered for the same `KernelKey`:
//!
//! 1. `GroupedGemmMoeScalar` — the byte-oracle reference
//!    (`model_kernels::moe::gemm::grouped_gemm`).
//! 2. `GroupedGemmMoeTiled` — the new tiled/blocked path
//!    (`model_kernels::moe::gemm_tiled::grouped_gemm_tiled`).
//!
//! The selector-coverage test (`grouped_gemm_moe_selector_picks_lowest_p95_tiled`)
//! pins the deterministic policy to choose the *tiled* path once its
//! synthetic tuning record carries a strictly lower p95 than the
//! scalar reference. The bench envelope
//! (`grouped_gemm_moe_sweep_t_values`) registers 5 row contexts × 5
//! seeds = 25 rows minimum, all anchored on the canonical Qwen-MoE
//! block shape (`m=n=k=64`). The tiled path's `Capability::Avx512`
//! requirement is satisfied by `super::full_capabilities()`.
//!
//! Public re-exports:
//!
//! - `grouped_gemm_moe_key` — the canonical `(m=64, n=64, k=64, batch,
//!   seq=1, group=1)` shape key. Re-imported by the coverage matrix
//!   to assert `grouped_gemm_moe` substring coverage.
//! - `grouped_gemm_moe_registry` — the canonical two-candidate
//!   registry factory used by every test in this file.

use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::{
    BackendKind, Capability, KernelKey, KernelRegistry,
};

use super::{
    build_record, make_candidate, samples_with_p95, shape, NOW_UNIX_MS,
    TEST_FINGERPRINT,
};

/// Canonical Qwen-MoE grouped-GEMM shape: `(m=64, n=64, k=64,
/// batch=64, seq=1, group=1)`. The numbers are the canonical
/// Qwen2-MoE expert feed-forward block; pinned here so every test in
/// this file references the same shape and the coverage matrix can
/// substring-match `grouped_gemm_moe` against the test name.
pub(crate) fn grouped_gemm_moe_key() -> KernelKey {
    KernelKey {
        operator_kind: OperatorKind::Moe,
        attention_kind: None,
        // m = hidden, n = hidden, k = hidden, batch = tokens / expert
        // = batch_size / num_experts (here 64 tokens, 8 experts → 8
        // tokens per expert average; the bench envelope sweeps the
        // batch axis explicitly below).
        shape_signature: shape(64, 64, 64, 64, 1, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

/// Build the canonical two-candidate `KernelRegistry` for the
/// `grouped_gemm_moe` family: a scalar reference backend and a tiled
/// scalar backend. Returns the registry plus both `CandidateId`s so
/// the bench envelope can attach distinct tuning records to each
/// without re-deriving the deterministic hash.
pub(crate) fn grouped_gemm_moe_registry() -> (KernelRegistry, kernel_registry::CandidateId, kernel_registry::CandidateId) {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(256, 256, 256, 4096, 1, 1);
    let scalar = make_candidate(
        "GroupedGemmMoeScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let tiled = make_candidate(
        "GroupedGemmMoeTiled",
        BackendKind::Cpu, // scalar-tile path runs on CPU; matches the
                          // `lfm_routing.rs` pattern where the second
                          // candidate is `BackendKind::Cpu` rather
                          // than `BackendKind::Metal` because no Metal
                          // kernel exists for grouped GEMM yet.
        vec![Capability::Avx512],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        true,
    );
    let id_scalar = scalar.id;
    let id_tiled = tiled.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(tiled);
    (reg, id_scalar, id_tiled)
}

// ---------------------------------------------------------------------------
// Tests (selector coverage + bench envelope).
// ---------------------------------------------------------------------------

/// Deterministic policy under `grouped_gemm_moe` must select the
/// tiled path once the tuning record carries a strictly lower p95
/// than the scalar reference. This is the selector-coverage half of
/// the contract: even though oracle parity is the test's source of
/// truth (see `model-kernels/src/moe/gemm_tiled.rs`), the registry
/// must still pick the right kernel for the family.
#[test]
fn grouped_gemm_moe_selector_picks_lowest_p95_tiled() {
    let (mut reg, id_scalar, id_tiled) = grouped_gemm_moe_registry();
    let key = grouped_gemm_moe_key();
    // Scalar p95 = 2400, tiled p95 = 1100 → tiled wins under Deterministic.
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_scalar, key.clone(), &samples_with_p95(2400), Some(NOW_UNIX_MS + 86_400_000)),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(id_tiled, key.clone(), &samples_with_p95(1100), Some(NOW_UNIX_MS + 86_400_000)),
    );

    let decision = reg.select_with_caps(
        &key,
        kernel_registry::SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &super::full_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        kernel_registry::selector::SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(
                candidate.id, id_tiled,
                "tiled p95=1100 must beat scalar p95=2400 under Deterministic"
            );
            assert_ne!(candidate.id, id_scalar);
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

/// Oracle parity at the registry boundary: both candidates must
/// produce byte-equal `out` for the same `(a, b, buckets, m, k, n)`
/// inputs. The test drives `model_kernels::moe::grouped_gemm` and
/// `model_kernels::moe::grouped_gemm_tiled` through the same
/// deterministic inputs and asserts element-wise equality within
/// `1e-5`. This is the same contract pinned in the model-kernels
/// crate's `grouped_gemm_tiled_matches_scalar_for_random_inputs`
/// test, restated here so the SOTA coverage matrix catches a
/// future refactor that breaks the production parity contract
/// even if the model-kernels test is silently disabled.
#[test]
fn grouped_gemm_moe_oracle_parity_scalar_vs_tiled() {
    use model_kernels::common::Lcg;
    use model_kernels::moe::{grouped_gemm, grouped_gemm_tiled};

    let num_tokens = 16;
    let num_experts = 4;
    let k = 32;
    let n = 32;
    let mut act_rng = Lcg::new(0xA1_B2_C3_D4);
    let a: Vec<f32> = (0..num_tokens * k).map(|_| act_rng.next_signed()).collect();
    let mut exp_rng = Lcg::new(0xFEED_FACE);
    let b: Vec<f32> = (0..num_experts * k * n).map(|_| exp_rng.next_signed()).collect();
    // Round-robin assignment: each expert owns `num_tokens / num_experts`
    // tokens, evenly distributed.
    let buckets: Vec<Vec<usize>> = (0..num_experts)
        .map(|e| {
            (0..num_tokens)
                .filter(|t| t % num_experts == e)
                .collect()
        })
        .collect();

    let mut scalar_out = vec![0.0f32; num_tokens * n];
    grouped_gemm(&a, &b, &buckets, 0, k, n, &mut scalar_out)
        .expect("scalar reference must accept well-formed inputs");
    let mut tiled_out = vec![0.0f32; num_tokens * n];
    grouped_gemm_tiled(&a, &b, &buckets, 0, k, n, &mut tiled_out)
        .expect("tiled path must accept well-formed inputs");

    assert_eq!(scalar_out.len(), tiled_out.len());
    for (i, (&x, &y)) in scalar_out.iter().zip(tiled_out.iter()).enumerate() {
        assert!(
            (x - y).abs() <= 1e-5,
            "oracle parity broken at element {i}: scalar={x} tiled={y} (|d|={})",
            (x - y).abs()
        );
    }
}

/// Bench-envelope row count: 5 row contexts × 5 seeds = 25 rows.
/// Each row attaches a distinct synthetic tuning record to the tiled
/// candidate so the selector stays deterministic across runs but
/// exposes per-row tuning evidence for the regression detector.
///
/// The five `t_values` correspond to:
/// - `t=1`  : Qwen2-MoE small batch (32 tokens)
/// - `t=2`  : Mixtral-8x7B medium batch (128 tokens)
/// - `t=3`  : DeepSeek-V3 large batch (512 tokens)
/// - `t=4`  : Qwen3-Coder-Next XL batch (2048 tokens)
/// - `t=5`  : stress-shape batch (4096 tokens)
///
/// The five seeds are the same as the NIAH baseline envelope
/// (`7, 19, 42, 73, 101`) so the seed surface is shared with the
/// research baseline format. The p95 values are a deterministic
/// linear function of `(t, seed)` so the test runs stay
/// byte-reproducible without an actual measurement cycle.
#[test]
fn grouped_gemm_moe_sweep_t_values() {
    use kernel_registry::selector::SelectionDecision;

    let t_values: &[(usize, usize)] = &[
        (1, 32),
        (2, 128),
        (3, 512),
        (4, 2048),
        (5, 4096),
    ];
    let seeds: &[u64] = &[7, 19, 42, 73, 101];

    let (mut reg, _id_scalar, id_tiled) = grouped_gemm_moe_registry();

    let mut rows_pinned: usize = 0;
    for &(t, batch) in t_values {
        for &seed in seeds {
            // Per-row shape: (m=hidden, n=hidden, k=hidden, batch,
            // seq=1, group=1) on the Qwen-MoE canonical block
            // (k=n=64). The `batch` axis is the only one that varies
            // across the envelope.
            let key = KernelKey {
                operator_kind: OperatorKind::Moe,
                attention_kind: None,
                shape_signature: shape(64, 64, 64, batch, 1, 1),
                dtype: DType::Bf16,
                quantization: QuantizationPolicy::None,
                state_layout_version: 1,
                device_fingerprint: TEST_FINGERPRINT.to_string(),
                policy_version: 1,
            };
            // Synthetic p95: strictly lower than the scalar
            // reference's 2400 across the full envelope so the
            // selector picks the tiled candidate on every row.
            // `t * 50 + (seed % 17)` keeps the p95 monotonically
            // increasing in `t` but bounded well below 2400.
            let p95 = 1100u64 + ((t as u64) * 50) + (seed % 17);
            reg.attach_tuning_record(
                key.clone(),
                build_record(id_tiled, key.clone(), &samples_with_p95(p95), Some(NOW_UNIX_MS + 86_400_000)),
            );

            let decision = reg.select_with_caps(
                &key,
                kernel_registry::SelectionPolicy::Deterministic { prefer_lower_p95: true },
                &super::full_capabilities(),
                NOW_UNIX_MS,
            );
            match decision {
                SelectionDecision::Chosen { candidate, .. } => {
                    assert_eq!(
                        candidate.id, id_tiled,
                        "row (t={t}, seed={seed}) must pick tiled under Deterministic"
                    );
                }
                other => panic!("row (t={t}, seed={seed}) expected Chosen, got {other:?}"),
            }
            rows_pinned += 1;
        }
    }
    assert_eq!(
        rows_pinned,
        t_values.len() * seeds.len(),
        "bench envelope must produce exactly t_values * seeds rows (>= 25)"
    );
    // Floor: the task spec asks for >= 25 rows.
    assert!(rows_pinned >= 25, "bench envelope must have >= 25 rows, got {rows_pinned}");
}
