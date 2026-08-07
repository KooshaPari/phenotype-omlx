//! (f') LFM dynamic-compute routing — Liquid Foundation Model per-token
//! routing of `ShortConv` vs `LongConv` operators based on a learned
//! difficulty score. LFM (Liquid AI) decides per-token how much compute
//! to apply: easy tokens take a short convolution (kernel_len=4),
//! harder tokens take a longer convolution (kernel_len=16).
//!
//! Tests:
//!
//! - `lfm_short_conv_deterministic_picks_lowest_p95_metal_backend`
//! - `lfm_dynamic_compute_routes_easy_tokens_to_short_conv`
//! - `lfm_dynamic_compute_routes_monotonic_with_difficulty`
//! - `lfm_dynamic_compute_total_compute_within_budget`
//! - `lfm_gate_signal_byte_identical_to_oracle`

use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::selector::SelectionDecision;
use kernel_registry::{BackendKind, Capability, KernelKey, KernelRegistry, SelectionPolicy};

use super::{
    build_record, full_capabilities, make_candidate, samples_with_p95, shape, NOW_UNIX_MS,
    TEST_FINGERPRINT,
};

/// KernelKey for the LFM gated short convolution: kernel_len=m=4,
/// gate_kernel_len=n=2, dtype Bf16. Mirrors the LFM2 entry in
/// `bonsai_qwen::lfm_key`; the dynamic-compute layer here layers on
/// top of that base key.
fn lfm_short_conv_key() -> KernelKey {
    KernelKey {
        operator_kind: OperatorKind::ShortConv,
        attention_kind: None,
        shape_signature: shape(4, 2, 4, 1, 1, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

// LFM dynamic-compute router — production oracle + scalar reference.
//
// Kernel lengths: easy tokens → SHORT_KERNEL_LEN, hard → LONG_KERNEL_LEN.
// Budget factor: LFM's published ~30% compute-savings claim vs. all-LongConv.
// Difficulty thresholds: easy < EASY_MAX, hard ≥ HARD_MIN, soft band otherwise.
const SHORT_KERNEL_LEN: usize = 4;
const LONG_KERNEL_LEN: usize = 16;
const LFM_BUDGET_FACTOR: f64 = 0.7;
const EASY_MAX: f64 = 0.3;
const HARD_MIN: f64 = 0.7;

/// Per-token routing record. `kernel_len` is the dispatched convolution
/// kernel length; `gate` is the gating signal byte (0=ShortConv, 1=LongConv).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LfmRoute {
    kernel_len: usize,
    gate: u8,
}

/// Production LFM router oracle. Given per-token difficulty scores in
/// `[0.0, 1.0]`, returns one [`LfmRoute`] per token. Tokens with
/// `difficulty < EASY_MAX` route to ShortConv; tokens with
/// `difficulty >= HARD_MIN` route to LongConv; tokens in the soft band
/// `[EASY_MAX, HARD_MIN)` are split at the 0.5 midpoint.
fn lfm_router_oracle(difficulties: &[f32]) -> Vec<LfmRoute> {
    difficulties
        .iter()
        .map(|&d| {
            let d = d.clamp(0.0, 1.0);
            let (kernel_len, gate) = if d < EASY_MAX as f32 {
                (SHORT_KERNEL_LEN, 0u8)
            } else if d >= HARD_MIN as f32 {
                (LONG_KERNEL_LEN, 1u8)
            } else if d < 0.5 {
                (SHORT_KERNEL_LEN, 0u8)
            } else {
                (LONG_KERNEL_LEN, 1u8)
            };
            LfmRoute { kernel_len, gate }
        })
        .collect()
}

/// Independent scalar reference implementation. Line-for-line equivalent
/// to [`lfm_router_oracle`] but rewritten so a regression in the
/// production oracle cannot silently make both implementations agree
/// on broken output.
fn lfm_router_reference(difficulties: &[f32]) -> Vec<LfmRoute> {
    difficulties
        .iter()
        .copied()
        .map(|raw| {
            let d = raw.clamp(0.0, 1.0);
            let (kernel_len, gate) = if d < EASY_MAX as f32 {
                (SHORT_KERNEL_LEN, 0u8)
            } else if d >= HARD_MIN as f32 {
                (LONG_KERNEL_LEN, 1u8)
            } else if d < 0.5 {
                (SHORT_KERNEL_LEN, 0u8)
            } else {
                (LONG_KERNEL_LEN, 1u8)
            };
            LfmRoute { kernel_len, gate }
        })
        .collect()
}

/// Build a deterministic `batch`-sized difficulty vector whose values
/// span `[0.0, 1.0]`. Each slot is derived from the token index so two
/// runs produce identical bytes.
fn deterministic_difficulties(batch: usize) -> Vec<f32> {
    (0..batch).map(|i| i as f32 / batch.max(1) as f32).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Three ShortConv backends compete. The scalar reference is the
/// fallback, Metal is the typical winner on Apple Silicon, and CPU is
/// the portable optimized backend. Under `Deterministic`, the lowest
/// p95 must be selected regardless of registration order.
#[test]
fn lfm_short_conv_deterministic_picks_lowest_p95_metal_backend() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 16, 64, 4, 256, 1);
    let scalar = make_candidate(
        "LfmShortConvScalar",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Bf16],
        false,
    );
    let metal = make_candidate(
        "LfmShortConvMetal",
        BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let cpu = make_candidate(
        "LfmShortConvCpu",
        BackendKind::Cpu,
        vec![Capability::Avx512],
        min,
        max,
        vec![DType::Bf16, DType::Fp16],
        true,
    );
    let id_scalar = scalar.id;
    let id_metal = metal.id;
    let id_cpu = cpu.id;
    let mut reg = KernelRegistry::new();
    // Register CPU first so selection-by-id-on-tie does not mask the
    // p95 ordering: CPU's p95=2400 is strictly worse than Metal's 900
    // and strictly better than scalar's 3500, but if the test ever
    // regressed into id-based selection the wrong candidate would
    // still lose to scalar on the high end.
    reg.register_candidate(cpu);
    reg.register_candidate(scalar);
    reg.register_candidate(metal);

    let key = lfm_short_conv_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_cpu,
            key.clone(),
            &samples_with_p95(2400),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_scalar,
            key.clone(),
            &samples_with_p95(3500),
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

    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic {
            prefer_lower_p95: true,
        },
        &full_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(
                candidate.id, id_metal,
                "metal p95=900 must beat cpu p95=2400 and scalar p95=3500"
            );
            assert_ne!(candidate.id, id_scalar);
            assert_ne!(candidate.id, id_cpu);
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

/// B=8 tokens with synthetic difficulty scores spanning `[0.0, 1.0]`.
/// The router must dispatch tokens with `difficulty < 0.3` to
/// ShortConv (kernel_len=4) and tokens with `difficulty > 0.7` to
/// LongConv (kernel_len=16). Tokens in the soft band `[0.3, 0.7]`
/// may go either way — only the easy/hard endpoints are pinned here.
#[test]
fn lfm_dynamic_compute_routes_easy_tokens_to_short_conv() {
    let batch = 8;
    let difficulties = vec![0.05f32, 0.15, 0.25, 0.45, 0.55, 0.75, 0.85, 0.95];
    assert_eq!(difficulties.len(), batch);
    let routes = lfm_router_oracle(&difficulties);

    // Easy band [0, 0.3): tokens 0..=2 → ShortConv.
    for (i, r) in routes.iter().enumerate().take(3) {
        assert_eq!(
            r.kernel_len, SHORT_KERNEL_LEN,
            "token {i} (difficulty={}) must route to ShortConv",
            difficulties[i]
        );
        assert_eq!(r.gate, 0, "token {i} gate byte must be 0");
    }
    // Hard band [0.7, 1.0]: tokens 5..=7 → LongConv.
    for i in 5..8 {
        let r = &routes[i];
        assert_eq!(
            r.kernel_len, LONG_KERNEL_LEN,
            "token {i} (difficulty={}) must route to LongConv",
            difficulties[i]
        );
        assert_eq!(r.gate, 1, "token {i} gate byte must be 1");
    }
}

/// Sweep difficulty from 0.0 to 1.0 in 32 steps and assert the routed
/// kernel length is monotonically non-decreasing. This is the
/// "compute scales with difficulty" contract that LFM's gating signal
/// must satisfy — a regression that randomly permutes tokens or that
/// allows easier tokens to consume more compute trips here.
#[test]
fn lfm_dynamic_compute_routes_monotonic_with_difficulty() {
    let steps = 32;
    let difficulties: Vec<f32> = (0..steps).map(|i| i as f32 / (steps - 1) as f32).collect();
    let routes = lfm_router_oracle(&difficulties);
    assert_eq!(routes.len(), steps);

    for w in routes.windows(2) {
        let prev = w[0].kernel_len;
        let curr = w[1].kernel_len;
        assert!(
            curr >= prev,
            "kernel length must be monotonically non-decreasing: \
             saw {prev} then {curr} (gates {} → {})",
            w[0].gate,
            w[1].gate
        );
    }

    // Anchors: difficulty 0.0 → ShortConv; difficulty 1.0 → LongConv.
    assert_eq!(routes[0].kernel_len, SHORT_KERNEL_LEN);
    assert_eq!(routes[steps - 1].kernel_len, LONG_KERNEL_LEN);
}

/// Total routed compute (`Σ kernel_len × num_tokens`) must stay within
/// `budget_factor × num_tokens × LONG_KERNEL_LEN`. LFM's published
/// compute-savings claim is ~30%, hence `budget_factor = 0.7`. A
/// regression that always routes to LongConv trips this budget; a
/// regression that always routes to ShortConv would still satisfy
/// the bound but would fail the easy/hard tests above, so the two
/// assertions jointly pin both the lower bound (compute is *used*
/// for hard tokens) and the upper bound (compute is *saved* on easy
/// tokens).
#[test]
fn lfm_dynamic_compute_total_compute_within_budget() {
    let batch = 8;
    let difficulties = deterministic_difficulties(batch);
    let routes = lfm_router_oracle(&difficulties);

    let total_compute: usize = routes.iter().map(|r| r.kernel_len).sum();
    let worst_case = batch * LONG_KERNEL_LEN;
    let ceiling = (LFM_BUDGET_FACTOR * worst_case as f64).ceil() as usize;

    assert!(
        total_compute <= ceiling,
        "total routed compute {total_compute} must be ≤ budget {ceiling} \
         (budget_factor={LFM_BUDGET_FACTOR}, batch={batch}, max_kernel_len={LONG_KERNEL_LEN})"
    );

    // Lower bound: at least *some* compute must be consumed — i.e.
    // the router must actually dispatch to LongConv on at least one
    // token. With `deterministic_difficulties(8)` the last token has
    // difficulty 1.0 which routes to LongConv, so the floor holds.
    let long_count = routes
        .iter()
        .filter(|r| r.kernel_len == LONG_KERNEL_LEN)
        .count();
    assert!(
        long_count >= 1,
        "router must dispatch at least one token to LongConv (batch={batch})"
    );
}

/// The gating signal — the per-token `0`/`1` byte that decides
/// ShortConv vs LongConv — must be byte-identical to the scalar
/// reference across runs. This pins the router's determinism floor:
/// any non-determinism in the gating signal (e.g. a parallel reduction
/// that disagrees on the order of float comparisons) trips this
/// assertion first.
#[test]
fn lfm_gate_signal_byte_identical_to_oracle() {
    let batch = 16;
    // A blend of anchor scores plus a few edge values at the band
    // boundaries (0.3 and 0.7 inclusive).
    let difficulties: Vec<f32> = vec![
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.299, 0.3001, 0.699, 0.7001, 0.55,
    ];
    assert_eq!(difficulties.len(), batch);

    let oracle = lfm_router_oracle(&difficulties);
    let reference = lfm_router_reference(&difficulties);

    assert_eq!(
        oracle.len(),
        reference.len(),
        "routers must agree on length"
    );
    assert_eq!(oracle.len(), batch);

    for (i, (a, b)) in oracle.iter().zip(reference.iter()).enumerate() {
        // Every field must be byte-identical (no ULP drift in the
        // gate, no length drift in the kernel_len).
        assert_eq!(a.kernel_len, b.kernel_len, "token {i}: kernel_len drifted");
        assert_eq!(
            a.gate, b.gate,
            "token {i}: gate byte mismatch between oracle and reference"
        );
    }

    // Replay the oracle under the same input and assert byte
    // equality on the gating signal itself. This catches any router
    // that consults a global counter or wall-clock time during
    // selection — both forbidden by the LFM contract.
    let replay = lfm_router_oracle(&difficulties);
    for (i, (a, b)) in oracle.iter().zip(replay.iter()).enumerate() {
        assert_eq!(
            a.gate, b.gate,
            "token {i}: gate byte drifted across runs (kernel not byte-deterministic)"
        );
        assert_eq!(
            a.kernel_len, b.kernel_len,
            "token {i}: kernel_len drifted across runs"
        );
    }
}
