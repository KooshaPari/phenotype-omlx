//! Selector-coverage tests for the SOTA operator families added by
//! commits c05e3fa, fc4474c, 98939af, ceebda5, c7c4b52.
//!
//! Each family below corresponds to a row in
//! `docs/sessions/20260718-metal-model-runtime/02_SPECIFICATIONS.md`
//! §Model Acceptance Matrix:
//!
//! - Mamba / RWKV-7  : recurrent-hybrid (fc4474c)
//! - ZAYA            : CCA block-parallel (98939af)
//! - LFM             : gated short convolution (98939af)
//! - DeepSeek MLA    : compressed-kv cache (c05e3fa)
//! - DeepSeek MTP    : speculative proposal (c05e3fa)
//! - Bonsai          : ternary matmul (ceebda5)
//! - Qwen            : DeltaNet (ceebda5)
//! - LLaDA / Dream   : diffusion decoder (c7c4b52)
//! - MoD             : sparse depth routing (OperatorKind::Moe discriminator)
//!
//! For every family the test (a) registers a reference scalar backend
//! plus two optimized backends, (b) attaches synthetic tuning records
//! with *distinct* p95 latencies so the deterministic policy has a
//! non-trivial winner, and (c) asserts the selector picked the lowest-p95
//! backend. The observability test pins the trace shape; the rejection
//! test pins dtype enforcement.
//!
//! Convention: `now_unix_ms = 1_700_000_000_000` matches the project
//! baseline so contract tests stay time-independent.

use kernel_registry::compat::DType;
use kernel_registry::selector::{RejectionReason, SelectionDecision};
use kernel_registry::{
    BackendKind, Candidate, CandidateId, Capability, DeviceCaps, KernelKey, KernelRegistry,
    Measurement, SelectionPolicy, ShapeSignature, TuningRecord,
};

pub(crate) const NOW_UNIX_MS: u64 = 1_700_000_000_000;
pub(crate) const TEST_FINGERPRINT: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

// Shared builders used by every per-family module below.

pub(crate) fn shape(m: usize, n: usize, k: usize, batch: usize, seq: usize, group: usize) -> ShapeSignature {
    ShapeSignature { m, n, k, batch, seq, group }
}

pub(crate) fn build_record(
    candidate_id: CandidateId,
    key: KernelKey,
    samples: &[u64],
    expires_at_unix_ms: Option<u64>,
) -> TuningRecord {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let median = sorted[n / 2];
    let p95_idx = ((n as f64 * 0.95).ceil() as usize).saturating_sub(1).min(n - 1);
    let p99_idx = ((n as f64 * 0.99).ceil() as usize).saturating_sub(1).min(n - 1);
    let mean = sorted.iter().sum::<u64>() as f64 / n as f64;
    let variance = sorted
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let measurements: Vec<Measurement> = samples
        .iter()
        .enumerate()
        .map(|(i, &latency_ns)| Measurement::with_metadata(i as u32, latency_ns, None, None, 0))
        .collect();
    TuningRecord {
        candidate_id,
        key,
        measurements,
        median_ns: median,
        p95_ns: sorted[p95_idx],
        p99_ns: sorted[p99_idx],
        variance_ns2: variance as u64,
        median_energy_j: None,
        median_dispatches: None,
        samples: n,
        warmup_discarded: 3,
        compiler: "metal-msl".to_string(),
        compiler_version: "3.2".to_string(),
        captured_at_unix_ms: NOW_UNIX_MS,
        source_revision: "rev-sota".to_string(),
        expires_at_unix_ms,
        quality: None,
    }
}

pub(crate) fn make_candidate(
    name: &str,
    backend: BackendKind,
    requires: Vec<Capability>,
    min_shape: ShapeSignature,
    max_shape: ShapeSignature,
    supports_dtypes: Vec<DType>,
    tunable: bool,
) -> Candidate {
    Candidate {
        id: CandidateId::derive(name, backend),
        name: name.to_string(),
        backend,
        source_hash: format!("sha256:{name}"),
        requires,
        min_shape,
        max_shape,
        supports_dtypes,
        tunable,
    }
}

pub(crate) fn fresh_capabilities() -> DeviceCaps {
    DeviceCaps {
        capabilities: vec![
            Capability::MetalGpu,
            Capability::Bf16,
            Capability::Fp16,
            Capability::MetalMs3,
        ],
    }
}

pub(crate) fn full_capabilities() -> DeviceCaps {
    DeviceCaps {
        capabilities: vec![
            Capability::MetalGpu,
            Capability::Bf16,
            Capability::Fp16,
            Capability::MetalMs3,
            Capability::Avx512,
        ],
    }
}

// Stable samples yielding exactly the requested p95:
// 20 samples with the value of interest as the upper quintile (sample 19)
// so ceil(0.95 * 20) - 1 = 18 (idx 18 → sample 19) becomes the p95 index.
// Construction: lower 18 samples ≤ lower_p95, sample 18 = p95_value,
// sample 19 = p95_value + 1 to break ties without affecting p95.
pub(crate) fn samples_with_p95(p95: u64) -> Vec<u64> {
    let mut v: Vec<u64> = (0..18).map(|i| p95.saturating_sub(50 + i as u64)).collect();
    v.push(p95);
    v.push(p95 + 1);
    v
}

#[test]
fn selector_runs_are_deterministic_across_sota_operator_families() {
    // Run the deterministic policy twice against the Mamba registry and
    // confirm we get the same chosen id both times.
    let (reg, _, _, id_metal) = recurrent::mamba_registry();
    let d1 = reg.select_with_caps(
        &recurrent::mamba_key(),
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let d2 = reg.select_with_caps(
        &recurrent::mamba_key(),
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    let id1 = d1.selected();
    let id2 = d2.selected();
    assert_eq!(id1, id2, "selector must be deterministic across calls");
    assert_eq!(id1, Some(id_metal));
}

#[test]
fn selector_rejects_when_no_candidate_supports_ternary_dtype() {
    // The Bonsai ternary matmul key demands Int8 + Ternary policy. If a
    // candidate only supports Fp32/Fp16 the dtype filter must reject it.
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 1, 4);
    let scalar = make_candidate(
        "TernaryScalarWrongDtype",
        BackendKind::Reference,
        vec![],
        min,
        max,
        vec![DType::Fp32, DType::Fp16], // no Int8
        false,
    );
    let id = scalar.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    let decision = reg.select_with_caps(
        &bonsai_qwen::bonsai_key(),
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Rejected { rejections, considered } => {
            assert!(considered.contains(&id));
            assert!(rejections.iter().any(|r| matches!(r.reason, RejectionReason::UnsupportedDtype(_))),
                "expected UnsupportedDtype rejection, got {rejections:?}");
        }
        other => panic!("expected Rejected, got {other:?}"),
    }
}

mod attention;
mod attention_sliding_window;
mod bonsai_qwen;
mod builders_integration;
mod diffusion;
mod mod_routing;
mod recurrent;