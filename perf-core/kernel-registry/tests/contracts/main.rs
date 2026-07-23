//! Contract tests for the kernel-registry crate.
//!
//! These tests are written FIRST (TDD). They pin the public contract
//! documented in `docs/sessions/20260718-metal-model-runtime/02_SPECIFICATIONS.md`
//! and `04_IMPLEMENTATION_STRATEGY.md`.
//!
//! Conventions:
//! - `device_fingerprint` is always a stable 64-hex string (matches
//!   `sha256("test-device-v1")`).
//! - All candidate names are derived from a stable string so that
//!   `CandidateId` collisions are reproducible.
//! - `now_unix_ms` is a fixed integer (`1_700_000_000_000`) so the tests are
//!   deterministic regardless of wall-clock time.

use kernel_registry::compat::OperatorKind;
use kernel_registry::{
    BackendKind, Candidate, CandidateId, Capability, DType, KernelKey, Measurement,
    QuantizationPolicy, ShapeSignature, TuningRecord,
};

pub(crate) const TEST_DEVICE_FINGERPRINT: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
pub(crate) const NOW_UNIX_MS: u64 = 1_700_000_000_000;

pub(crate) fn shape(m: usize, n: usize, k: usize) -> ShapeSignature {
    ShapeSignature { m, n, k, batch: 1, seq: 1, group: 1 }
}

pub(crate) fn shape_with(
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    seq: usize,
    group: usize,
) -> ShapeSignature {
    ShapeSignature { m, n, k, batch, seq, group }
}

pub(crate) fn key_with(
    operator: OperatorKind,
    fingerprint: &str,
    policy_version: u32,
) -> KernelKey {
    KernelKey {
        operator_kind: operator,
        attention_kind: None,
        shape_signature: shape(64, 64, 64),
        dtype: DType::Fp16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: fingerprint.to_string(),
        policy_version,
    }
}

pub(crate) fn candidate_from(
    name: &str,
    backend: BackendKind,
    requires: Vec<Capability>,
) -> Candidate {
    let id_digest = kernel_registry::fast_hash_bytes(name.as_bytes());
    Candidate {
        id: CandidateId(id_digest),
        name: name.to_string(),
        backend,
        // Stable, deterministic artifact hash for the candidate. Use a
        // fixture value so contract tests do not depend on file IO.
        source_hash: format!("sha256:{name}"),
        requires,
        min_shape: shape(1, 1, 1),
        max_shape: shape_with(4096, 4096, 4096, 64, 4096, 64),
        supports_dtypes: vec![DType::Fp16, DType::Bf16],
        tunable: true,
        engine_name: None,
        properties: std::collections::HashMap::new(),
    }
}

pub(crate) fn measurement(sample: u32, latency_ns: u64) -> Measurement {
    Measurement::with_metadata(sample, latency_ns, None, None, 0)
}

pub(crate) fn tuning_record(
    candidate_id: CandidateId,
    key: KernelKey,
    samples: &[u64],
    expires_at_unix_ms: Option<u64>,
    compiler: &str,
    compiler_version: &str,
    source_revision: &str,
) -> TuningRecord {
    let measurements: Vec<Measurement> = samples
        .iter()
        .enumerate()
        .map(|(i, &latency)| measurement(i as u32, latency))
        .collect();
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];
    let p95_idx =
        ((sorted.len() as f64 * 0.95).ceil() as usize).saturating_sub(1).min(sorted.len() - 1);
    let p99_idx =
        ((sorted.len() as f64 * 0.99).ceil() as usize).saturating_sub(1).min(sorted.len() - 1);
    let p95 = sorted[p95_idx];
    let p99 = sorted[p99_idx];
    let mean = sorted.iter().sum::<u64>() as f64 / sorted.len() as f64;
    let variance = sorted
        .iter()
        .map(|&x| ((x as f64) - mean).powi(2))
        .sum::<f64>()
        / sorted.len() as f64;
    TuningRecord {
        candidate_id,
        key,
        measurements,
        median_ns: median,
        p95_ns: p95,
        p99_ns: p99,
        variance_ns2: variance as u64,
        median_energy_j: None,
        median_dispatches: None,
        samples: samples.len(),
        warmup_discarded: 3,
        compiler: compiler.to_string(),
        compiler_version: compiler_version.to_string(),
        captured_at_unix_ms: NOW_UNIX_MS,
        source_revision: source_revision.to_string(),
        expires_at_unix_ms,
        quality: None,
    }
}

mod bounded_tuner;
mod candidate;
mod kernel_key;
mod policies;
mod selector;
mod serde_;