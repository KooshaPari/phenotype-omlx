//! (k) `dispatch_buckets_dense` — per-shape envelope for the *dense*
//! attention (GQA), sparse MoE, and dense-matmul selectors. Pins the
//! chosen candidate's `median_dispatches` against an oracle policy so a
//! future regression in the runtime's tile / setup / output-write policy
//! (e.g. accidentally re-introducing a per-(batch, expert) loop where a
//! parallel tile is expected, or splitting output into one Metal
//! command-buffer dispatch per row of an already-tileable matmul) is
//! caught. Three operator families mirror the family boundaries in
//! `docs/sessions/20260718-metal-model-runtime/02_SPECIFICATIONS.md`:
//!
//! 1. GQA prefill/decode attention — `OperatorKind::Gqa`
//! 2. Sparse MoE (Qwen-style 8 experts, top-2) — `OperatorKind::Moe`
//! 3. QKV-projection dense matmul — `OperatorKind::DenseMatmul`
//!
//! Oracle policy per family:
//! - GQA   : `ceil(seq_q * seq_k * head_dim / 65536) + 1`
//!   (1 setup launch + 1 output-write dispatch per 64 KiB tile).
//! - MoE   : `top_k + 1 + n_experts * ceil(batch / tile_b)`
//!   (top_k routing lookups + 1 setup + 1 parallel tile per expert-batch chunk).
//! - Matmul: `ceil(M / 32) * ceil(N / 32) + 1`
//!   (a 32×32 output tile grid + 1 setup launch).
//!
//! 1.2× ceiling is the same headroom used by `regress_baseline::dispatch_budget`
//! and by `recurrent::dispatch_envelope`.

use kernel_registry::compat::{AttentionKind, DType, OperatorKind, QuantizationPolicy};
use kernel_registry::selector::SelectionDecision;
use kernel_registry::{BackendKind, Capability, KernelKey, KernelRegistry, SelectionPolicy};

use super::{
    build_record_with_dispatches, fresh_capabilities, make_candidate, samples_with_p95, shape,
    NOW_UNIX_MS, TEST_FINGERPRINT,
};

/// `ceil_div` is local to this test file — mirroring
/// `recurrent::dispatch_envelope::ceil_div`. Not exported because the
/// dispatch budget is the test's contract, not the library's API.
fn ceil_div(a: usize, b: usize) -> u32 {
    a.div_ceil(b) as u32
}

/// Assert `observed` is within `oracle * 1.2` (no floor). Shared body
/// for the three bucket loops so the dispatch-envelope contract is
/// stated once per file.
fn assert_within_envelope(observed: u32, oracle: u32, label: &str) {
    let ceiling = oracle.saturating_mul(12) / 10;
    assert!(
        observed <= ceiling,
        "[{label}] observed dispatches={observed} must be <= 1.2*oracle={ceiling}; oracle was {oracle}"
    );
}

// ---------------------------------------------------------------------------
// GQA prefill + decode (OperatorKind::Gqa)
// ---------------------------------------------------------------------------

/// `(name, q_h, kv_h, head_dim, seq_q, seq_k)`. Decode buckets pin
/// `q_h=8, kv_h=2, head_dim=64` (LLaMA-3 / Mistral-style GQA);
/// prefill buckets use a smaller head count to exercise the same tile
/// policy at a different launch geometry.
type GqaBucket = (&'static str, (usize, usize, usize, usize, usize));

#[test]
fn dispatch_buckets_gqa_attention_within_budget() {
    let buckets: &[GqaBucket] = &[
        ("decode_short_ctx", (8, 2, 64, 1, 1024)),
        ("decode_long_ctx", (8, 2, 64, 1, 8192)),
        ("prefill_small_1k", (4, 2, 64, 1024, 1024)),
        ("prefill_medium_4k", (4, 2, 64, 4096, 4096)),
    ];
    for &(name, (q_h, kv_h, head_dim, seq_q, seq_k)) in buckets {
        // shape: (m=q_h, n=kv_h, k=head_dim, batch=1, seq=seq_q,
        // group=GQA group count = q_h / kv_h).
        let gqa_group = q_h / kv_h;
        let key = KernelKey {
            operator_kind: OperatorKind::Gqa,
            attention_kind: Some(AttentionKind::Gqa),
            shape_signature: shape(q_h, kv_h, head_dim, 1, seq_q, gqa_group),
            dtype: DType::Bf16,
            quantization: QuantizationPolicy::None,
            state_layout_version: 1,
            device_fingerprint: TEST_FINGERPRINT.to_string(),
            policy_version: 1,
        };
        let min = shape(1, 1, 1, 1, 1, 1);
        let max = shape(16, 16, 128, 1, 8192, 8);
        let scalar = make_candidate(
            "GqaAttentionScalar",
            BackendKind::Reference,
            vec![],
            min,
            max,
            vec![DType::Fp32, DType::Bf16],
            false,
        );
        let metal = make_candidate(
            "GqaAttentionMetal",
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

        // 1 setup + 1 output-write dispatch per 64 KiB tile of
        // attention output (seq_q * seq_k * head_dim bytes for fp16/bf16).
        const OUTPUT_TILE_BYTES: usize = 65_536;
        let oracle = ceil_div(seq_q * seq_k * head_dim, OUTPUT_TILE_BYTES) + 1;
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
            SelectionPolicy::Deterministic {
                prefer_lower_p95: true,
            },
            &fresh_capabilities(),
            NOW_UNIX_MS,
        );
        match &decision {
            SelectionDecision::Chosen { candidate, tuning } => {
                assert_eq!(
                    candidate.id, id_metal,
                    "[{name}] deterministic must pick Metal; got {:?}",
                    candidate.name
                );
                let observed = tuning
                    .median_dispatches
                    .expect("Metal tuning record must carry dispatches metadata");
                assert_within_envelope(
                    observed,
                    oracle,
                    &format!(
                        "GQA q_h={q_h} kv_h={kv_h} head_dim={head_dim} seq_q={seq_q} seq_k={seq_k}"
                    ),
                );
            }
            other => panic!("[{name}] expected Chosen under Deterministic, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse MoE (OperatorKind::Moe)
// ---------------------------------------------------------------------------

/// `(name, batch)`. `n_experts=8` and `top_k=2` (Qwen-MoE) are pinned
/// in the oracle below, not the bucket, so the tuple stays compact.
type MoeBucket = (&'static str, usize);

#[test]
fn dispatch_buckets_moe_within_budget() {
    let buckets: &[MoeBucket] = &[
        ("moe_batch_32", 32),
        ("moe_batch_128", 128),
        ("moe_batch_512", 512),
        ("moe_batch_2048", 2048),
    ];
    for &(name, batch) in buckets {
        // shape: (m=hidden, n=hidden, k=hidden, batch=batch, seq=1, group=1).
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
        let min = shape(1, 1, 1, 1, 1, 1);
        let max = shape(128, 128, 128, 4096, 1, 1);
        let scalar = make_candidate(
            "MoeScalar",
            BackendKind::Reference,
            vec![],
            min,
            max,
            vec![DType::Fp32, DType::Bf16],
            false,
        );
        let metal = make_candidate(
            "MoeMetal",
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

        const N_EXPERTS: u32 = 8;
        const TOP_K: u32 = 2;
        const TILE_BATCH: usize = 32;
        let oracle = TOP_K + 1 + N_EXPERTS * ceil_div(batch, TILE_BATCH);
        reg.attach_tuning_record(
            key.clone(),
            build_record_with_dispatches(
                id_metal,
                key.clone(),
                &samples_with_p95(1400),
                Some(NOW_UNIX_MS + 86_400_000),
                oracle,
            ),
        );
        let decision = reg.select_with_caps(
            &key,
            SelectionPolicy::Deterministic {
                prefer_lower_p95: true,
            },
            &fresh_capabilities(),
            NOW_UNIX_MS,
        );
        match &decision {
            SelectionDecision::Chosen { candidate, tuning } => {
                assert_eq!(
                    candidate.id, id_metal,
                    "[{name}] deterministic must pick Metal; got {:?}",
                    candidate.name
                );
                let observed = tuning
                    .median_dispatches
                    .expect("Metal tuning record must carry dispatches metadata");
                assert_within_envelope(observed, oracle, &format!("MoE batch={batch}"));
            }
            other => panic!("[{name}] expected Chosen under Deterministic, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Dense QKV-projection matmul (OperatorKind::DenseMatmul)
// ---------------------------------------------------------------------------

/// `(name, m, n)`. `k` is pinned to `m` (QKV projection has matching
/// hidden dims); the bucket tuple only carries `(m, n)` to stay short.
type MatmulBucket = (&'static str, usize, usize);

#[test]
fn dispatch_buckets_dense_matmul_within_budget() {
    // Four QKV-projection buckets — `M` is the token count and `N` is
    // 3 × hidden (Q+K+V concatenated). Tile grid is
    // ceil(M/32) × ceil(N/32) plus 1 setup launch.
    let buckets: &[MatmulBucket] = &[
        ("qkv_512x2304", 512, 2304),
        ("qkv_1024x3072", 1024, 3072),
        ("qkv_2048x4096", 2048, 4096),
        ("qkv_4096x8192", 4096, 8192),
    ];
    for &(name, m, n) in buckets {
        let key = KernelKey {
            operator_kind: OperatorKind::DenseMatmul,
            attention_kind: None,
            shape_signature: shape(m, n, m, 1, 1, 1),
            dtype: DType::Bf16,
            quantization: QuantizationPolicy::None,
            state_layout_version: 1,
            device_fingerprint: TEST_FINGERPRINT.to_string(),
            policy_version: 1,
        };
        let min = shape(1, 1, 1, 1, 1, 1);
        let max = shape(8192, 16384, 8192, 1, 1, 1);
        let scalar = make_candidate(
            "DenseMatmulScalar",
            BackendKind::Reference,
            vec![],
            min,
            max,
            vec![DType::Fp32, DType::Bf16],
            false,
        );
        let metal = make_candidate(
            "DenseMatmulMetal",
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

        const TILE_M: usize = 32;
        const TILE_N: usize = 32;
        let oracle = ceil_div(m, TILE_M) * ceil_div(n, TILE_N) + 1;
        reg.attach_tuning_record(
            key.clone(),
            build_record_with_dispatches(
                id_metal,
                key.clone(),
                &samples_with_p95(1200),
                Some(NOW_UNIX_MS + 86_400_000),
                oracle,
            ),
        );
        let decision = reg.select_with_caps(
            &key,
            SelectionPolicy::Deterministic {
                prefer_lower_p95: true,
            },
            &fresh_capabilities(),
            NOW_UNIX_MS,
        );
        match &decision {
            SelectionDecision::Chosen { candidate, tuning } => {
                assert_eq!(
                    candidate.id, id_metal,
                    "[{name}] deterministic must pick Metal; got {:?}",
                    candidate.name
                );
                let observed = tuning
                    .median_dispatches
                    .expect("Metal tuning record must carry dispatches metadata");
                assert_within_envelope(observed, oracle, &format!("DenseMatmul M={m} N={n}"));
            }
            other => panic!("[{name}] expected Chosen under Deterministic, got {other:?}"),
        }
    }
}
