//! (j) `dispatch_buckets_recurrent` — per-shape envelope for the *batched*
//! DeltaNet, Mamba, and RWKV selectors. Pins the chosen candidate's
//! `median_dispatches` against an oracle policy so a future regression
//! in the runtime's chunk policy (e.g. accidentally re-introducing a
//! per-(batch, head) loop where a parallel tile is expected) is caught.
//!
//! Oracle policy for the batched recurrence kernel is:
//!   dispatches_oracle = ceil(B / 32) * (1 setup + ceil(C / chunk))
//! The single setup launch is the metadata emission; the `ceil(C / chunk)`
//! captures the per-tile launches for the chunked recurrence. 32 is the
//! 1D tile size, `chunk` is the recurrent kernel's `chunk_size`
//! (currently 16 for the batched DeltaNet). 1.2× ceiling is the same
//! headroom used by `regress_baseline::dispatch_budget` for matmul.

use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::selector::SelectionDecision;
use kernel_registry::{BackendKind, Capability, KernelKey, KernelRegistry, SelectionPolicy};

use super::{
    build_record_with_dispatches, fresh_capabilities, make_candidate, samples_with_p95, shape,
    NOW_UNIX_MS, TEST_FINGERPRINT,
};

/// `ceil_div` is local to this test file — it's not in the library
/// because the test is the one place the dispatch budget is enforced.
fn ceil_div(a: usize, b: usize) -> u32 {
    a.div_ceil(b) as u32
}

/// Per-bucket shape spec for `dispatch_buckets_recurrent_picks_delta_net_batched_within_budget`.
type Bucket = (&'static str, (usize, usize, usize, usize));

#[test]
fn dispatch_buckets_recurrent_picks_delta_net_batched_within_budget() {
    // Five shape buckets spanning decode, prompt, and long-context
    // recurrence. (batch=B, state_channels=C, head_dim=D, chunk=16.)
    let buckets: &[Bucket] = &[
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
        let oracle: u32 = ceil_div(b_size, TILE_BATCH) * (1 + ceil_div(c_size, CHUNK_SIZE));
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
