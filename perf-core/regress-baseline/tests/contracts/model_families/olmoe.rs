//! OLMoE-1B-7B per-stage end-to-end conformance trace (turn-14).
//!
//! Generalizes the Qwen-MoE v2 trace to the Open-LLM-MoE topology:
//! `num_experts=64, top_k=8, shared_experts=1`. The shape used here
//! is intentionally small (4 tokens, hidden=4) so the scalar reference
//! path matches the tiled kernels byte-for-byte while still exercising
//! the 64-expert router + 8-way top-k dispatch.

use serde_json::json;

use super::super::{
    assert_close_envelope, checked_in_baselines_dir, BaselineRecorder, VerifyResult,
};

const OLMOE_NUM_TOKENS: usize = 4;
const OLMOE_NUM_EXPERTS: usize = 64;
const OLMOE_TOP_K: usize = 8;
const OLMOE_HIDDEN: usize = 4;
const OLMOE_K: usize = 4;
const OLMOE_CAPACITY_FACTOR: f32 = 1.5;
const OLMOE_SEED: u64 = 0xCAFE_BABE_DEAD_BEEF_u64;

/// Deterministic OLMoE per-token activation seed. Mirrors the salts
/// used in `tests/qwen_bonsai/olmoe_moe.rs` so the test-suite and the
/// contract test produce identical buffers.
fn olmoe_det(n: usize, salt: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut lcg: u64 = OLMOE_SEED ^ salt;
    for _ in 0..n {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push(((lcg >> 33) as f32) / (u32::MAX as f32) - 0.5);
    }
    out
}

/// Compute the full OLMoE end-to-end output envelope. Mirrors the
/// Qwen-MoE v2 helper exactly: 4 inputs → 5 outputs (top picks +
/// router logits + shared/routed/reduced/writeback buffers).
#[allow(clippy::too_many_lines, clippy::type_complexity)]
fn compute_olmoe_moe_end_to_end_output() -> (Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    use model_kernels::moe::{
        coalesced_writeback, grouped_gemm_tiled, moe_dispatch, router_topk, shared_expert,
        stage_expert_outputs, weighted_reduce_tiled,
    };
    let n_t = OLMOE_NUM_TOKENS;
    let n_e = OLMOE_NUM_EXPERTS;
    let top_k = OLMOE_TOP_K;
    let h = OLMOE_HIDDEN;
    let k = OLMOE_K;
    // Router logits: 64 floats per token, seeded per-token via LCG.
    let router_logits: Vec<f32> = (0..n_t)
        .flat_map(|t| olmoe_det(n_e, 0xE0_01 + t as u64))
        .collect();
    let mut assignments: Vec<(usize, f32)> = Vec::with_capacity(n_t);
    let mut picks_per_token: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n_t);
    let mut top_picks: Vec<usize> = Vec::with_capacity(n_t);
    for t in 0..n_t {
        let picks = router_topk(&router_logits[t * n_e..(t + 1) * n_e], n_e, top_k, 0)
            .expect("router_topk must accept well-formed inputs");
        assignments.push(picks[0]);
        top_picks.push(picks[0].0);
        picks_per_token.push(picks);
    }
    let plan = moe_dispatch(
        &(0..n_t).collect::<Vec<_>>(),
        &assignments,
        n_e,
        OLMOE_CAPACITY_FACTOR,
    )
    .expect("dispatch must accept well-formed inputs");

    // Activations + shared-expert weight + per-expert routed weights.
    let a = olmoe_det(n_t * k, 0xA_CE);
    let w = olmoe_det(h * h, 0xB_EE);
    let b: Vec<f32> = (0..n_e)
        .flat_map(|e| olmoe_det(k * h, 0xB0_E0 + e as u64))
        .collect();

    // Shared-expert projection + routed-GEMM (tiled) into `[n_t, h]`.
    let mut shared_out = vec![0.0f32; n_t * h];
    shared_expert(&a, &w, &mut shared_out).expect("shared_expert must accept well-formed inputs");
    let mut routed = vec![0.0f32; n_t * h];
    grouped_gemm_tiled(&a, &b, &plan.expert_buckets, 0, k, h, &mut routed)
        .expect("grouped_gemm_tiled must accept well-formed inputs");

    // Build `[n_t, top_k, h]` expert-outs + per-token top-k weights:
    // slot 0 from the routed GEMM, slots 1..top_k from scalar matmuls
    // over the remaining picks (matches the Qwen-MoE v2 pattern).
    let mut expert_outs = vec![0.0f32; n_t * top_k * h];
    let mut weights = vec![0.0f32; n_t * top_k];
    for (t, picks) in picks_per_token.iter().enumerate() {
        expert_outs[(t * top_k) * h..(t * top_k + 1) * h]
            .copy_from_slice(&routed[t * h..(t + 1) * h]);
        for (slot, &(expert, _w)) in picks.iter().enumerate().skip(1) {
            let b_off = expert * k * h;
            for j in 0..h {
                let acc: f32 = (0..k).map(|kk| a[t * k + kk] * b[b_off + kk * h + j]).sum();
                expert_outs[(t * top_k + slot) * h + j] = acc;
            }
        }
        for (e_idx, &(_, w)) in picks.iter().enumerate() {
            weights[t * top_k + e_idx] = w;
        }
    }

    // Tiled weighted-reduce + stage + coalesced writeback.
    let mut reduced_out = vec![0.0f32; n_t * h];
    weighted_reduce_tiled(&expert_outs, &weights, top_k, h, &mut reduced_out)
        .expect("weighted_reduce_tiled must accept well-formed inputs");
    let stage = stage_expert_outputs(&routed, &plan, h)
        .expect("stage_expert_outputs must accept well-formed inputs");
    let mut writeback_out = vec![0.0f32; n_t * h];
    coalesced_writeback(&stage, n_t, h, &mut writeback_out)
        .expect("coalesced_writeback must accept well-formed inputs");

    (
        top_picks,
        router_logits,
        shared_out,
        reduced_out,
        writeback_out,
    )
}

/// `olmoe_moe_end_to_end` baseline round-trip.
#[test]
fn olmoe_moe_end_to_end_baseline_round_trip() {
    let (top_picks, router_logits, shared_out, reduced_out, writeback_out) =
        compute_olmoe_moe_end_to_end_output();
    let to_f64 = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| x as f64).collect() };
    let capacity =
        (OLMOE_CAPACITY_FACTOR * OLMOE_NUM_TOKENS as f32 / OLMOE_NUM_EXPERTS as f32).ceil() as u64;
    let inputs = json!({
        "kernel": "olmoe_moe_end_to_end",
        "num_tokens": OLMOE_NUM_TOKENS,
        "num_experts": OLMOE_NUM_EXPERTS,
        "top_k": OLMOE_TOP_K,
        "hidden": OLMOE_HIDDEN,
        "k": OLMOE_K,
        "capacity_factor": OLMOE_CAPACITY_FACTOR,
        "seed": OLMOE_SEED,
    });
    let expected_out = json!({
        "kernel": "olmoe_moe_end_to_end",
        "num_tokens": OLMOE_NUM_TOKENS,
        "num_experts": OLMOE_NUM_EXPERTS,
        "top_k": OLMOE_TOP_K,
        "hidden": OLMOE_HIDDEN,
        "k": OLMOE_K,
        "capacity_factor": OLMOE_CAPACITY_FACTOR,
        "capacity": capacity,
        "top_picks": top_picks.iter().map(|&x| x as u64).collect::<Vec<u64>>(),
        "router_logits": to_f64(&router_logits),
        "shared_out": to_f64(&shared_out),
        "reduced_out": to_f64(&reduced_out),
        "writeback_out": to_f64(&writeback_out),
    });
    let recorder = BaselineRecorder::new(checked_in_baselines_dir());
    let file = recorder.load().expect("load checked-in baselines");
    assert!(
        file.baselines.contains_key("olmoe_moe_end_to_end"),
        "checked-in baselines must contain olmoe_moe_end_to_end entry"
    );
    let recorded = file.baselines["olmoe_moe_end_to_end"].clone();
    assert_eq!(
        BaselineRecorder::hash_inputs(&inputs),
        recorded.input_hash,
        "input_hash drift: re-record the olmoe_moe_end_to_end entry"
    );
    assert_close_envelope(&recorded.output, &expected_out);
    let r = recorder
        .verify("olmoe_moe_end_to_end", &inputs, &recorded.output)
        .expect("verify olmoe_moe_end_to_end");
    assert_eq!(r, VerifyResult::Ok);
}
