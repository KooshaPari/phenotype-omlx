//! Each test pins down a different acceptance-matrix row from
//! `02_SPECIFICATIONS.md`:
//!
//! | Test | Family / row | Trace shape |
//! |---|---|---|
//! | `cca_block_baseline_round_trip` | ZAYA — CCA & compact nonlinear expert path | Three block summaries + query → softmax-weighted output |
//! | `mla_cache_baseline_round_trip` | DeepSeek — MLA, routed experts, proposal/MTP | Four `compressed_kv` + `k_rope` cache entries + query |
//! | `qwen_deltanet_moe_end_to_end_baseline_round_trip` | Qwen agentic — long-context decode, GQA or DeltaNet state, sparse MoE | DeltaNet recurrence over a four-token chunk → top-k sparse-MoE |
//! | `qwen_moe_end_to_end_v2_baseline_round_trip` | Qwen agentic — sparse-MoE per-stage composition (tiled GEMM + tiled reduce + writeback) | Router → dispatch → grouped_gemm_tiled → weighted_reduce_tiled → stage_expert_outputs + coalesced_writeback over `num_tokens=4, num_experts=3, top_k=2, hidden=4, k=4, capacity_factor=2.0` |
//! | `moe_writeback_2x8_baseline_round_trip` | Qwen agentic — turn-12 dispatch-aware DRAM writeback | top_k=2, 8 tokens, 3 experts, hidden=4, LCG-seeded expert outputs |
//! | `olmoe_moe_end_to_end_baseline_round_trip` | OLMoE-1B-7B — turn-14 generalized per-stage composition (64 experts, top_k=8, 1 shared expert) | Same per-stage chain as Qwen-MoE v2 with `num_tokens=4, num_experts=64, top_k=8, hidden=4, k=4, capacity_factor=1.5` |
//!
//! The per-family `seed` is part of the inputs so the input hash
//! distinguishes each trace from any other shape that happens to share
//! the same field names.

use super::{assert_close_envelope, checked_in_baselines_dir, BaselineRecorder, VerifyResult};
use serde_json::json;

#[test]
fn cca_block_baseline_round_trip() {
    let recorder = BaselineRecorder::new(checked_in_baselines_dir());
    let file = recorder.load().expect("load checked-in baselines");
    assert!(file.baselines.contains_key("cca_block_attend"));

    let inputs = json!({
        "kernel": "cca_block_attend",
        "head_dim": 4,
        "block_count": 3,
        "blocks": [
            {
                "indices": [0, 1, 2, 3],
                "summary": [0.5_f64, -0.25, 1.0, 0.0],
                "scale": 1.0_f64
            },
            {
                "indices": [4, 5],
                "summary": [0.0_f64, 1.0, -0.5, 0.25],
                "scale": 0.5_f64
            },
            {
                "indices": [6, 7, 8, 9, 10, 11],
                "summary": [0.75_f64, 0.75, -0.1, -0.1],
                "scale": 1.25_f64
            }
        ],
        "query": [0.1_f64, 0.2, 0.3, 0.4],
        "seed": 0xCAFE_BABE_u64,
    });

    let recorded = file.baselines["cca_block_attend"].clone();
    assert_eq!(
        BaselineRecorder::hash_inputs(&inputs),
        recorded.input_hash,
        "cca_block_attend input_hash drift",
    );

    let expected_out = json!({
        "kernel": "cca_block_attend",
        "head_dim": 4,
        "block_count": 3,
        "out": [
            2.241303051607532_f64,
            1.7212455131881892_f64,
            0.9867472524778097_f64,
            -0.05199278968404647_f64,
        ],
    });
    assert_close_envelope(&recorded.output, &expected_out);

    let r = recorder
        .verify("cca_block_attend", &inputs, &recorded.output)
        .expect("verify cca_block_attend");
    assert_eq!(r, VerifyResult::Ok);
}

#[test]
fn mla_cache_baseline_round_trip() {
    let recorder = BaselineRecorder::new(checked_in_baselines_dir());
    let file = recorder.load().expect("load checked-in baselines");
    assert!(file.baselines.contains_key("mla_cache_attend"));

    let inputs = json!({
        "kernel": "mla_cache_attend",
        "d_latent": 4,
        "d_rope": 2,
        "seq_k": 4,
        "cache": [
            {
                "compressed_kv": [0.5_f64, -0.25, 1.0, 0.0],
                "k_rope": [0.1_f64, 0.2]
            },
            {
                "compressed_kv": [0.0_f64, 1.0, -0.5, 0.25],
                "k_rope": [-0.1_f64, 0.3]
            },
            {
                "compressed_kv": [0.75_f64, 0.75, -0.1, -0.1],
                "k_rope": [0.4_f64, -0.4]
            },
            {
                "compressed_kv": [0.2_f64, -0.2, 0.8, 0.6],
                "k_rope": [0.0_f64, 0.5]
            }
        ],
        "q_latent": [0.1_f64, 0.2, 0.3, 0.4],
        "q_rope": [0.5_f64, -0.5],
        "seed": 0xCAFE_BABE_u64,
    });

    let recorded = file.baselines["mla_cache_attend"].clone();
    assert_eq!(
        BaselineRecorder::hash_inputs(&inputs),
        recorded.input_hash,
        "mla_cache_attend input_hash drift",
    );

    let expected_out = json!({
        "kernel": "mla_cache_attend",
        "d_latent": 4,
        "d_rope": 2,
        "seq_k": 4,
        "out": [
            0.4212736878640673_f64,
            0.3243108994605529_f64,
            0.31111078283933086_f64,
            0.15425821296383402_f64,
        ],
    });
    assert_close_envelope(&recorded.output, &expected_out);

    let r = recorder
        .verify("mla_cache_attend", &inputs, &recorded.output)
        .expect("verify mla_cache_attend");
    assert_eq!(r, VerifyResult::Ok);
}

#[test]
fn qwen_deltanet_moe_end_to_end_baseline_round_trip() {
    let recorder = BaselineRecorder::new(checked_in_baselines_dir());
    let file = recorder.load().expect("load checked-in baselines");
    assert!(file
        .baselines
        .contains_key("qwen_deltanet_moe_end_to_end"));

    let inputs = json!({
        "kernel": "qwen_deltanet_moe_end_to_end",
        "head_dim": 2,
        "chunk_size": 4,
        "num_experts": 3,
        "top_k": 2,
        "capacity_factor": 1.5_f64,
        "seed_deltanet": 0xCAFE_BABE_u64,
        "seed_moe": 0xDEADBEEF_u64,
        "s_state": [
            [0.06991786752596547_f64, 0.03830528700763702_f64],
            [0.10781900698602688_f64, 0.4324679840513347_f64],
        ],
        "q": [
            [0.04246751020139372_f64, 0.23250469920865613_f64],
            [-0.11554406685392743_f64, 0.0122591474763446_f64],
            [-0.3425595539522917_f64, 0.05847873316181851_f64],
            [0.3132157016514764_f64, 0.3660663285138862_f64],
        ],
        "k": [
            [-0.44447468408744906_f64, -0.22367991373788565_f64],
            [0.19145389257663536_f64, -0.2094852785324945_f64],
            [-0.48348651593935527_f64, -0.3930602376598259_f64],
            [0.40759846942788025_f64, -0.15746015830490268_f64],
        ],
        "v": [
            [-0.37016338951183403_f64, -0.21605705313476337_f64],
            [-0.3663536859987412_f64, 0.1304810916468312_f64],
            [0.03904277585934779_f64, -0.3234472624437308_f64],
            [-0.05480652918228729_f64, 0.1619786984142897_f64],
        ],
        "beta": [
            0.2964370854789305_f64,
            0.43967315887807096_f64,
            0.4646655497572698_f64,
            0.4241783206411222_f64,
        ],
        "expert_weights": [
            [
                [-0.4072812395386209_f64, 0.1947473234742071_f64],
                [-0.21147098911193407_f64, 0.2517428778224292_f64],
            ],
            [
                [0.3136655956474912_f64, 0.22561408446039677_f64],
                [0.28389310702125237_f64, 0.173980483162687_f64],
            ],
            [
                [-0.13600417514424673_f64, -0.3498030810500954_f64],
                [0.02641908556073458_f64, 0.22842827701744894_f64],
            ],
        ],
        "router_logits": [
            -0.8440545527627564_f64,
            0.0954977262077159_f64,
            0.3527071863988631_f64,
        ],
    });

    let recorded = file.baselines["qwen_deltanet_moe_end_to_end"].clone();
    assert_eq!(
        BaselineRecorder::hash_inputs(&inputs),
        recorded.input_hash,
        "qwen_deltanet_moe_end_to_end input_hash drift",
    );

    let expected_out = json!({
        "kernel": "qwen_deltanet_moe_end_to_end",
        "head_dim": 2,
        "chunk_size": 4,
        "num_experts": 3,
        "top_k": 2,
        "capacity_factor": 1.5_f64,
        "capacity": 2,
        "top_picks": [2, 1],
        "deltanet_outputs": [
            [0.03498257255832317_f64, 0.10367783352946564_f64],
            [-0.007415759375221491_f64, -0.002294202351932063_f64],
            [-0.011464295507259211_f64, -0.002774929016837402_f64],
            [0.06961694677808955_f64, 0.189481050198067_f64],
        ],
        "moe_shared_out": [
            0.00016106075082136448_f64,
            0.04923174960616833_f64,
        ],
        "moe_reduced_out": [
            -0.00498551897116146_f64,
            0.015134984270487883_f64,
        ],
    });
    assert_close_envelope(&recorded.output, &expected_out);

    let r = recorder
        .verify("qwen_deltanet_moe_end_to_end", &inputs, &recorded.output)
        .expect("verify qwen_deltanet_moe_end_to_end");
    assert_eq!(r, VerifyResult::Ok);
}

/// Compute the canonical MoE writeback output for the
/// `moe_writeback_2x8` baseline. The trace shape is:
///
/// - `num_tokens = 8`, `num_experts = 3`, `top_k = 2`, `hidden = 4`.
/// - Expert outputs `[num_tokens, top_k, hidden]` are LCG-seeded
///   with `seed = 0x57A6_BA11` (the same salt the bench envelope uses).
/// - Per top-k slot, the dispatcher is called once with a
///   round-robin assignment: token `t` is routed to expert
///   `(t + k_slot * stride) % num_experts` where `stride = num_experts / top_k = 1`.
/// - Each slot runs `stage_expert_outputs` + `coalesced_writeback`
///   against a zeroed `[num_tokens, hidden]` residual buffer.
/// - The baseline `out` is the **first row** of the residual buffer
///   (4 floats for `hidden = 4`) — the canonical oracle that the
///   persistence test pins down.
///
/// This helper exists so the trace test (which lives in this same
/// file) and the bench envelope (which lives in
/// `model-kernels/tests/grouped_gemm_bench.rs`) share one source of
/// truth for the canonical inputs. The output is fully deterministic
/// — any drift in `expert_outs`, the round-robin stride, the
/// dispatcher's tie-break, or the writeback kernel will surface as a
/// `verify_fails_on_different_input_hash` style failure.
fn compute_moe_writeback_2x8_output() -> Vec<f32> {
    use model_kernels::common::Lcg;
    use model_kernels::moe::{coalesced_writeback, moe_dispatch, stage_expert_outputs};

    let num_tokens: usize = 8;
    let num_experts: usize = 3;
    let top_k: usize = 2;
    let hidden: usize = 4;

    let mut rng = Lcg::new(0x57A6_BA11);
    let expert_outs: Vec<f32> = (0..num_tokens * top_k * hidden)
        .map(|_| rng.next_signed())
        .collect();

    let stride = (num_experts / top_k).max(1);
    let token_indices: Vec<usize> = (0..num_tokens).collect();
    let per_slot_assignments: Vec<Vec<(usize, f32)>> = (0..top_k)
        .map(|k_slot| {
            (0..num_tokens)
                .map(|t| ((t + k_slot * stride) % num_experts, 1.0))
                .collect()
        })
        .collect();

    let mut out = vec![0.0f32; num_tokens * hidden];
    for (k_slot, assignments) in per_slot_assignments.iter().enumerate() {
        let plan = moe_dispatch(&token_indices, assignments, num_experts, 2.0)
            .expect("dispatch must accept well-formed inputs");
        let mut slot_outs = Vec::with_capacity(num_tokens * hidden);
        for t in 0..num_tokens {
            let start = (t * top_k + k_slot) * hidden;
            slot_outs.extend_from_slice(&expert_outs[start..start + hidden]);
        }
        let stage = stage_expert_outputs(&slot_outs, &plan, hidden)
            .expect("stage must accept well-formed inputs");
        coalesced_writeback(&stage, num_tokens, hidden, &mut out)
            .expect("writeback must accept well-formed inputs");
    }

    out[0..hidden].to_vec()
}

/// MoE writeback baseline round-trip. The `inputs` JSON carries
/// every field that influences `BaselineRecorder::hash_inputs` so
/// the persisted `input_hash` reproduces byte-for-byte. The
/// `expected_out` is built by calling [`compute_moe_writeback_2x8_output`]
/// at test time (so it always reflects the current kernel's
/// behaviour) and then committed to `baselines.json` via a
/// byte-equality pin against the recorded entry.
#[test]
fn moe_writeback_2x8_baseline_round_trip() {
    let out = compute_moe_writeback_2x8_output();
    let inputs = json!({
        "kernel": "moe_writeback",
        "num_tokens": 8,
        "num_experts": 3,
        "top_k": 2,
        "hidden": 4,
        "seed": 0x57A6_BA11_u64,
    });

    let recorder = BaselineRecorder::new(checked_in_baselines_dir());
    let file = recorder.load().expect("load checked-in baselines");
    assert!(
        file.baselines.contains_key("moe_writeback_2x8"),
        "checked-in baselines must contain moe_writeback_2x8 entry"
    );

    let recorded = file.baselines["moe_writeback_2x8"].clone();
    assert_eq!(
        BaselineRecorder::hash_inputs(&inputs),
        recorded.input_hash,
        "moe_writeback_2x8 input_hash drift",
    );

    let out_json: Vec<f64> = out.iter().map(|&x| x as f64).collect();
    let expected_out = json!({
        "kernel": "moe_writeback",
        "num_tokens": 8,
        "num_experts": 3,
        "top_k": 2,
        "hidden": 4,
        "out": out_json,
    });
    assert_close_envelope(&recorded.output, &expected_out);

    let r = recorder
        .verify("moe_writeback_2x8", &inputs, &recorded.output)
        .expect("verify moe_writeback_2x8");
    assert_eq!(r, VerifyResult::Ok);
}

// =========================================================================
// qwen_moe_end_to_end_v2 — sparse-MoE per-stage composition (tiled GEMM +
// tiled reduce + writeback). Mirrors `model-kernels/tests/qwen_bonsai/qwen_moe_v2.rs`.

const QWEN_MOE_V2_NUM_TOKENS: usize = 4;
const QWEN_MOE_V2_NUM_EXPERTS: usize = 3;
const QWEN_MOE_V2_TOP_K: usize = 2;
const QWEN_MOE_V2_HIDDEN: usize = 4;
const QWEN_MOE_V2_K: usize = 4;
const QWEN_MOE_V2_CAPACITY_FACTOR: f32 = 2.0;
const QWEN_MOE_V2_SEED: u64 = 0xCAFE_BABE_DEAD_BEEF;

fn qwen_moe_v2_det(n: usize, salt: u64) -> Vec<f32> {
    (0..n).map(|_| model_kernels::common::Lcg::new(QWEN_MOE_V2_SEED ^ salt).next_signed()).collect()
}

/// Returns `(top_picks, router_logits, shared_out, reduced_out,
/// writeback_out)` for the `qwen_moe_end_to_end_v2` pipeline. Same
/// pipeline + salts as the integration test in
/// `model-kernels/tests/qwen_bonsai/qwen_moe_v2.rs`.
#[allow(clippy::type_complexity)]
fn compute_qwen_moe_end_to_end_v2_output()
-> (Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    use model_kernels::moe::{
        coalesced_writeback, grouped_gemm_tiled, moe_dispatch, router_topk, shared_expert,
        stage_expert_outputs, weighted_reduce_tiled,
    };
    let n_t = QWEN_MOE_V2_NUM_TOKENS;
    let n_e = QWEN_MOE_V2_NUM_EXPERTS;
    let top_k = QWEN_MOE_V2_TOP_K;
    let h = QWEN_MOE_V2_HIDDEN;
    let k = QWEN_MOE_V2_K;

    // Router logits (per-token salt) + per-token top-k picks.
    let router_logits: Vec<f32> = (0..n_t)
        .flat_map(|t| qwen_moe_v2_det(n_e, 0xE0_01 + t as u64))
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
    let plan = moe_dispatch(&(0..n_t).collect::<Vec<_>>(), &assignments, n_e, QWEN_MOE_V2_CAPACITY_FACTOR)
        .expect("dispatch must accept well-formed inputs");

    // Activations and weights — same salts as the integration test.
    let a = qwen_moe_v2_det(n_t * k, 0xA_CE);
    let w = qwen_moe_v2_det(h * h, 0xB_EE);
    let b: Vec<f32> = (0..n_e).flat_map(|e| qwen_moe_v2_det(k * h, 0xB0_E0 + e as u64)).collect();

    // Shared-expert projection + routed-GEMM (tiled) into `[n_t, h]`.
    let mut shared_out = vec![0.0f32; n_t * h];
    shared_expert(&a, &w, &mut shared_out).expect("shared_expert must accept well-formed inputs");
    let mut routed = vec![0.0f32; n_t * h];
    grouped_gemm_tiled(&a, &b, &plan.expert_buckets, 0, k, h, &mut routed)
        .expect("grouped_gemm_tiled must accept well-formed inputs");

    // Build `[n_t, top_k, h]` expert-outs + per-token top-k weights:
    // slot 0 from the routed GEMM, slot 1 from a scalar matmul over
    // the second top-k pick.
    let mut expert_outs = vec![0.0f32; n_t * top_k * h];
    let mut weights = vec![0.0f32; n_t * top_k];
    for (t, picks) in picks_per_token.iter().enumerate() {
        expert_outs[(t * top_k) * h..(t * top_k + 1) * h]
            .copy_from_slice(&routed[t * h..(t + 1) * h]);
        let b_off = picks[1].0 * k * h;
        for j in 0..h {
            let acc: f32 = (0..k).map(|kk| a[t * k + kk] * b[b_off + kk * h + j]).sum();
            expert_outs[(t * top_k + 1) * h + j] = acc;
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

    (top_picks, router_logits, shared_out, reduced_out, writeback_out)
}

/// `qwen_moe_end_to_end_v2` baseline round-trip. `inputs` carries every
/// field that influences `BaselineRecorder::hash_inputs`. `expected_out`
/// is built via [`compute_qwen_moe_end_to_end_v2_output`] at test time
/// and pinned to `baselines.json` via byte-equality against the
/// recorded entry (`f32 -> f64` promotion is exact).
#[test]
fn qwen_moe_end_to_end_v2_baseline_round_trip() {
    let (top_picks, router_logits, shared_out, reduced_out, writeback_out) =
        compute_qwen_moe_end_to_end_v2_output();
    let to_f64 = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| x as f64).collect() };
    let capacity = (QWEN_MOE_V2_CAPACITY_FACTOR * QWEN_MOE_V2_NUM_TOKENS as f32
        / QWEN_MOE_V2_NUM_EXPERTS as f32)
        .ceil() as u64;
    let inputs = json!({
        "kernel": "qwen_moe_end_to_end_v2",
        "num_tokens": QWEN_MOE_V2_NUM_TOKENS,
        "num_experts": QWEN_MOE_V2_NUM_EXPERTS,
        "top_k": QWEN_MOE_V2_TOP_K,
        "hidden": QWEN_MOE_V2_HIDDEN,
        "k": QWEN_MOE_V2_K,
        "capacity_factor": QWEN_MOE_V2_CAPACITY_FACTOR,
        "seed": QWEN_MOE_V2_SEED,
    });
    let expected_out = json!({
        "kernel": "qwen_moe_end_to_end_v2",
        "num_tokens": QWEN_MOE_V2_NUM_TOKENS,
        "num_experts": QWEN_MOE_V2_NUM_EXPERTS,
        "top_k": QWEN_MOE_V2_TOP_K,
        "hidden": QWEN_MOE_V2_HIDDEN,
        "k": QWEN_MOE_V2_K,
        "capacity_factor": QWEN_MOE_V2_CAPACITY_FACTOR,
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
        file.baselines.contains_key("qwen_moe_end_to_end_v2"),
        "checked-in baselines must contain qwen_moe_end_to_end_v2 entry"
    );
    let recorded = file.baselines["qwen_moe_end_to_end_v2"].clone();
    assert_eq!(BaselineRecorder::hash_inputs(&inputs), recorded.input_hash);
    assert_close_envelope(&recorded.output, &expected_out);
    let r = recorder
        .verify("qwen_moe_end_to_end_v2", &inputs, &recorded.output)
        .expect("verify qwen_moe_end_to_end_v2");
    assert_eq!(r, VerifyResult::Ok);
}

// ============================================================================
// OLMoE-1B-7B per-stage end-to-end conformance trace (turn-14).
//
// Generalizes the Qwen-MoE v2 trace to the Open-LLM-MoE topology:
//   num_experts=64, top_k=8, shared_experts=1. The shape used here
//   is intentionally small (4 tokens, hidden=4) so the scalar
//   reference path matches the tiled kernels byte-for-byte while
//   still exercising the 64-expert router + 8-way top-k dispatch.
// ============================================================================

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
    let plan = moe_dispatch(&(0..n_t).collect::<Vec<_>>(), &assignments, n_e, OLMOE_CAPACITY_FACTOR)
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

    (top_picks, router_logits, shared_out, reduced_out, writeback_out)
}

/// `olmoe_moe_end_to_end` baseline round-trip.
#[test]
fn olmoe_moe_end_to_end_baseline_round_trip() {
    let (top_picks, router_logits, shared_out, reduced_out, writeback_out) =
        compute_olmoe_moe_end_to_end_output();
    let to_f64 = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| x as f64).collect() };
    let capacity = (OLMOE_CAPACITY_FACTOR * OLMOE_NUM_TOKENS as f32 / OLMOE_NUM_EXPERTS as f32)
        .ceil() as u64;
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
