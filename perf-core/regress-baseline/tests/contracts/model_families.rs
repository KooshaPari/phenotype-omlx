//! End-to-end trace baselines for the documented model families.
//!
//! Each test pins down a different acceptance-matrix row from
//! `02_SPECIFICATIONS.md`:
//!
//! | Test | Family / row | Trace shape |
//! |---|---|---|
//! | `cca_block_baseline_round_trip` | ZAYA — CCA & compact nonlinear expert path | Three block summaries + query → softmax-weighted output |
//! | `mla_cache_baseline_round_trip` | DeepSeek — MLA, routed experts, proposal/MTP | Four `compressed_kv` + `k_rope` cache entries + query |
//! | `qwen_deltanet_moe_end_to_end_baseline_round_trip` | Qwen agentic — long-context decode, GQA or DeltaNet state, sparse MoE | DeltaNet recurrence over a four-token chunk → top-k sparse-MoE |
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
