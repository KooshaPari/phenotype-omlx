//! DeepSeek MLA cache baseline.

use serde_json::json;

use super::super::{
    assert_close_envelope, checked_in_baselines_dir, BaselineRecorder, VerifyResult,
};

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
