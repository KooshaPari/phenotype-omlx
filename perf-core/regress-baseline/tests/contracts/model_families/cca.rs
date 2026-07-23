//! ZAYA CCA & compact nonlinear expert path baseline.

use serde_json::json;

use super::super::{
    assert_close_envelope, checked_in_baselines_dir, BaselineRecorder, VerifyResult,
};

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
