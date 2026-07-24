//! `verify` failure modes.
//!
//! Pin down the four observable verify outcomes that aren't [`VerifyResult::Ok`]:
//!
//! 1. `InputHashMismatch` — the baseline was recorded for a different
//!    input shape. The test mutates a single field of the inputs
//!    (`tie_break_seed`) and asserts the recorded-vs-supplied hashes
//!    diverge.
//! 2. `Mismatch { field, expected, actual }` — the input hashes match
//!    but the output drifted. `field` must be the dotted JSON path to
//!    the drifted leaf (`out.1.0` in the canonical SWA fixture).
//! 3. `Mismatch { field: "<entry>", .. }` — verifying a kernel name that
//!    was never recorded produces a synthetic mismatch at the synthetic
//!    `<entry>` path so callers don't have to special-case it.
//! 4. After `record` overwrites an existing baseline, re-verifying
//!    against the *old* output must produce `Mismatch`, not `Ok` (i.e.
//!    the overwrite is observable through `verify`).

use super::{moe_inputs, swa_inputs, ternary_inputs, BaselineRecorder, VerifyResult};
use serde_json::json;
use tempfile::TempDir;

#[test]
fn verify_fails_on_different_input_hash() {
    let tmp = TempDir::new().expect("tempdir");
    let recorder = BaselineRecorder::new(tmp.path());
    let inputs = moe_inputs();
    let outputs = json!({ "picks": [] });
    recorder
        .record("k", &inputs, outputs.clone())
        .expect("record");
    // Tweak the input by changing the seed.
    let mut other = inputs.clone();
    other
        .as_object_mut()
        .unwrap()
        .insert("tie_break_seed".to_string(), serde_json::Value::from(1u64));
    let result = recorder
        .verify("k", &other, &outputs)
        .expect("verify must run");
    match result {
        VerifyResult::InputHashMismatch { expected, actual } => {
            assert_ne!(expected, actual);
        }
        other => panic!("expected InputHashMismatch, got {:?}", other),
    }
}

#[test]
fn verify_returns_mismatch_with_field_path() {
    let tmp = TempDir::new().expect("tempdir");
    let recorder = BaselineRecorder::new(tmp.path());
    let inputs = swa_inputs();
    let baseline = json!({
        "out": [
            [1.0_f32, 2.0],
            [3.0, 4.0]
        ]
    });
    recorder
        .record("swa", &inputs, baseline.clone())
        .expect("record");
    // Drift: change out[1][0] from 3.0 to 3.5.
    let mut drifted = baseline.clone();
    drifted["out"][1][0] = json!(3.5_f32);
    let r = recorder.verify("swa", &inputs, &drifted).expect("verify");
    match r {
        VerifyResult::Mismatch {
            field,
            expected,
            actual,
        } => {
            assert_eq!(field, "out.1.0");
            assert_eq!(expected, json!(3.0_f32));
            assert_eq!(actual, json!(3.5_f32));
        }
        other => panic!("expected Mismatch at out.1.0, got {:?}", other),
    }
}

#[test]
fn verify_unknown_kernel_returns_mismatch() {
    let tmp = TempDir::new().expect("tempdir");
    let recorder = BaselineRecorder::new(tmp.path());
    let r = recorder
        .verify("never_recorded", &json!({}), &json!({}))
        .expect("verify");
    assert!(matches!(r, VerifyResult::Mismatch { ref field, .. } if field == "<entry>"));
}

#[test]
fn update_baseline_overwrites_previous() {
    let tmp = TempDir::new().expect("tempdir");
    let recorder = BaselineRecorder::new(tmp.path());
    let inputs = ternary_inputs();
    let v1 = json!({ "packed": [0x01_u8] });
    let v2 = json!({ "packed": [0x02_u8] });
    recorder
        .record("ternary_pack_4of4", &inputs, v1.clone())
        .expect("record v1");
    recorder
        .record("ternary_pack_4of4", &inputs, v2.clone())
        .expect("record v2");
    let file = recorder.load().expect("load");
    assert_eq!(file.baselines["ternary_pack_4of4"].output, v2);
    let r = recorder
        .verify("ternary_pack_4of4", &inputs, &v2)
        .expect("verify v2");
    assert_eq!(r, VerifyResult::Ok);
    let r_old = recorder
        .verify("ternary_pack_4of4", &inputs, &v1)
        .expect("verify v1 (should now mismatch)");
    assert!(matches!(r_old, VerifyResult::Mismatch { .. }));
}
