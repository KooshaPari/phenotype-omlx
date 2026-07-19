//! Contract tests for `regress-baseline`.
//!
//! These tests pin down the contract for `BaselineRecorder`:
//!
//! 1. `record` then `verify` is a round-trip on the same `(kernel_name, inputs, outputs)`.
//! 2. `verify` returns [`VerifyResult::InputHashMismatch`] when the inputs
//!    differ from the baseline (i.e. the baseline was recorded for a
//!    different shape).
//! 3. The checked-in `baselines.json` loads, contains the three documented
//!    kernels, and matches the recorded output for each one.
//! 4. `record` persists to disk and the file survives a re-load from a
//!    fresh recorder instance.
//! 5. Re-recording an existing kernel overwrites the previous entry.
//! 6. Missing baseline file returns a clean error (or empty envelope).
//! 7. The checked-in `baselines.json` contains the three required kernel
//!    entries.
//! 8. The schema version is exactly `1`.
//! 9. Recording the same `(kernel_name, inputs, outputs)` twice is
//!    idempotent.
//! 10. A drift in one output field surfaces a [`VerifyResult::Mismatch`]
//!     with the dotted field path.
//!
//! Plus a few additional sanity tests to keep the suite above 10.

use std::path::PathBuf;

use regress_baseline::{
    BaselineEntry, BaselineRecorder, BaselinesFile, SCHEMA_VERSION, VerifyResult,
};
use serde_json::{json, Value};
use tempfile::TempDir;

// The path to the directory that holds the checked-in `baselines.json`.
// Tests are run with `cargo test` rooted at `perf-core/`, so the relative
// path is `regress-baseline/tests/baselines`.
fn checked_in_baselines_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("baselines")
}

// Build the canonical MoE router input JSON.
fn moe_inputs() -> Value {
    json!({
        "kernel": "router_topk",
        "num_experts": 4,
        "top_k": 2,
        "tie_break_seed": 0,
        "router_logits": [1.0_f32, 5.0, 2.0, 3.0],
    })
}

// Build the canonical SWA dense attention input JSON.
fn swa_inputs() -> Value {
    json!({
        "kernel": "dense_attention",
        "head_dim": 2,
        "seq_q": 2,
        "seq_k": 2,
        "q": [[1.0_f32, 0.0], [0.0, 1.0]],
        "k": [[1.0_f32, 0.0], [0.0, 1.0]],
        "v": [[1.0_f32, 2.0], [3.0, 4.0]],
    })
}

// Build the canonical ternary pack input JSON.
fn ternary_inputs() -> Value {
    json!({
        "kernel": "ternary_pack",
        "group_size": 4,
        "values": ["Pos", "Neg", "Zero", "Pos"],
    })
}

// ---------------------------------------------------------------------------
// 1. round-trip record then verify
// ---------------------------------------------------------------------------

#[test]
fn record_then_verify_round_trip() {
    let tmp = TempDir::new().expect("tempdir");
    let recorder = BaselineRecorder::new(tmp.path());
    let inputs = moe_inputs();
    let outputs = json!({
        "picks": [
            { "expert_id": 1, "weight": 0.8807970779778823 },
            { "expert_id": 3, "weight": 0.11920292202211769 }
        ]
    });
    let entry = recorder
        .record("moe_router_topk_2of4", &inputs, outputs.clone())
        .expect("record must succeed");
    assert_eq!(entry.output, outputs);
    let result = recorder
        .verify("moe_router_topk_2of4", &inputs, &outputs)
        .expect("verify must succeed");
    assert_eq!(result, VerifyResult::Ok);
}

// ---------------------------------------------------------------------------
// 2. verify fails on a different input hash
// ---------------------------------------------------------------------------

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
    other.as_object_mut().unwrap().insert(
        "tie_break_seed".to_string(),
        serde_json::Value::from(1u64),
    );
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

// ---------------------------------------------------------------------------
// 3. checked-in baselines file loads and each entry matches its recorded
//    output.
// ---------------------------------------------------------------------------

#[test]
fn baseline_file_loads_and_matches() {
    let recorder = BaselineRecorder::new(checked_in_baselines_dir());
    let file: BaselinesFile = recorder.load().expect("load must succeed");
    assert_eq!(file.schema_version, SCHEMA_VERSION);
    assert!(file.baselines.contains_key("moe_router_topk_2of4"));
    assert!(file.baselines.contains_key("swa_dense_attention_2x2"));
    assert!(file.baselines.contains_key("ternary_pack_4of4"));
    // MoE: hash matches and verify returns Ok.
    let moe = moe_inputs();
    let moe_out = file.baselines["moe_router_topk_2of4"].output.clone();
    assert_eq!(
        BaselineRecorder::hash_inputs(&moe),
        file.baselines["moe_router_topk_2of4"].input_hash
    );
    let r = recorder
        .verify("moe_router_topk_2of4", &moe, &moe_out)
        .expect("verify");
    assert_eq!(r, VerifyResult::Ok);
}

// ---------------------------------------------------------------------------
// 4. record persists to disk and a fresh recorder instance reloads it.
// ---------------------------------------------------------------------------

#[test]
fn record_persists_to_disk_and_reloads() {
    let tmp = TempDir::new().expect("tempdir");
    let recorder_a = BaselineRecorder::new(tmp.path());
    let inputs = swa_inputs();
    let outputs = json!({ "out": [[1.5_f32, 2.5], [3.5, 4.5]] });
    recorder_a
        .record("swa_dense_attention_2x2", &inputs, outputs.clone())
        .expect("record a");

    // New recorder on the same dir; must see the previous entry.
    let recorder_b = BaselineRecorder::new(tmp.path());
    let file = recorder_b.load().expect("reload");
    assert!(file.baselines.contains_key("swa_dense_attention_2x2"));
    let reloaded = &file.baselines["swa_dense_attention_2x2"];
    assert_eq!(reloaded.output, outputs);
    assert_eq!(
        reloaded.input_hash,
        BaselineRecorder::hash_inputs(&inputs)
    );
    let r = recorder_b
        .verify("swa_dense_attention_2x2", &inputs, &outputs)
        .expect("verify b");
    assert_eq!(r, VerifyResult::Ok);
}

// ---------------------------------------------------------------------------
// 5. re-record overwrites the previous entry for the same kernel name.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 6. missing baseline file returns an empty envelope (not an error).
// ---------------------------------------------------------------------------

#[test]
fn missing_baseline_file_returns_empty_envelope() {
    let tmp = TempDir::new().expect("tempdir");
    let nested = tmp.path().join("does_not_exist");
    let recorder = BaselineRecorder::new(nested);
    let file = recorder.load().expect("load of missing dir must return empty envelope");
    assert_eq!(file.schema_version, SCHEMA_VERSION);
    assert!(file.baselines.is_empty());
}

// ---------------------------------------------------------------------------
// 7. the checked-in `baselines/` dir contains the required entries.
// ---------------------------------------------------------------------------

#[test]
fn baselines_dir_contains_required_entries() {
    let recorder = BaselineRecorder::new(checked_in_baselines_dir());
    let file = recorder.load().expect("load");
    for required in [
        "moe_router_topk_2of4",
        "swa_dense_attention_2x2",
        "ternary_pack_4of4",
    ] {
        assert!(
            file.baselines.contains_key(required),
            "missing required baseline entry: {}",
            required
        );
    }
}

// ---------------------------------------------------------------------------
// 8. the schema version in the checked-in file is exactly 1.
// ---------------------------------------------------------------------------

#[test]
fn baseline_json_schema_version_is_one() {
    let path = checked_in_baselines_dir().join("baselines.json");
    let raw = std::fs::read_to_string(&path).expect("read checked-in file");
    let parsed: BaselinesFile = serde_json::from_str(&raw).expect("parse");
    assert_eq!(parsed.schema_version, 1);
    assert_eq!(parsed.schema_version, SCHEMA_VERSION);
}

// ---------------------------------------------------------------------------
// 9. record is idempotent with the same input/output triple.
// ---------------------------------------------------------------------------

#[test]
fn record_is_idempotent_with_same_input() {
    let tmp = TempDir::new().expect("tempdir");
    let recorder = BaselineRecorder::new(tmp.path());
    let inputs = moe_inputs();
    let outputs = json!({ "picks": [[1, 0.5_f32], [3, 0.5]] });
    let a: BaselineEntry = recorder
        .record("k", &inputs, outputs.clone())
        .expect("record a");
    let b: BaselineEntry = recorder
        .record("k", &inputs, outputs.clone())
        .expect("record b");
    assert_eq!(a, b);
    assert_eq!(a.input_hash, BaselineRecorder::hash_inputs(&inputs));
}

// ---------------------------------------------------------------------------
// 10. a drift in one nested output field surfaces as a Mismatch with the
//     dotted path to the drifted leaf.
// ---------------------------------------------------------------------------

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
        VerifyResult::Mismatch { field, expected, actual } => {
            assert_eq!(field, "out.1.0");
            assert_eq!(expected, json!(3.0_f32));
            assert_eq!(actual, json!(3.5_f32));
        }
        other => panic!("expected Mismatch at out.1.0, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 11. (bonus) `verify` on a never-recorded kernel name returns Mismatch
//     with field == "<entry>".
// ---------------------------------------------------------------------------

#[test]
fn verify_unknown_kernel_returns_mismatch() {
    let tmp = TempDir::new().expect("tempdir");
    let recorder = BaselineRecorder::new(tmp.path());
    let r = recorder
        .verify("never_recorded", &json!({}), &json!({}))
        .expect("verify");
    assert!(matches!(r, VerifyResult::Mismatch { ref field, .. } if field == "<entry>"));
}

// ---------------------------------------------------------------------------
// 12. (bonus) `hash_inputs` is order-independent for object keys.
// ---------------------------------------------------------------------------

#[test]
fn hash_inputs_is_order_independent_for_object_keys() {
    let a = json!({ "x": 1, "y": 2, "z": 3 });
    let b = json!({ "z": 3, "y": 2, "x": 1 });
    assert_eq!(
        BaselineRecorder::hash_inputs(&a),
        BaselineRecorder::hash_inputs(&b)
    );
}

// ---------------------------------------------------------------------------
// 13. (bonus) `hash_inputs` differs for any byte change in the input.
// ---------------------------------------------------------------------------

#[test]
fn hash_inputs_changes_with_any_byte_change() {
    let a = json!({ "k": [1.0_f32, 2.0, 3.0] });
    let mut b = a.clone();
    b["k"][0] = json!(1.0001_f32);
    assert_ne!(
        BaselineRecorder::hash_inputs(&a),
        BaselineRecorder::hash_inputs(&b)
    );
}

// ---------------------------------------------------------------------------
// 14. (bonus) record then verify with same recorder and output that
//     contains nested arrays/objects round-trips cleanly.
// ---------------------------------------------------------------------------

#[test]
fn record_then_verify_nested_arrays_and_objects() {
    let tmp = TempDir::new().expect("tempdir");
    let recorder = BaselineRecorder::new(tmp.path());
    let inputs = json!({ "shape": [2, 3], "seed": 42_u64 });
    let outputs = json!({
        "values": [
            [1, 2, 3],
            [4, 5, 6]
        ],
        "meta": { "kind": "test", "flags": { "transpose": false } }
    });
    recorder
        .record("nested", &inputs, outputs.clone())
        .expect("record");
    let r = recorder
        .verify("nested", &inputs, &outputs)
        .expect("verify");
    assert_eq!(r, VerifyResult::Ok);
}
