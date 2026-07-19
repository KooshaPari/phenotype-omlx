//! `record` → `verify` round-trip and idempotency contracts.
//!
//! Covers the happy-path verification: after `record(k, in, out)` succeeds,
//! a subsequent `verify(k, in, out)` must return [`VerifyResult::Ok`],
//! regardless of whether the verifier is the same recorder that produced
//! the baseline, and regardless of how nested the output payload is.
//!
//! Also pins down idempotency: calling `record` twice with identical
//! `(kernel_name, inputs, outputs)` returns equivalent [`BaselineEntry`]
//! values and produces no observable drift.

use super::{moe_inputs, BaselineEntry, BaselineRecorder, VerifyResult};
use serde_json::json;
use tempfile::TempDir;

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
