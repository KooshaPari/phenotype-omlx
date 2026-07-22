//! Input-hash contract.
//!
//! `BaselineRecorder::hash_inputs` must:
//!
//! 1. Be order-independent for object keys: structurally-equivalent JSON
//!    objects hash the same regardless of declaration order.
//! 2. Differ for any byte change: a single-bit flip in any leaf
//!    (including a tiny float delta) must produce a different hash.

use super::BaselineRecorder;
use serde_json::json;

#[test]
fn hash_inputs_is_order_independent_for_object_keys() {
    let a = json!({ "x": 1, "y": 2, "z": 3 });
    let b = json!({ "z": 3, "y": 2, "x": 1 });
    assert_eq!(
        BaselineRecorder::hash_inputs(&a),
        BaselineRecorder::hash_inputs(&b)
    );
}

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
