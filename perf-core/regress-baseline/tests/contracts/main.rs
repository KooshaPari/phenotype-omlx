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
//!
//! The suite is split across per-topic sub-modules:
//!
//! - [`record_verify`] — round-trip & idempotency contracts.
//! - [`mismatch`] — verify failure modes (input-hash drift, output drift,
//!   unknown kernel, overwrite-then-stale-verify).
//! - [`persistence`] — on-disk reload, empty-envelope fallback, checked-in
//!   directory, schema-version pin.
//! - [`hashing`] — input-hash order-independence & sensitivity.
//! - [`model_families`] — end-to-end trace baselines for ZAYA CCA,
//!   DeepSeek MLA, Qwen DeltaNet+MoE.

use std::path::PathBuf;

use regress_baseline::{
    BaselineEntry, BaselineRecorder, BaselinesFile, VerifyResult, SCHEMA_VERSION,
};
use serde_json::{json, Value};

mod hashing;
mod mismatch;
mod model_families;
mod persistence;
mod record_verify;

/// Path to the directory that holds the checked-in `baselines.json`.
///
/// Tests are run with `cargo test` rooted at `perf-core/`, so the relative
/// path is `regress-baseline/tests/baselines`.
pub(crate) fn checked_in_baselines_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("baselines")
}

/// Build the canonical MoE router input JSON.
pub(crate) fn moe_inputs() -> Value {
    json!({
        "kernel": "router_topk",
        "num_experts": 4,
        "top_k": 2,
        "tie_break_seed": 0,
        "router_logits": [1.0_f32, 5.0, 2.0, 3.0],
    })
}

/// Build the canonical SWA dense attention input JSON.
pub(crate) fn swa_inputs() -> Value {
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

/// Build the canonical ternary pack input JSON.
pub(crate) fn ternary_inputs() -> Value {
    json!({
        "kernel": "ternary_pack",
        "group_size": 4,
        "values": ["Pos", "Neg", "Zero", "Pos"],
    })
}

/// Field-by-field tolerance check (abs=1e-5, rel=1e-4) so the
/// verification contract is explicit instead of relying on
/// `find_first_diff`'s structural equality.
///
/// Used by the per-model-family trace tests in [`model_families`] to
/// compare a recorded baseline against its expected-output snapshot.
pub(crate) fn assert_close_envelope(out_a: &Value, out_b: &Value) {
    fn close(a: f64, b: f64) -> bool {
        let diff = (a - b).abs();
        if diff <= 1e-5 {
            return true;
        }
        let denom = a.abs().max(b.abs()).max(f64::MIN_POSITIVE);
        diff / denom <= 1e-4
    }

    fn walk(exp: &Value, act: &Value, path: &str) -> Option<String> {
        match (exp, act) {
            (Value::Object(e), Value::Object(a)) => {
                let mut keys: Vec<&String> = e.keys().collect();
                keys.sort();
                for k in keys {
                    let exp_v = &e[k];
                    let act_v = a.get(k).unwrap_or(&Value::Null);
                    let child = if path.is_empty() {
                        k.clone()
                    } else {
                        format!("{path}.{k}")
                    };
                    if let Some(p) = walk(exp_v, act_v, &child) {
                        return Some(p);
                    }
                }
                None
            }
            (Value::Array(e), Value::Array(a)) => {
                let n = e.len().max(a.len());
                for i in 0..n {
                    let exp_v = e.get(i).unwrap_or(&Value::Null);
                    let act_v = a.get(i).unwrap_or(&Value::Null);
                    let child = if path.is_empty() {
                        format!("{i}")
                    } else {
                        format!("{path}.{i}")
                    };
                    if let Some(p) = walk(exp_v, act_v, &child) {
                        return Some(p);
                    }
                }
                None
            }
            (Value::Number(e), Value::Number(a)) => {
                let e_f = e.as_f64().expect("f64 expected");
                let a_f = a.as_f64().expect("f64 expected");
                if close(e_f, a_f) {
                    None
                } else {
                    Some(format!(
                        "{path}: expected={e_f}, actual={a_f}, abs={}, rel={}",
                        (e_f - a_f).abs(),
                        (e_f - a_f).abs() / e_f.abs().max(a_f.abs()).max(f64::MIN_POSITIVE),
                    ))
                }
            }
            // Scalars (strings, bools, integers, nulls) compared strictly:
            // they only appear in the output envelopes as enum-like
            // discriminators (e.g. `"kernel": "cca_block_attend"`,
            // `"capacity": 2`, `"top_picks": [2, 1]`), so an exact match
            // is the right contract here. Any drift in these fields is a
            // shape mismatch that the float tolerance must not paper
            // over.
            (e, a) if e == a => None,
            (e, a) => Some(format!("{path}: expected={e}, actual={a}")),
        }
    }

    if let Some(p) = walk(out_a, out_b, "") {
        panic!("baseline output drifted at {p}");
    }
}
