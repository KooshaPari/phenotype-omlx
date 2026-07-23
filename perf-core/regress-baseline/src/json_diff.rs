//! Stable-JSON canonicalization + parallel-walk diff helpers.
//!
//! These are the hashing primitives [`crate::recorder::BaselineRecorder`]
//! uses to compare inputs and outputs against recorded baselines. They
//! are pure functions over [`serde_json::Value`] and have no I/O or
//! state of their own.

use std::collections::BTreeMap;

use serde_json::Value;

/// Recursively sort map keys so canonical-JSON hash is order-independent.
pub(crate) fn canonicalize(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut sorted: BTreeMap<String, Value> = BTreeMap::new();
            for (k, vv) in map {
                sorted.insert(k.clone(), canonicalize(vv));
            }
            let mut out = serde_json::Map::new();
            for (k, vv) in sorted {
                out.insert(k, vv);
            }
            Value::Object(out)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(canonicalize).collect()),
        other => other.clone(),
    }
}

/// Walk two JSON values in parallel and return the dotted path of the
/// first mismatch (or `None` if they are equal). Path is built from map
/// keys and array indices (`values.0`, `expert_buckets.2.0`, ...).
pub(crate) fn find_first_diff(
    expected: &Value,
    actual: &Value,
    path: &str,
) -> Option<(String, Value, Value)> {
    if expected == actual {
        return None;
    }
    match (expected, actual) {
        (Value::Object(e), Value::Object(a)) => {
            // Check all expected keys in stable order.
            let mut keys: Vec<&String> = e.keys().collect();
            keys.sort();
            for k in keys {
                let exp_v = &e[k];
                let act_v = a.get(k).unwrap_or(&Value::Null);
                let child = format!(
                    "{}.{}",
                    if path.is_empty() {
                        k.clone()
                    } else {
                        format!("{path}.{k}")
                    },
                    ""
                );
                if let Some(diff) = find_first_diff(exp_v, act_v, &child[..child.len() - 1]) {
                    return Some(diff);
                }
            }
            // Surplus keys on `actual` count as a mismatch.
            for k in a.keys() {
                if !e.contains_key(k) {
                    let p = if path.is_empty() {
                        k.clone()
                    } else {
                        format!("{path}.{k}")
                    };
                    return Some((p, Value::Null, a[k].clone()));
                }
            }
            None
        }
        (Value::Array(e), Value::Array(a)) => {
            let n = e.len().max(a.len());
            for i in 0..n {
                let exp_v = e.get(i).unwrap_or(&Value::Null);
                let act_v = a.get(i).unwrap_or(&Value::Null);
                let p = if path.is_empty() {
                    format!("{i}")
                } else {
                    format!("{path}.{i}")
                };
                if let Some(diff) = find_first_diff(exp_v, act_v, &p) {
                    return Some(diff);
                }
            }
            None
        }
        _ => Some((path.to_string(), expected.clone(), actual.clone())),
    }
}
