//! Byte-stable EvaluationReport v0.1 verifier.
//!
//! This module mirrors `pheno-harness/scripts/verify_contract.py`: sorted-key
//! canonical JSON, SHA-256 hashes, frozen enum sets, and derived totals.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const SCHEMA_HASH: &str = "533dd0fa0d9b36145ef2e23a5c32aed39a67bc09bd36822b58289b61d5640a2e";
const SCHEMA_V01: &str = include_str!("schema_v01.json");
const SUITES: &[&str] = &[
    "mmlu-pro",
    "gpqa-diamond",
    "aime",
    "arc-agi-2",
    "livecodebench",
    "aider-polyglot",
    "swe-bench",
    "swe-bench-pro",
    "bfcl",
    "terminal-bench",
];
const STATUSES: &[&str] = &["ok", "wrong", "error", "skipped"];
const JUDGES: &[&str] = &["deterministic", "regex", "llm"];
const VARIANTS: &[&str] = &["stock", "ours"];
const JUDGE_MODES: &[&str] = &["deterministic", "llm"];
const ENERGY: &[&str] = &["none", "m1_pmu", "nvidia_smi"];
const EVIDENCE: &[&str] = &[
    "live verified",
    "historical",
    "reported",
    "inferred",
    "unknown",
];
const WINNERS: &[&str] = &["stock", "ours", "tie"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyOutcome {
    Accept,
    Reject { message: String },
    InternalMismatch { message: String },
}

fn reject(message: impl Into<String>) -> VerifyOutcome {
    VerifyOutcome::Reject {
        message: message.into(),
    }
}
fn field<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.get(key)
}
fn text<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    field(v, key)?.as_str()
}
fn allowed(value: Option<&str>, values: &[&str]) -> bool {
    value.is_some_and(|v| values.contains(&v))
}
fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

/// Canonical JSON: recursively sorted object keys, compact separators, UTF-8.
pub fn canonical_bytes(value: &Value) -> Vec<u8> {
    match value {
        Value::Object(map) => {
            let mut out = Vec::from([b'{']);
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
            for (i, (key, value)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                out.extend(serde_json::to_vec(key).expect("JSON string cannot fail"));
                out.push(b':');
                out.extend(canonical_bytes(value));
            }
            out.push(b'}');
            out
        }
        Value::Array(values) => {
            let mut out = Vec::from([b'[']);
            for (i, value) in values.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                out.extend(canonical_bytes(value));
            }
            out.push(b']');
            out
        }
        _ => serde_json::to_vec(value).expect("JSON value cannot fail"),
    }
}

pub fn sha256_hex(value: &Value) -> String {
    hex(&Sha256::digest(canonical_bytes(value)))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn remove_hash_chain(value: &Value) -> Option<Value> {
    let Value::Object(map) = value else {
        return None;
    };
    let mut body = map.clone();
    body.remove("hash_chain");
    Some(Value::Object(body))
}

pub fn verify_self() -> VerifyOutcome {
    let schema: Value = match serde_json::from_str(SCHEMA_V01) {
        Ok(value) => value,
        Err(error) => return VerifyOutcome::InternalMismatch { message: format!("embedded schema JSON invalid: {error}") },
    };
    if SCHEMA_HASH.len() == 64
        && SCHEMA_HASH.bytes().all(|b| b.is_ascii_hexdigit())
        && sha256_hex(&schema) == SCHEMA_HASH
    {
        VerifyOutcome::Accept
    } else {
        VerifyOutcome::InternalMismatch {
            message: "invalid frozen schema hash".into(),
        }
    }
}

pub fn verify_artifact(artifact: &Value) -> VerifyOutcome {
    if verify_self() != VerifyOutcome::Accept {
        return VerifyOutcome::InternalMismatch {
            message: "self-test failed".into(),
        };
    }
    if text(artifact, "contract_version") != Some("0.1") {
        return reject("C1 contract_version mismatch");
    }
    let run = field(artifact, "run").and_then(Value::as_object);
    let Some(run) = run else {
        return reject("missing run");
    };
    for (key, values) in [
        ("variant", VARIANTS),
        ("judge_mode", JUDGE_MODES),
        ("energy_source", ENERGY),
        ("evidence_label", EVIDENCE),
    ] {
        if !allowed(run.get(key).and_then(Value::as_str), values) {
            return reject(format!("invalid run.{key}"));
        }
    }
    let comp = field(artifact, "comparator")
        .and_then(Value::as_object)
        .ok_or(());
    let Ok(comp) = comp else {
        return reject("missing comparator");
    };
    if !allowed(comp.get("winner").and_then(Value::as_str), WINNERS) {
        return reject("invalid comparator.winner");
    }
    let totals = field(artifact, "totals").and_then(Value::as_object);
    let Some(totals) = totals else {
        return reject("missing totals");
    };
    if !allowed(
        totals.get("evidence_label").and_then(Value::as_str),
        EVIDENCE,
    ) {
        return reject("invalid totals.evidence_label");
    }
    if text(artifact, "schema_hash") != Some(SCHEMA_HASH) {
        return reject("C2 schema_hash mismatch");
    }
    let Some(body) = remove_hash_chain(artifact) else {
        return reject("artifact must be object");
    };
    let Some(chain) = field(artifact, "hash_chain").and_then(Value::as_object) else {
        return reject("missing hash_chain");
    };
    let top_level_hash = sha256_hex(&body);
    if chain.get("top_level_sha256").and_then(Value::as_str) != Some(top_level_hash.as_str()) {
        return reject("C3 top-level hash mismatch");
    }
    let suites = field(artifact, "suites")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut ids = Vec::new();
    let mut total_cells = 0_u64;
    let mut total_passed = 0_u64;
    for suite in &suites {
        let Some(suite_obj) = suite.as_object() else {
            return reject("suite must be object");
        };
        let Some(name) = suite_obj.get("suite").and_then(Value::as_str) else {
            return reject("missing suite.name");
        };
        if !SUITES.contains(&name)
            || !allowed(
                suite_obj.get("evidence_label").and_then(Value::as_str),
                EVIDENCE,
            )
        {
            return reject("invalid suite enum");
        }
        let Some(tasks) = suite_obj.get("task_results").and_then(Value::as_array) else {
            return reject("missing task_results");
        };
        let mut seen = BTreeSet::new();
        let mut ok = 0_u64;
        for task in tasks {
            let Some(t) = task.as_object() else {
                return reject("task must be object");
            };
            let Some(id) = t.get("task_id").and_then(Value::as_str) else {
                return reject("missing task_id");
            };
            if !seen.insert(id) {
                return reject(format!("duplicate task_id {id}"));
            }
            ids.push(id.to_owned());
            if !allowed(t.get("status").and_then(Value::as_str), STATUSES)
                || !allowed(t.get("judge").and_then(Value::as_str), JUDGES)
            {
                return reject("invalid task enum");
            }
            if t.get("status").and_then(Value::as_str) == Some("ok") {
                ok += 1;
            }
        }
        let n = suite_obj.get("n").and_then(Value::as_u64).unwrap_or(0);
        total_cells += n;
        total_passed += suite_obj.get("passed").and_then(Value::as_u64).unwrap_or(0);
        let pass = suite_obj
            .get("pass_at_1")
            .and_then(Value::as_f64)
            .unwrap_or(-1.0);
        if round4(pass) != round4(if n == 0 { 0.0 } else { ok as f64 / n as f64 }) {
            return reject(format!("C5 pass@1 mismatch for {name}"));
        }
    }
    ids.sort();
    let joined = ids.join("\n");
    let id_hash = hex(&Sha256::digest(joined.as_bytes()));
    if chain.get("task_ids_sorted_sha256").and_then(Value::as_str) != Some(id_hash.as_str()) {
        return reject("C3b task ID hash mismatch");
    }
    if totals.get("cells").and_then(Value::as_u64) != Some(total_cells)
        || totals.get("passed").and_then(Value::as_u64) != Some(total_passed)
    {
        return reject("C6 totals mismatch");
    }
    if total_cells > 0
        && round4(
            totals
                .get("pass_at_1")
                .and_then(Value::as_f64)
                .unwrap_or(-1.0),
        ) != round4(total_passed as f64 / total_cells as f64)
    {
        return reject("C6 totals.pass_at_1 mismatch");
    }
    VerifyOutcome::Accept
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_keys_are_sorted() {
        let v: Value = serde_json::from_str(r#"{"b":1,"a":"é"}"#).unwrap();
        assert_eq!(
            String::from_utf8(canonical_bytes(&v)).unwrap(),
            r#"{"a":"é","b":1}"#
        );
    }
    #[test]
    fn sha256_uses_utf8_canonical_bytes() {
        let v: Value = serde_json::from_str(r#"{"a":"é"}"#).unwrap();
        assert_eq!(sha256_hex(&v).len(), 64);
    }
}
