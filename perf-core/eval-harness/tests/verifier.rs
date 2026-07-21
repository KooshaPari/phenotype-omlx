use eval_harness::verifier::{canonical_bytes, sha256_hex, verify_artifact, VerifyOutcome};
use serde_json::json;
use sha2::{Digest, Sha256};

#[test]
fn verifier_rejects_wrong_contract_version() {
    let artifact = json!({"contract_version": "0.2"});
    assert!(matches!(
        verify_artifact(&artifact),
        VerifyOutcome::Reject { .. }
    ));
}

#[test]
fn verifier_accepts_canonical_minimal_report() {
    let mut artifact = json!({
        "contract_version": "0.1",
        "run": {
            "variant": "ours",
            "judge_mode": "deterministic",
            "energy_source": "none",
            "evidence_label": "live verified"
        },
        "comparator": {"winner": "ours"},
        "schema_hash": "533dd0fa0d9b36145ef2e23a5c32aed39a67bc09bd36822b58289b61d5640a2e",
        "suites": [{
            "suite": "mmlu-pro",
            "evidence_label": "live verified",
            "n": 1,
            "passed": 1,
            "pass_at_1": 1.0,
            "task_results": [{
                "task_id": "task-1",
                "status": "ok",
                "judge": "deterministic",
            }]
        }],
        "totals": {
            "cells": 1,
            "passed": 1,
            "pass_at_1": 1.0,
            "evidence_label": "live verified"
        }
    });
    let id_hash = format!("{:x}", Sha256::digest(b"task-1"));
    let body_hash = sha256_hex(&artifact);
    artifact["hash_chain"] = json!({
        "top_level_sha256": body_hash,
        "task_ids_sorted_sha256": id_hash
    });
    assert_eq!(verify_artifact(&artifact), VerifyOutcome::Accept);
    assert!(!canonical_bytes(&artifact).is_empty());
}
