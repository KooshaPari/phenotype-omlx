//! On-disk persistence contract.
//!
//! Pins down how the recorder interacts with the filesystem:
//!
//! - A fresh recorder over the same directory sees the previously
//!   recorded entry (`record_persists_to_disk_and_reloads`).
//! - Reading from a non-existent directory yields an empty envelope
//!   rather than an error (`missing_baseline_file_returns_empty_envelope`).
//! - The checked-in `tests/baselines/` directory loads cleanly and
//!   contains the three documented kernel entries.
//! - The on-disk schema version is exactly `1` and matches
//!   [`regress_baseline::SCHEMA_VERSION`].
//! - `baseline_file_loads_and_matches` is the master round-trip check
//!   for the checked-in envelope: for every documented kernel the
//!   recorded output satisfies `verify` on the canonical inputs.

use super::{
    checked_in_baselines_dir, moe_inputs, swa_inputs, BaselineRecorder, BaselinesFile,
    SCHEMA_VERSION,
};
use serde_json::json;
use tempfile::TempDir;

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
    assert_eq!(r, super::VerifyResult::Ok);
}

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
    assert_eq!(r, super::VerifyResult::Ok);
}

#[test]
fn missing_baseline_file_returns_empty_envelope() {
    let tmp = TempDir::new().expect("tempdir");
    let nested = tmp.path().join("does_not_exist");
    let recorder = BaselineRecorder::new(nested);
    let file = recorder.load().expect("load of missing dir must return empty envelope");
    assert_eq!(file.schema_version, SCHEMA_VERSION);
    assert!(file.baselines.is_empty());
}

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

#[test]
fn baseline_json_schema_version_is_one() {
    let path = checked_in_baselines_dir().join("baselines.json");
    let raw = std::fs::read_to_string(&path).expect("read checked-in file");
    let parsed: BaselinesFile = serde_json::from_str(&raw).expect("parse");
    assert_eq!(parsed.schema_version, 1);
    assert_eq!(parsed.schema_version, SCHEMA_VERSION);
}
