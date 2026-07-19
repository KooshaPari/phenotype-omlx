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

// ---------------------------------------------------------------------------
// 15. ZAYA CCA block-parallel trace baseline round-trips through the
//     checked-in `baselines.json` envelope.
//
//     Acceptance matrix row: `ZAYA | CCA and compact nonlinear expert
//     path` (02_SPECIFICATIONS.md:36). The trace records three block
//     summaries, a query, and the softmax-weighted output. `seed`
//     (= 0xCAFE_BABE) is part of the inputs so the input hash
//     distinguishes this trace from any other shape that happens to
//     share the same field names.
// ---------------------------------------------------------------------------

fn assert_close_envelope(out_a: &Value, out_b: &Value) {
    // Field-by-field tolerance check (abs=1e-5, rel=1e-4) so the
    // verification contract is explicit instead of relying on
    // `find_first_diff`'s structural equality.
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

// ---------------------------------------------------------------------------
// 16. DeepSeek MLA cache trace baseline round-trips through the
//     checked-in `baselines.json` envelope.
//
//     Acceptance matrix row: `DeepSeek | MLA, routed experts, proposal
//     or MTP path` (02_SPECIFICATIONS.md:34). The trace records four
//     `compressed_kv` + `k_rope` cache entries plus a query and the
//     softmax-weighted compressed-KV output.
// ---------------------------------------------------------------------------

#[test]
fn mla_cache_baseline_round_trip() {
    let recorder = BaselineRecorder::new(checked_in_baselines_dir());
    let file = recorder.load().expect("load checked-in baselines");
    assert!(file.baselines.contains_key("mla_cache_attend"));

    let inputs = json!({
        "kernel": "mla_cache_attend",
        "d_latent": 4,
        "d_rope": 2,
        "seq_k": 4,
        "cache": [
            {
                "compressed_kv": [0.5_f64, -0.25, 1.0, 0.0],
                "k_rope": [0.1_f64, 0.2]
            },
            {
                "compressed_kv": [0.0_f64, 1.0, -0.5, 0.25],
                "k_rope": [-0.1_f64, 0.3]
            },
            {
                "compressed_kv": [0.75_f64, 0.75, -0.1, -0.1],
                "k_rope": [0.4_f64, -0.4]
            },
            {
                "compressed_kv": [0.2_f64, -0.2, 0.8, 0.6],
                "k_rope": [0.0_f64, 0.5]
            }
        ],
        "q_latent": [0.1_f64, 0.2, 0.3, 0.4],
        "q_rope": [0.5_f64, -0.5],
        "seed": 0xCAFE_BABE_u64,
    });

    let recorded = file.baselines["mla_cache_attend"].clone();
    assert_eq!(
        BaselineRecorder::hash_inputs(&inputs),
        recorded.input_hash,
        "mla_cache_attend input_hash drift",
    );

    let expected_out = json!({
        "kernel": "mla_cache_attend",
        "d_latent": 4,
        "d_rope": 2,
        "seq_k": 4,
        "out": [
            0.4212736878640673_f64,
            0.3243108994605529_f64,
            0.31111078283933086_f64,
            0.15425821296383402_f64,
        ],
    });
    assert_close_envelope(&recorded.output, &expected_out);

    let r = recorder
        .verify("mla_cache_attend", &inputs, &recorded.output)
        .expect("verify mla_cache_attend");
    assert_eq!(r, VerifyResult::Ok);
}

// ---------------------------------------------------------------------------
// 17. Qwen DeltaNet + sparse-MoE end-to-end trace baseline round-trips
//     through the checked-in `baselines.json` envelope.
//
//     Acceptance matrix row: `Qwen agentic | Long-context decode,
//     tool-use traces, GQA or DeltaNet state, sparse MoE`
//     (02_SPECIFICATIONS.md:33). The trace captures the DeltaNet
//     recurrence over a four-token chunk, followed by a top-k sparse-MoE
//     invocation that consumes the averaged chunk hidden state.
// ---------------------------------------------------------------------------

#[test]
fn qwen_deltanet_moe_end_to_end_baseline_round_trip() {
    let recorder = BaselineRecorder::new(checked_in_baselines_dir());
    let file = recorder.load().expect("load checked-in baselines");
    assert!(file.baselines.contains_key("qwen_deltanet_moe_end_to_end"));

    let inputs = json!({
        "kernel": "qwen_deltanet_moe_end_to_end",
        "head_dim": 2,
        "chunk_size": 4,
        "num_experts": 3,
        "top_k": 2,
        "capacity_factor": 1.5_f64,
        "seed_deltanet": 0xCAFE_BABE_u64,
        "seed_moe": 0xDEADBEEF_u64,
        "s_state": [
            [0.06991786752596547_f64, 0.03830528700763702_f64],
            [0.10781900698602688_f64, 0.4324679840513347_f64],
        ],
        "q": [
            [0.04246751020139372_f64, 0.23250469920865613_f64],
            [-0.11554406685392743_f64, 0.0122591474763446_f64],
            [-0.3425595539522917_f64, 0.05847873316181851_f64],
            [0.3132157016514764_f64, 0.3660663285138862_f64],
        ],
        "k": [
            [-0.44447468408744906_f64, -0.22367991373788565_f64],
            [0.19145389257663536_f64, -0.2094852785324945_f64],
            [-0.48348651593935527_f64, -0.3930602376598259_f64],
            [0.40759846942788025_f64, -0.15746015830490268_f64],
        ],
        "v": [
            [-0.37016338951183403_f64, -0.21605705313476337_f64],
            [-0.3663536859987412_f64, 0.1304810916468312_f64],
            [0.03904277585934779_f64, -0.3234472624437308_f64],
            [-0.05480652918228729_f64, 0.1619786984142897_f64],
        ],
        "beta": [
            0.2964370854789305_f64,
            0.43967315887807096_f64,
            0.4646655497572698_f64,
            0.4241783206411222_f64,
        ],
        "expert_weights": [
            [
                [-0.4072812395386209_f64, 0.1947473234742071_f64],
                [-0.21147098911193407_f64, 0.2517428778224292_f64],
            ],
            [
                [0.3136655956474912_f64, 0.22561408446039677_f64],
                [0.28389310702125237_f64, 0.173980483162687_f64],
            ],
            [
                [-0.13600417514424673_f64, -0.3498030810500954_f64],
                [0.02641908556073458_f64, 0.22842827701744894_f64],
            ],
        ],
        "router_logits": [
            -0.8440545527627564_f64,
            0.0954977262077159_f64,
            0.3527071863988631_f64,
        ],
    });

    let recorded = file.baselines["qwen_deltanet_moe_end_to_end"].clone();
    assert_eq!(
        BaselineRecorder::hash_inputs(&inputs),
        recorded.input_hash,
        "qwen_deltanet_moe_end_to_end input_hash drift",
    );

    let expected_out = json!({
        "kernel": "qwen_deltanet_moe_end_to_end",
        "head_dim": 2,
        "chunk_size": 4,
        "num_experts": 3,
        "top_k": 2,
        "capacity_factor": 1.5_f64,
        "capacity": 2,
        "top_picks": [2, 1],
        "deltanet_outputs": [
            [0.03498257255832317_f64, 0.10367783352946564_f64],
            [-0.007415759375221491_f64, -0.002294202351932063_f64],
            [-0.011464295507259211_f64, -0.002774929016837402_f64],
            [0.06961694677808955_f64, 0.189481050198067_f64],
        ],
        "moe_shared_out": [
            0.00016106075082136448_f64,
            0.04923174960616833_f64,
        ],
        "moe_reduced_out": [
            -0.00498551897116146_f64,
            0.015134984270487883_f64,
        ],
    });
    assert_close_envelope(&recorded.output, &expected_out);

    let r = recorder
        .verify("qwen_deltanet_moe_end_to_end", &inputs, &recorded.output)
        .expect("verify qwen_deltanet_moe_end_to_end");
    assert_eq!(r, VerifyResult::Ok);
}
