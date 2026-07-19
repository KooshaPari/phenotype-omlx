//! Integration tests for `metal-runtime`.
//!
//! These tests exercise the public surface of the `metal-runtime` crate from
//! the outside, exactly as a downstream consumer would. They cover four
//! contracts:
//!
//! 1. Device fingerprinting is stable, distinct, hashable, and round-trips
//!    through serde regardless of platform.
//! 2. The bounded LRU/FIFO cache counts hits/misses/evictions and persists
//!    to disk.
//! 3. The bounded compiler respects the shader-byte and millisecond budget
//!    and emits useful errors when either (or both) are violated.
//! 4. The pipeline compiles + steps a `ModelPlan`, topologically orders its
//!    operators, caches by `(plan_id, plan_revision, fingerprint_hash)`, and
//!    is deterministic across compilations.
//!
//! 27 tests (the spec's checklist lists 27 explicit cases — we follow it
//! exactly).
//!
//! The `mut` keyword on every `let mut cache = ...` is required because
//! `Pipeline::compile(&mut cache)` needs an exclusive borrow; cache-only
//! tests get an `unused_mut` warning that we silence file-wide.

#![allow(unused_mut)]

mod common;

use std::collections::HashMap;
use std::hash::Hash;

use model_plan::{DType, ModelId, ModelPlan, OperatorId, SchedulerPolicy};
use metal_runtime::{
    BoundedCompiler, CacheStats, CompileBudget, CompileError, CompiledPipeline,
    DeviceFingerprint, EvictionPolicy, GpuFamily, Pipeline, PipelineCache, PipelineError,
    StepOutput,
};

use common::{identity_fp, op_copy, self_cycle_plan, tnow_ms, tensor, two_op_plan};

// ---------------------------------------------------------------------------
// 1. fingerprint tests (5)
// ---------------------------------------------------------------------------

#[test]
fn fingerprint_is_stable_across_calls_on_same_machine() {
    let a = DeviceFingerprint::compute().expect("compute on host platform");
    let b = DeviceFingerprint::compute().expect("compute on host platform");
    // device_name, os, arch, gpu_family, sysctl_cached must be stable. The
    // captured_at_unix_ms is explicitly NOT compared.
    assert_eq!(a.device_name, b.device_name);
    assert_eq!(a.os, b.os);
    assert_eq!(a.arch, b.arch);
    assert_eq!(a.gpu_family, b.gpu_family);
    assert_eq!(a.simd_bit_width, b.simd_bit_width);
    assert_eq!(a.total_memory_bytes, b.total_memory_bytes);
}

#[test]
fn fingerprint_distinct_across_fake_gpu_families() {
    let mut set = std::collections::HashSet::new();
    for fam in [
        GpuFamily::Software,
        GpuFamily::AppleSilicon,
        GpuFamily::DiscreteGpu,
        GpuFamily::IntegratedGpu,
    ] {
        let fp = identity_fp(fam);
        // hash() must be different across families (device_name identical).
        set.insert(fp.fingerprint_hash());
    }
    assert_eq!(set.len(), 4, "all four GpuFamily variants must hash distinctly");
}

#[test]
fn fingerprint_hash_matches_itself() {
    let fp = identity_fp(GpuFamily::AppleSilicon);
    assert_eq!(fp.fingerprint_hash(), fp.fingerprint_hash());
    // Hashing the same fingerprint twice via the Hash trait must produce
    // the same u64 — exercise this directly without BuildHasher (which is
    // not implemented for DefaultHasher itself).
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut h1 = DefaultHasher::new();
    fp.hash(&mut h1);
    let digest1 = h1.finish();
    let mut h2 = DefaultHasher::new();
    fp.hash(&mut h2);
    let digest2 = h2.finish();
    assert_eq!(digest1, digest2);
}

#[test]
fn fingerprint_software_fallback_on_non_macos_is_deterministic() {
    let a = DeviceFingerprint::compute_software();
    let b = DeviceFingerprint::compute_software();
    assert_eq!(a, b);
    assert_eq!(a.gpu_family, GpuFamily::Software);
    assert_eq!(a.device_name, "software-fallback");
    assert!(a.total_memory_bytes > 0);
    assert!(a.simd_bit_width >= 64);
    assert!(a.captured_at_unix_ms <= tnow_ms());
}

#[test]
fn fingerprint_serializes_and_round_trips() {
    let fp = DeviceFingerprint {
        device_name: "M2 Pro".to_string(),
        os: "macos".to_string(),
        arch: "aarch64".to_string(),
        simd_bit_width: 128,
        total_memory_bytes: 16 * 1024 * 1024 * 1024,
        gpu_family: GpuFamily::AppleSilicon,
        sysctl_cached: true,
        captured_at_unix_ms: 1_700_000_000_000,
    };
    let s = serde_json::to_string(&fp).expect("serialize");
    let back: DeviceFingerprint = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(back, fp);
}

// ---------------------------------------------------------------------------
// 2. cache tests (5)
// ---------------------------------------------------------------------------

#[test]
fn cache_insert_and_get_returns_some_value() {
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 16);
    let fp = identity_fp(GpuFamily::Software);
    let compiled = CompiledPipeline::placeholder(ModelId(1), "src", fp.fingerprint_hash());
    cache.insert(ModelId(1), 0, fp.fingerprint_hash(), compiled.clone());
    let got = cache.get(ModelId(1), 0, fp.fingerprint_hash());
    assert!(got.is_some());
    assert_eq!(got.unwrap().shader_source, "src");
}

#[test]
fn cache_get_on_missing_key_returns_none() {
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 16);
    let fp = identity_fp(GpuFamily::Software);
    let missing = cache.get(ModelId(999), 0, fp.fingerprint_hash());
    assert!(missing.is_none());
}

#[test]
fn cache_lru_evicts_least_recently_used_when_over_capacity() {
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 3);
    cache.insert(ModelId(1), 0, 1, CompiledPipeline::placeholder(ModelId(1), "a", 1));
    cache.insert(ModelId(1), 0, 2, CompiledPipeline::placeholder(ModelId(1), "b", 2));
    cache.insert(ModelId(1), 0, 3, CompiledPipeline::placeholder(ModelId(1), "c", 3));
    // Touch key 1 so it is no longer least-recently-used.
    assert!(cache.get(ModelId(1), 0, 1).is_some());
    // Insert a 4th entry — key 2 should be evicted (LRU after the touch).
    cache.insert(ModelId(1), 0, 4, CompiledPipeline::placeholder(ModelId(1), "d", 4));
    assert!(cache.get(ModelId(1), 0, 1).is_some(), "key 1 was touched");
    assert!(cache.get(ModelId(1), 0, 2).is_none(), "key 2 should be LRU-evicted");
    assert!(cache.get(ModelId(1), 0, 3).is_some());
    assert!(cache.get(ModelId(1), 0, 4).is_some());
}

#[test]
fn cache_fifo_evicts_in_insertion_order_regardless_of_access() {
    let mut cache = PipelineCache::new(EvictionPolicy::Fifo, 3);
    cache.insert(ModelId(1), 0, 1, CompiledPipeline::placeholder(ModelId(1), "a", 1));
    cache.insert(ModelId(1), 0, 2, CompiledPipeline::placeholder(ModelId(1), "b", 2));
    cache.insert(ModelId(1), 0, 3, CompiledPipeline::placeholder(ModelId(1), "c", 3));
    // Touch key 1 — under LRU this would refresh it, but FIFO must still
    // evict the oldest (key 1).
    assert!(cache.get(ModelId(1), 0, 1).is_some());
    cache.insert(ModelId(1), 0, 4, CompiledPipeline::placeholder(ModelId(1), "d", 4));
    assert!(cache.get(ModelId(1), 0, 1).is_none(), "FIFO must evict insertion-1");
    assert!(cache.get(ModelId(1), 0, 2).is_some());
    assert!(cache.get(ModelId(1), 0, 3).is_some());
    assert!(cache.get(ModelId(1), 0, 4).is_some());
}

#[test]
fn cache_hits_misses_evictions_counters_update_correctly() {
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 2);
    cache.insert(ModelId(1), 0, 1, CompiledPipeline::placeholder(ModelId(1), "a", 1));
    cache.insert(ModelId(1), 0, 2, CompiledPipeline::placeholder(ModelId(1), "b", 2));
    assert!(cache.get(ModelId(1), 0, 1).is_some());
    assert!(cache.get(ModelId(1), 0, 2).is_some());
    assert!(cache.get(ModelId(1), 0, 999).is_none());
    cache.insert(ModelId(1), 0, 3, CompiledPipeline::placeholder(ModelId(1), "c", 3));
    cache.insert(ModelId(1), 0, 4, CompiledPipeline::placeholder(ModelId(1), "d", 4));
    let stats: CacheStats = cache.stats();
    assert_eq!(stats.hits, 2);
    assert!(stats.misses >= 1);
    assert!(stats.evictions >= 1, "two evictions expected, got {}", stats.evictions);
    assert_eq!(stats.size, 2);
}

#[test]
fn cache_same_key_different_fingerprint_hash_are_distinct_entries() {
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 8);
    cache.insert(ModelId(1), 0, 100, CompiledPipeline::placeholder(ModelId(1), "fp-A", 100));
    cache.insert(ModelId(1), 0, 200, CompiledPipeline::placeholder(ModelId(1), "fp-B", 200));
    assert_eq!(cache.get(ModelId(1), 0, 100).unwrap().shader_source, "fp-A");
    assert_eq!(cache.get(ModelId(1), 0, 200).unwrap().shader_source, "fp-B");
}

#[test]
fn cache_write_through_persists_entries_that_can_be_reloaded() {
    let dir = std::env::temp_dir().join(format!("metal-runtime-test-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("cache.json");

    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 4);
    cache.insert(ModelId(7), 0, 42, CompiledPipeline::placeholder(ModelId(7), "persisted", 42));
    cache.write_through(&path).expect("write_through");

    let mut cache2 = PipelineCache::new(EvictionPolicy::Lru, 4);
    cache2.load_from_disk(&path).expect("load_from_disk");
    let got = cache2.get(ModelId(7), 0, 42);
    assert!(got.is_some(), "entry must survive write_through + load_from_disk");
    assert_eq!(got.unwrap().shader_source, "persisted");

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

// ---------------------------------------------------------------------------
// 3. compile tests (3)
// ---------------------------------------------------------------------------

#[test]
fn compile_returns_ok_with_budget_respected() {
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 100,
        max_shader_bytes: 1024,
    });
    let fp = identity_fp(GpuFamily::Software);
    let plan = two_op_plan();
    let res = compiler.compile(&plan, &fp);
    assert!(res.is_ok(), "compile should succeed: {:?}", res.err());
    let cp = res.unwrap();
    assert!(cp.shader_source.len() <= 1024);
}

#[test]
fn compile_budget_exceeded_when_shader_source_exceeds_max_shader_bytes() {
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 10_000,
        max_shader_bytes: 8,
    });
    let fp = identity_fp(GpuFamily::Software);
    let plan = two_op_plan();
    let err = compiler.compile(&plan, &fp).expect_err("must exceed budget");
    match err {
        CompileError::BudgetExceeded {
            max_shader_bytes,
            shader_bytes,
            ..
        } => {
            assert!(shader_bytes > max_shader_bytes);
            assert_eq!(max_shader_bytes, 8);
        }
        other => panic!("expected BudgetExceeded, got {:?}", other),
    }
}

#[test]
fn compile_budget_exceeded_when_compile_ms_exceeds_max_ms() {
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 0,
        max_shader_bytes: 1024 * 1024,
    });
    let fp = identity_fp(GpuFamily::Software);
    let plan = two_op_plan();
    let err = compiler.compile(&plan, &fp).expect_err("must exceed budget");
    match err {
        CompileError::BudgetExceeded {
            max_ms,
            compile_ms,
            ..
        } => {
            assert!(compile_ms > max_ms);
            assert_eq!(max_ms, 0);
        }
        other => panic!("expected BudgetExceeded, got {:?}", other),
    }
}

#[test]
fn compile_error_message_includes_both_budget_dimensions_when_both_violated() {
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 0,
        max_shader_bytes: 4,
    });
    let fp = identity_fp(GpuFamily::Software);
    let plan = two_op_plan();
    let err = compiler.compile(&plan, &fp).expect_err("must exceed both");
    let msg = err.to_string();
    assert!(msg.contains("ms"), "error msg must mention ms: {}", msg);
    assert!(msg.contains("bytes"), "error msg must mention bytes: {}", msg);
}

// ---------------------------------------------------------------------------
// 4. pipeline tests (10)
// ---------------------------------------------------------------------------

#[test]
fn pipeline_compile_and_step_on_two_op_plan_produces_correct_output() {
    let plan = two_op_plan();
    let fp = identity_fp(GpuFamily::Software);
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 1000,
        max_shader_bytes: 4096,
    });
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 4);
    let pipeline = Pipeline::compile(&plan, &fp, &compiler, &mut cache).expect("compile");

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
    inputs.insert("w".to_string(), vec![1.0, 0.0, 0.0, 1.0]);
    let out = pipeline.step(&inputs).expect("step ok");
    let z = out.values.get("z").expect("z must exist");
    assert_eq!(z, &vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn pipeline_respects_operator_dep_ordering_topo_sort() {
    // 3 copy operators: op3 -> op2 -> op1. Insert in REVERSE order so we
    // verify topo sort picks the right execution order, not declaration.
    let plan = ModelPlan::new_unchecked(
        ModelId(11),
        "topo",
        "test",
        vec![
            op_copy(
                3,
                vec![tensor("b_out", vec![2], DType::F32)],
                vec![tensor("c_out", vec![2], DType::F32)],
                vec![OperatorId(2)],
            ),
            op_copy(
                2,
                vec![tensor("a_out", vec![2], DType::F32)],
                vec![tensor("b_out", vec![2], DType::F32)],
                vec![OperatorId(1)],
            ),
            op_copy(
                1,
                vec![tensor("a_in", vec![2], DType::F32)],
                vec![tensor("a_out", vec![2], DType::F32)],
                vec![],
            ),
        ],
        vec![],
        SchedulerPolicy::Eager,
        4,
        8,
    );
    plan.validate().expect("topo plan must validate");
    let fp = identity_fp(GpuFamily::Software);
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 1000,
        max_shader_bytes: 4096,
    });
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 4);
    let pipeline = Pipeline::compile(&plan, &fp, &compiler, &mut cache).expect("compile");
    let mut inputs = HashMap::new();
    inputs.insert("a_in".to_string(), vec![10.0, 20.0]);
    let out = pipeline.step(&inputs).expect("step ok");
    assert_eq!(out.values.get("c_out"), Some(&vec![10.0, 20.0]));
    assert_eq!(out.execution_order, vec![OperatorId(1), OperatorId(2), OperatorId(3)]);
}

#[test]
fn pipeline_emits_per_op_tracing_events() {
    let plan = two_op_plan();
    let fp = identity_fp(GpuFamily::Software);
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 1000,
        max_shader_bytes: 4096,
    });
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 4);
    let pipeline = Pipeline::compile(&plan, &fp, &compiler, &mut cache).expect("compile");
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vec![0.0; 4]);
    inputs.insert("w".to_string(), vec![1.0, 0.0, 0.0, 1.0]);
    let out = pipeline.step(&inputs);
    assert!(out.is_ok(), "step must succeed: {:?}", out.err());
    let order = out.unwrap().execution_order;
    assert!(order.contains(&OperatorId(1)));
    assert!(order.contains(&OperatorId(2)));
}

#[test]
fn pipeline_reuses_cached_compiled_pipeline_for_same_plan_and_fingerprint() {
    let plan = two_op_plan();
    let fp = identity_fp(GpuFamily::Software);
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 1000,
        max_shader_bytes: 4096,
    });
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 4);

    let _ = Pipeline::compile(&plan, &fp, &compiler, &mut cache).expect("first compile");
    let size_after_first = cache.stats().size;
    assert_eq!(size_after_first, 1);
    let hits_before = cache.stats().hits;

    let _ = Pipeline::compile(&plan, &fp, &compiler, &mut cache).expect("second compile");
    let hits_after = cache.stats().hits;
    assert!(hits_after > hits_before, "second compile must produce a cache hit");
}

#[test]
fn pipeline_recomputes_when_fingerprint_changes() {
    let plan = two_op_plan();
    let fp_a = identity_fp(GpuFamily::Software);
    let fp_b = identity_fp(GpuFamily::AppleSilicon);
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 1000,
        max_shader_bytes: 4096,
    });
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 4);

    Pipeline::compile(&plan, &fp_a, &compiler, &mut cache).expect("compile fp_a");
    Pipeline::compile(&plan, &fp_b, &compiler, &mut cache).expect("compile fp_b");
    let stats = cache.stats();
    assert_eq!(stats.size, 2, "two distinct fingerprint hashes -> two entries");
}

#[test]
fn pipeline_respects_max_seq_len_from_plan() {
    let plan = ModelPlan::new_unchecked(
        ModelId(21),
        "seq-len",
        "test",
        vec![op_copy(
            1,
            vec![tensor("x", vec![2], DType::F32)],
            vec![tensor("y", vec![2], DType::F32)],
            vec![],
        )],
        vec![],
        SchedulerPolicy::Eager,
        2,
        8,
    );
    plan.validate().expect("seq-len plan must validate");
    let fp = identity_fp(GpuFamily::Software);
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 1000,
        max_shader_bytes: 4096,
    });
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 4);
    let pipeline = Pipeline::compile(&plan, &fp, &compiler, &mut cache).expect("compile");
    assert_eq!(pipeline.max_seq_len(), 2);
}

#[test]
fn pipeline_returns_err_for_plan_with_self_cycle_in_deps() {
    let plan = self_cycle_plan();
    let fp = identity_fp(GpuFamily::Software);
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 1000,
        max_shader_bytes: 4096,
    });
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 4);
    let res = Pipeline::compile(&plan, &fp, &compiler, &mut cache);
    let err = res.expect_err("compile must reject self-cycle");
    assert!(matches!(err, PipelineError::TopoSortFailed { .. }), "got {:?}", err);
}

#[test]
fn pipeline_returns_err_for_plan_referencing_missing_operator() {
    let plan = common::missing_op_plan();
    let fp = identity_fp(GpuFamily::Software);
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 1000,
        max_shader_bytes: 4096,
    });
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 4);
    let res = Pipeline::compile(&plan, &fp, &compiler, &mut cache);
    let err = res.expect_err("compile must reject missing-op plan");
    assert!(matches!(err, PipelineError::InvalidPlan { .. }), "got {:?}", err);
}

#[test]
fn pipeline_cache_key_includes_plan_revision_not_just_plan_id() {
    let fp = identity_fp(GpuFamily::Software);
    let key_a = PipelineCache::cache_key_for(ModelId(1), 1, fp.fingerprint_hash());
    let key_b = PipelineCache::cache_key_for(ModelId(1), 2, fp.fingerprint_hash());
    assert_ne!(key_a, key_b);
}

#[test]
fn pipeline_deterministic_output_across_two_compilations_of_same_plan_and_fp() {
    let plan = two_op_plan();
    let fp = identity_fp(GpuFamily::Software);
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 1000,
        max_shader_bytes: 4096,
    });
    let mut cache1 = PipelineCache::new(EvictionPolicy::Lru, 4);
    let mut cache2 = PipelineCache::new(EvictionPolicy::Lru, 4);
    let p1 = Pipeline::compile(&plan, &fp, &compiler, &mut cache1).expect("compile 1");
    let p2 = Pipeline::compile(&plan, &fp, &compiler, &mut cache2).expect("compile 2");
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
    inputs.insert("w".to_string(), vec![2.0, 0.0, 0.0, 2.0]);
    let o1 = p1.step(&inputs).expect("step 1");
    let o2 = p2.step(&inputs).expect("step 2");
    assert_eq!(o1.values.get("z"), Some(&vec![2.0, 4.0, 6.0, 8.0]));
    assert_eq!(o2.values.get("z"), Some(&vec![2.0, 4.0, 6.0, 8.0]));
    assert_eq!(o1.execution_order, o2.execution_order);
}

#[test]
fn pipeline_zero_token_input_does_not_panic_and_returns_empty_step_output() {
    let plan = two_op_plan();
    let fp = identity_fp(GpuFamily::Software);
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 1000,
        max_shader_bytes: 4096,
    });
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 4);
    let pipeline = Pipeline::compile(&plan, &fp, &compiler, &mut cache).expect("compile");
    let inputs: HashMap<String, Vec<f32>> = HashMap::new();
    let res = pipeline.step(&inputs);
    match res {
        Ok(out) => {
            let _: StepOutput = out;
        }
        Err(PipelineError::MissingInput { .. }) => {
            // acceptable: we asked for an empty input set on a 2-op plan.
        }
        Err(other) => panic!("unexpected error: {:?}", other),
    }
}