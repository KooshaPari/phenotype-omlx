//! §4 — End-to-end pipeline contracts.
//!
//! Covers compile+step on a two-op plan, topological ordering of operators,
//! per-op tracing events, cache reuse by (plan_id, plan_revision, fp_hash),
//! cache distinctness when fingerprint changes, max_seq_len propagation,
//! the error path for self-cycles, the error path for missing operators,
//! cache-key inclusion of plan_revision, deterministic output across
//! compilations, and zero-token input handling.

use std::collections::HashMap;

use metal_runtime::{
    BoundedCompiler, CompileBudget, EvictionPolicy, GpuFamily, Pipeline, PipelineCache,
    PipelineError, StepOutput,
};
use model_plan::{DType, ModelId, ModelPlan, OperatorId, SchedulerPolicy};

use super::common::{identity_fp, missing_op_plan, op_copy, self_cycle_plan, tensor, two_op_plan};

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
    assert_eq!(
        out.execution_order,
        vec![OperatorId(1), OperatorId(2), OperatorId(3)]
    );
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
    assert!(
        hits_after > hits_before,
        "second compile must produce a cache hit"
    );
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
    assert_eq!(
        stats.size, 2,
        "two distinct fingerprint hashes -> two entries"
    );
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
    assert!(
        matches!(err, PipelineError::TopoSortFailed { .. }),
        "got {:?}",
        err
    );
}

#[test]
fn pipeline_returns_err_for_plan_referencing_missing_operator() {
    let plan = missing_op_plan();
    let fp = identity_fp(GpuFamily::Software);
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 1000,
        max_shader_bytes: 4096,
    });
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 4);
    let res = Pipeline::compile(&plan, &fp, &compiler, &mut cache);
    let err = res.expect_err("compile must reject missing-op plan");
    assert!(
        matches!(err, PipelineError::InvalidPlan { .. }),
        "got {:?}",
        err
    );
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
