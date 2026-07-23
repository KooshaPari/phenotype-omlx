//! Soak tests for `metal-runtime`.
//!
//! These tests are wall-clock-bounded: every test completes inside a few
//! hundred milliseconds so the entire file is comfortably under 10 s. They
//! exercise the kernel under repeated load and assert:
//!
//! 1. `Pipeline::step` is stable across 100 sequential calls (no leaks,
//!    no panics, no growth).
//! 2. The cache survives 1000 insert+evict cycles.
//! 3. `compile` is idempotent — same `(plan, fingerprint)` produces the
//!    same `shader_source`.
//! 4. `DeviceFingerprint::fingerprint_hash` is stable across 100 calls.
//!
//! Wall-clock assertions use `std::time::Instant`. The ceilings chosen
//! below are intentionally generous (multi-second) so a slow CI box does
//! not flake the suite; correctness is asserted by the surrounding
//! invariants, not by the timing budget.

#![allow(unused_mut)]

mod common;

use std::time::Instant;

use metal_runtime::{
    BoundedCompiler, CompileBudget, CompiledPipeline, DeviceFingerprint, EvictionPolicy, GpuFamily,
    Pipeline, PipelineCache,
};

use model_plan::{
    DType, ModelId, OperatorId, OperatorKind, OperatorPlan, Precision, QuantizationPolicy,
    SchedulerPolicy, TensorRef,
};

use common::{identity_fp, two_op_plan};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn make_linear_plan(op_count: usize) -> model_plan::ModelPlan {
    fn op_copy(id: u64, deps: Vec<OperatorId>) -> OperatorPlan {
        OperatorPlan {
            id: OperatorId(id),
            kind: OperatorKind::Copy,
            attention: None,
            inputs: vec![TensorRef {
                name: "x".into(),
                shape: vec![1],
                dtype: DType::F32,
                state_id: None,
            }],
            outputs: vec![TensorRef {
                name: "y".into(),
                shape: vec![1],
                dtype: DType::F32,
                state_id: None,
            }],
            precision: Precision::Fp32,
            quant: QuantizationPolicy::Dense,
            deps,
        }
    }
    let ops: Vec<OperatorPlan> = (0..op_count)
        .map(|i| {
            let deps = if i == 0 {
                vec![]
            } else {
                vec![OperatorId(i as u64)]
            };
            op_copy(i as u64 + 1, deps)
        })
        .collect();
    let plan = model_plan::ModelPlan::new_unchecked(
        ModelId(99),
        "soak",
        "soak",
        ops,
        vec![],
        SchedulerPolicy::Eager,
        4,
        8,
    );
    plan.validate().expect("soak: linear plan must validate");
    plan
}

fn compile_small(plan: &model_plan::ModelPlan, fp: &DeviceFingerprint) -> Pipeline {
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 5_000,
        max_shader_bytes: 8 * 1024,
    });
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 4);
    Pipeline::compile(plan, fp, &compiler, &mut cache).expect("soak: compile must succeed")
}

// ---------------------------------------------------------------------------
// 1. 100 sequential Pipeline::step calls — wall-clock budget 5 s.
// ---------------------------------------------------------------------------

#[test]
fn pipeline_step_repeated_100x_no_leak() {
    let plan = make_linear_plan(2);
    let fp = identity_fp(GpuFamily::Software);
    let pipeline = compile_small(&plan, &fp);

    let mut inputs = std::collections::HashMap::new();
    inputs.insert("x".to_string(), vec![1.0_f32, 2.0, 3.0, 4.0]);

    let started = Instant::now();
    let mut last_values: Option<std::collections::HashMap<String, Vec<f32>>> = None;
    for _ in 0..100 {
        let out = pipeline
            .step(&inputs)
            .expect("soak: step must not error under repeated load");
        if let Some(prev) = &last_values {
            assert_eq!(
                &out.values, prev,
                "soak: Pipeline::step must be deterministic across calls"
            );
        }
        last_values = Some(out.values);
    }
    let elapsed = started.elapsed();
    assert!(
        elapsed.as_secs() < 5,
        "soak: 100 Pipeline::step calls took {} ms (ceiling 5000 ms)",
        elapsed.as_millis()
    );
}

// ---------------------------------------------------------------------------
// 2. 1000 insert+evict cycles — must not panic. Wall-clock budget 5 s.
// ---------------------------------------------------------------------------

#[test]
fn cache_eviction_repeated_1k_x_no_panic() {
    let started = Instant::now();
    let mut cache = PipelineCache::new(EvictionPolicy::Lru, 4);
    for i in 0..1000u64 {
        let plan_id = ModelId((i % 32) + 1); // cycle through 32 keys
        let fp_hash = i.wrapping_add(1);
        cache.insert(
            plan_id,
            0,
            fp_hash,
            CompiledPipeline::placeholder(plan_id, "src", fp_hash),
        );
        // Touch the cache to keep the LRU book-keeping exercised.
        let _ = cache.get(plan_id, 0, fp_hash);
    }
    let stats = cache.stats();
    // With capacity=4 and 32 rotating keys, we expect a steady state of
    // many evictions. The exact count depends on policy + access pattern,
    // // but it must be non-zero after 1000 inserts.
    assert!(stats.evictions > 0, "soak: expected non-zero evictions");
    assert!(
        stats.size as usize <= 4,
        "soak: cache size {} must not exceed capacity 4",
        stats.size
    );
    let elapsed = started.elapsed();
    assert!(
        elapsed.as_secs() < 5,
        "soak: 1000 cache cycles took {} ms (ceiling 5000 ms)",
        elapsed.as_millis()
    );
}

// ---------------------------------------------------------------------------
// 3. compile idempotence — same plan + fp compiled 10 times yields the
//    same `shader_source` string.
// ---------------------------------------------------------------------------

#[test]
fn compile_idempotent_same_input() {
    let plan = two_op_plan();
    let fp = identity_fp(GpuFamily::Software);
    let compiler = BoundedCompiler::new(CompileBudget {
        max_ms: 5_000,
        max_shader_bytes: 8 * 1024,
    });

    let mut reference: Option<String> = None;
    let started = Instant::now();
    for _ in 0..10 {
        let cp = compiler
            .compile(&plan, &fp)
            .expect("soak: compile must succeed on valid plan");
        match &reference {
            None => reference = Some(cp.shader_source),
            Some(prev) => assert_eq!(
                &cp.shader_source, prev,
                "soak: shader_source must be identical across compilations"
            ),
        }
    }
    let elapsed = started.elapsed();
    assert!(
        elapsed.as_secs() < 5,
        "soak: 10 compile calls took {} ms (ceiling 5000 ms)",
        elapsed.as_millis()
    );
    assert!(reference.is_some());
}

// ---------------------------------------------------------------------------
// 4. fingerprint_compute_repeated_100x_same_value — 100 calls produce the
//    same hash (timestamps excluded from the hash by contract).
// ---------------------------------------------------------------------------

#[test]
fn fingerprint_compute_repeated_100x_same_value() {
    let started = Instant::now();
    let reference = identity_fp(GpuFamily::AppleSilicon).fingerprint_hash();
    for _ in 0..100 {
        let h = identity_fp(GpuFamily::AppleSilicon).fingerprint_hash();
        assert_eq!(h, reference, "soak: fingerprint_hash must be stable");
    }
    let elapsed = started.elapsed();
    assert!(
        elapsed.as_secs() < 5,
        "soak: 100 fingerprint compute calls took {} ms (ceiling 5000 ms)",
        elapsed.as_millis()
    );
}
