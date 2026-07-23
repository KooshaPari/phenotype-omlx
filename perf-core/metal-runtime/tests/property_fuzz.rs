//! Property-based fuzz tests for `metal-runtime`.
//!
//! These tests use [`proptest`] to drive the public surface with randomized
//! inputs. Each property runs 64 cases (fast — `ProptestConfig::with_cases(64)`)
//! so the entire file completes in a small fraction of a second.
//!
//! Properties under test (the contract for `metal-runtime` invariants):
//!
//! 1. **Cache key distinctness** — inserting two distinct cache keys must
//!    not cause one to mask the other.
//! 2. **Compile shader-budget never exceeded** — successful compiles always
//!    produce a `shader_source.len()` ≤ `budget.max_shader_bytes`.
//! 3. **Pipeline step determinism** — two sequential `Pipeline::step` calls
//!    on the same inputs produce the same outputs.
//! 4. **Topo sort is total and stable** — every operator appears exactly
//!    once in the result.
//! 5. **Fingerprint hash changes on field change** — mutating a single
//!    field changes the fingerprint hash.
//! 6. **Compile error is deterministic** — the same invalid input produces
//!    the same error twice.

#![allow(unused_mut)]

mod common;

use proptest::prelude::*;

use metal_runtime::{
    BoundedCompiler, CompileBudget, CompileError, CompiledPipeline, DeviceFingerprint,
    EvictionPolicy, GpuFamily, Pipeline, PipelineCache,
};

use model_plan::{
    DType, ModelId, OperatorId, OperatorKind, OperatorPlan, Precision, QuantizationPolicy,
    SchedulerPolicy, TensorRef,
};

use common::identity_fp;

// ---------------------------------------------------------------------------
// Shared plan-building helpers (kept local so proptest shrinking can see them).
// ---------------------------------------------------------------------------

fn op_copy(id: u64, inputs: usize, outputs: usize, deps: Vec<OperatorId>) -> OperatorPlan {
    OperatorPlan {
        id: OperatorId(id),
        kind: OperatorKind::Copy,
        attention: None,
        inputs: (0..inputs)
            .map(|i| TensorRef {
                name: format!("i{}", i),
                shape: vec![1],
                dtype: DType::F32,
                state_id: None,
            })
            .collect(),
        outputs: (0..outputs)
            .map(|i| TensorRef {
                name: format!("o{}", i),
                shape: vec![1],
                dtype: DType::F32,
                state_id: None,
            })
            .collect(),
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps,
    }
}

fn linear_plan(op_count: usize) -> model_plan::ModelPlan {
    // Op i depends on op (i-1) for i > 0. Each op has 1 in / 1 out.
    let ops: Vec<OperatorPlan> = (0..op_count)
        .map(|i| {
            let deps = if i == 0 {
                vec![]
            } else {
                vec![OperatorId(i as u64)]
            };
            op_copy(i as u64 + 1, 1, 1, deps)
        })
        .collect();
    let plan = model_plan::ModelPlan::new_unchecked(
        ModelId(42),
        "fuzz-linear",
        "fuzz",
        ops,
        vec![],
        SchedulerPolicy::Eager,
        4,
        8,
    );
    plan.validate().expect("linear plan must validate");
    plan
}

fn fp_with_seed(seed: u64) -> DeviceFingerprint {
    // Deterministic fingerprint parameterized by a u64 seed; lets proptest
    // exercise multiple distinct fingerprints without colliding with the
    // real `compute()` path.
    let mut base = identity_fp(GpuFamily::Software);
    base.device_name = format!("fuzz-{seed}");
    base
}

// ---------------------------------------------------------------------------
// proptest: 6 properties, 64 cases each.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// **Property 1: cache distinct keys don't collide.** Inserting key `a`
    /// then asking for key `b` must still return `None` for `b`, even when
    /// the cache is under capacity.
    #[test]
    fn cache_distinct_keys_dont_collide(a in 0u64..1000, b in 0u64..1000) {
        prop_assume!(a != b, "a == b is the trivial identity case");
        let mut cache = PipelineCache::new(EvictionPolicy::Lru, 16);
        let plan_a = ModelId(a);
        let plan_b = ModelId(b);
        let fp = fp_with_seed(0);
        let fp_hash = fp.fingerprint_hash();
        cache.insert(
            plan_a,
            0,
            fp_hash,
            CompiledPipeline::placeholder(plan_a, "src-a", fp_hash),
        );
        // After inserting only `a`, asking for `b` must return None.
        let got_b = cache.get(plan_b, 0, fp_hash);
        prop_assert!(got_b.is_none(), "key b must not be shadowed by key a");
        // Sanity: key a is still there.
        prop_assert!(cache.get(plan_a, 0, fp_hash).is_some());
    }

    /// **Property 2: compile shader budget is never exceeded.** For inputs
    /// within the plan-validation contract, a successful compile always
    /// returns a `CompiledPipeline` whose shader source fits the budget.
    #[test]
    fn compile_shader_within_budget(op_count in 1usize..8, max_shader_bytes in 64usize..4096) {
        let plan = linear_plan(op_count);
        let fp = fp_with_seed(7);
        let budget = CompileBudget { max_ms: 60_000, max_shader_bytes };
        let compiler = BoundedCompiler::new(budget);
        match compiler.compile(&plan, &fp) {
            Ok(cp) => {
                prop_assert!(
                    cp.shader_source.len() <= budget.max_shader_bytes,
                    "shader source len {} exceeded budget {}",
                    cp.shader_source.len(),
                    budget.max_shader_bytes
                );
            }
            Err(CompileError::BudgetExceeded { shader_bytes, .. }) => {
                // Budget-exceeded is the legitimate overflow signal; the
                // invariant under test is "Ok => within budget". Anything
                // else is a real bug.
                prop_assert!(
                    shader_bytes > budget.max_shader_bytes,
                    "BudgetExceeded reported but shader_bytes {} <= budget {}",
                    shader_bytes,
                    budget.max_shader_bytes
                );
            }
            Err(other) => prop_assert!(false, "unexpected compile error: {:?}", other),
        }
    }

    /// **Property 3: pipeline step is deterministic.** Two `Pipeline::step`
    /// calls on the same compiled pipeline and inputs produce identical
    /// outputs and identical execution orders.
    #[test]
    fn pipeline_step_deterministic(seed in 0u64..1000, n in 1usize..8) {
        let plan = linear_plan(n);
        let fp = fp_with_seed(seed);
        let compiler = BoundedCompiler::new(CompileBudget {
            max_ms: 5_000,
            max_shader_bytes: 8 * 1024,
        });
        let mut cache = PipelineCache::new(EvictionPolicy::Lru, 4);
        let pipeline = Pipeline::compile(&plan, &fp, &compiler, &mut cache)
            .expect("compile must succeed for valid plan");
        let mut inputs = std::collections::HashMap::new();
        // Op 1 input name is "i0" (see `op_copy` helper).
        inputs.insert("i0".to_string(), vec![1.0_f32, 2.0, 3.0, 4.0]);
        let out1 = pipeline.step(&inputs).expect("step 1");
        let out2 = pipeline.step(&inputs).expect("step 2");
        prop_assert_eq!(&out1.values, &out2.values);
        prop_assert_eq!(&out1.execution_order, &out2.execution_order);
    }

    /// **Property 4: topo-sort is total and stable.** Every operator in a
    /// valid plan appears exactly once in the topo-sorted output.
    #[test]
    fn topo_sort_is_total_and_stable(op_count in 1usize..16) {
        let plan = linear_plan(op_count);
        let topo = plan.topo_sort().expect("linear plan must topo-sort");
        // Every operator id appears exactly once.
        let mut seen: Vec<OperatorId> = topo.iter().map(|o| o.id).collect();
        seen.sort_by_key(|id| id.0);
        let expected: Vec<OperatorId> = (1..=op_count as u64).map(OperatorId).collect();
        prop_assert_eq!(seen, expected);
        // Total: no duplicates and full coverage.
        let unique: std::collections::HashSet<OperatorId> = topo.iter().map(|o| o.id).collect();
        prop_assert_eq!(unique.len(), op_count);
    }

    /// **Property 5: fingerprint hash changes when a single field changes.**
    /// Mutate exactly one of the contributing fields and the hash must
    /// differ.
    #[test]
    fn fingerprint_hash_changes_on_field_change(seed in 0u64..1000, delta in 0u8..7) {
        let base = fp_with_seed(seed);
        let base_hash = base.fingerprint_hash();
        let mutated = match delta {
            // 0..7 covers 6 mutable fields (device_name, os, arch,
            // simd_bit_width, total_memory_bytes, gpu_family). The 7th
            // enum value falls through to device_name with a different
            // suffix so every match arm mutates exactly one field.
            0 => {
                let mut m = base.clone();
                m.device_name = format!("{}-mut", base.device_name);
                m
            }
            1 => {
                let mut m = base.clone();
                m.os = format!("{}-mut", base.os);
                m
            }
            2 => {
                let mut m = base.clone();
                m.arch = format!("{}-mut", base.arch);
                m
            }
            3 => {
                let mut m = base.clone();
                m.simd_bit_width = base.simd_bit_width.wrapping_add(64);
                m
            }
            4 => {
                let mut m = base.clone();
                m.total_memory_bytes = base.total_memory_bytes.wrapping_add(1);
                m
            }
            5 => {
                let mut m = base.clone();
                m.gpu_family = match base.gpu_family {
                    GpuFamily::Software => GpuFamily::AppleSilicon,
                    GpuFamily::AppleSilicon => GpuFamily::DiscreteGpu,
                    GpuFamily::DiscreteGpu => GpuFamily::IntegratedGpu,
                    GpuFamily::IntegratedGpu => GpuFamily::Software,
                };
                m
            }
            _ => {
                // Fallback path — also mutate sysctl_cached. This field
                // is excluded from the hash on purpose (per the
                // documented contract), so the hash must be EQUAL.
                let mut m = base.clone();
                m.sysctl_cached = !m.sysctl_cached;
                let h = m.fingerprint_hash();
                prop_assert_eq!(h, base_hash);
                return Ok(());
            }
        };
        let mutated_hash = mutated.fingerprint_hash();
        prop_assert_ne!(
            mutated_hash, base_hash,
            "fingerprint hash must change when field {} is mutated",
            delta
        );
    }

    /// **Property 6: compile error messages are deterministic.** For the
    /// same budget-violating plan, two compile calls produce errors with
    /// the same *deterministic* fields (the elapsed-time `compile_ms`
    /// field intentionally varies with wall-clock and is excluded from
    /// this property).
    #[test]
    fn compile_error_is_deterministic(seed in 0u64..1000) {
        let plan = linear_plan(4);
        let fp = fp_with_seed(seed);
        // max_ms = 0 forces the wall-clock budget branch deterministically.
        let budget = CompileBudget {
            max_ms: 0,
            max_shader_bytes: 1024 * 1024,
        };
        let compiler = BoundedCompiler::new(budget);
        let e1 = compiler.compile(&plan, &fp).expect_err("max_ms=0 must error");
        let e2 = compiler.compile(&plan, &fp).expect_err("max_ms=0 must error");

        // The variant must match.
        match (&e1, &e2) {
            (CompileError::BudgetExceeded { .. }, CompileError::BudgetExceeded { .. }) => {}
            _ => prop_assert!(false, "variant mismatch: {:?} vs {:?}", e1, e2),
        }

        // The deterministic budget fields must agree exactly. The
        // elapsed `compile_ms` field intentionally varies between calls
        // (it embeds wall-clock) and is documented as non-deterministic
        // in `compile.rs`.
        if let (
            CompileError::BudgetExceeded {
                max_ms: m1,
                shader_bytes: s1,
                max_shader_bytes: msb1,
                ..
            },
            CompileError::BudgetExceeded {
                max_ms: m2,
                shader_bytes: s2,
                max_shader_bytes: msb2,
                ..
            },
        ) = (&e1, &e2)
        {
            prop_assert_eq!(m1, m2);
            prop_assert_eq!(s1, s2);
            prop_assert_eq!(msb1, msb2);
        }
    }
}
