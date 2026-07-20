//! Integration / contract tests for the model-plan domain crate.
//!
//! These tests exercise the public API across modules and are the
//! specification for the crate's behavior. Unit tests inside each module
//! cover internal helpers; this file pins the externally observable
//! contract.
//!
//! Driven TDD-style: written before the corresponding implementation. Each
//! `// RED:` marker notes the failure that this test must produce before
//! the matching code is implemented.

use model_plan::{
    AttentionKind, DType, InterpreterError, ModelId, ModelPlan, OperatorId, OperatorKind,
    OperatorPlan, Precision, QuantizationPolicy, ReferenceInterpreter, SchedulerPolicy,
    StateId, StateKind, StatePlan, TensorRef,
};
use serde_json::{json, Value};

/// Naive f32 dense matmul oracle: `c[i,j] = sum_k a[i,k] * b[k,j]`.
/// Used as ground truth for `ReferenceInterpreter::run` correctness tests.
fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Naive f32 SwiGLU oracle: `silu(g) * u`, where `silu(x) = x * sigmoid(x)`.
fn naive_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(g, u)| {
            let sig = 1.0 / (1.0 + (-*g).exp());
            let silu = *g * sig;
            silu * *u
        })
        .collect()
}

/// Build a minimal but valid `ModelPlan` with a single `DenseMatmul` operator
/// for tests that need a known-good plan they can mutate.
fn single_matmul_plan(m: usize, k: usize, n: usize) -> ModelPlan {
    let op_id = OperatorId(1);
    let a = TensorRef {
        name: "a".into(),
        shape: vec![m, k],
        dtype: DType::F32,
        state_id: None,
    };
    let b = TensorRef {
        name: "b".into(),
        shape: vec![k, n],
        dtype: DType::F32,
        state_id: None,
    };
    let c = TensorRef {
        name: "c".into(),
        shape: vec![m, n],
        dtype: DType::F32,
        state_id: None,
    };
    ModelPlan {
        id: ModelId(1),
        name: "toy".into(),
        model_family: "test".into(),
        operators: vec![OperatorPlan {
            id: op_id,
            kind: OperatorKind::DenseMatmul,
            attention: None,
            inputs: vec![a, b],
            outputs: vec![c],
            precision: Precision::Fp32,
            quant: QuantizationPolicy::Dense,
            deps: vec![],
        }],
        states: vec![],
        scheduler: SchedulerPolicy::Eager,
        max_seq_len: 16,
        vocab_size: 1,
    }
}

// ---------------------------------------------------------------------------
// Precision / Quantization / DType
// ---------------------------------------------------------------------------

#[test]
fn precision_bytes_for_each_variant() {
    assert_eq!(Precision::Fp32.bytes(), 4);
    assert_eq!(Precision::Fp16.bytes(), 2);
    assert_eq!(Precision::Bf16.bytes(), 2);
    assert_eq!(Precision::Int8.bytes(), 1);
    assert_eq!(Precision::UInt8.bytes(), 1);
}

#[test]
fn operator_kind_serde_round_trip() {
    let kinds = [
        OperatorKind::DenseMatmul,
        OperatorKind::GroupedMatmul { groups: 4 },
        OperatorKind::Rope,
        OperatorKind::RmsNorm,
        OperatorKind::LayerNorm,
        OperatorKind::Softmax,
        OperatorKind::SwiGLU,
        OperatorKind::GeLU,
        OperatorKind::SilU,
        OperatorKind::Embedding,
        OperatorKind::Sampling,
        OperatorKind::Arange,
        OperatorKind::Copy,
        OperatorKind::Add,
        OperatorKind::Mul,
        OperatorKind::Scatter,
        OperatorKind::Gather,
    ];
    for k in kinds {
        let s = serde_json::to_string(&k).expect("serialize");
        let back: OperatorKind = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(back, k, "round-trip mismatch for {:?}", k);
    }
}

#[test]
fn quantization_policy_rejects_invalid_group_size() {
    // Group size 0 is invalid for ternary quantization.
    let bad = QuantizationPolicy::Ternary {
        group_size: 0,
        bits: 2,
    };
    let err = ModelPlan::validate_quant(&bad).unwrap_err();
    assert!(matches!(err, model_plan::PlanError::InvalidQuantPolicy { .. }));
}

// ---------------------------------------------------------------------------
// ModelPlan validation
// ---------------------------------------------------------------------------

#[test]
fn model_plan_rejects_duplicate_operator_ids() {
    let mut plan = single_matmul_plan(2, 2, 2);
    plan.operators.push(plan.operators[0].clone());
    let err = plan.validate().unwrap_err();
    match err {
        model_plan::PlanError::DuplicateOperatorId { id } => assert_eq!(id, OperatorId(1)),
        other => panic!("expected DuplicateOperatorId, got {:?}", other),
    }
}

#[test]
fn model_plan_rejects_unknown_dep_id() {
    let mut plan = single_matmul_plan(2, 2, 2);
    plan.operators[0].deps.push(OperatorId(999));
    let err = plan.validate().unwrap_err();
    match err {
        model_plan::PlanError::UnknownDependency {
            operator,
            dependency,
        } => {
            assert_eq!(operator, OperatorId(1));
            assert_eq!(dependency, OperatorId(999));
        }
        other => panic!("expected UnknownDependency, got {:?}", other),
    }
}

#[test]
fn model_plan_rejects_state_owner_pointing_to_missing_operator() {
    let mut plan = single_matmul_plan(2, 2, 2);
    plan.states.push(StatePlan {
        id: StateId(7),
        kind: StateKind::KvCache,
        persistent: true,
        shape: vec![2, 2, 4],
        dtype: DType::F32,
        owner_operator: OperatorId(42),
        max_versions: 1,
    });
    let err = plan.validate().unwrap_err();
    match err {
        model_plan::PlanError::UnknownStateOwner {
            state,
            operator,
        } => {
            assert_eq!(state, StateId(7));
            assert_eq!(operator, OperatorId(42));
        }
        other => panic!("expected UnknownStateOwner, got {:?}", other),
    }
}

#[test]
fn model_plan_rejects_moe_top_k_above_expert_count() {
    let mut plan = single_matmul_plan(2, 2, 2);
    plan.scheduler = SchedulerPolicy::Moe {
        capacity_factor: 1.25,
        top_k: 16,
        num_experts: 8,
    };
    let err = plan.validate().unwrap_err();
    match err {
        model_plan::PlanError::InvalidScheduler { reason } => {
            assert!(
                reason.contains("top_k") && reason.contains("num_experts"),
                "reason should mention top_k and num_experts, got {:?}",
                reason
            );
        }
        other => panic!("expected InvalidScheduler, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// JSON serialization
// ---------------------------------------------------------------------------

#[test]
fn model_plan_round_trips_via_json_with_no_unknown_fields() {
    let plan = single_matmul_plan(2, 3, 4);
    let s = serde_json::to_string(&plan).expect("serialize");
    let back: ModelPlan = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(back, plan);
}

#[test]
fn model_plan_rejects_unknown_field_in_json() {
    let plan = single_matmul_plan(2, 2, 2);
    let mut v: Value = serde_json::to_value(&plan).expect("to_value");
    v["mystery_field"] = json!(42);
    let s = v.to_string();
    let err = serde_json::from_str::<ModelPlan>(&s).unwrap_err();
    assert!(
        err.to_string().contains("unknown field"),
        "expected unknown-field error, got: {}",
        err
    );
}

// ---------------------------------------------------------------------------
// Dimension overflow
// ---------------------------------------------------------------------------

#[test]
fn model_plan_rejects_dimension_overflow() {
    // Construct a tensor ref whose shape dimension overflows what we allow.
    // usize::MAX as a single dim is unreasonable and should be rejected.
    let a = TensorRef {
        name: "a".into(),
        shape: vec![usize::MAX],
        dtype: DType::F32,
        state_id: None,
    };
    let op = OperatorPlan {
        id: OperatorId(1),
        kind: OperatorKind::Copy,
        attention: None,
        inputs: vec![a.clone()],
        outputs: vec![a],
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps: vec![],
    };
    let plan = ModelPlan {
        id: ModelId(1),
        name: "overflow".into(),
        model_family: "test".into(),
        operators: vec![op],
        states: vec![],
        scheduler: SchedulerPolicy::Eager,
        max_seq_len: 16,
        vocab_size: 1,
    };
    let err = plan.validate().unwrap_err();
    match err {
        model_plan::PlanError::DimensionOverflow { operator, dim } => {
            assert_eq!(operator, OperatorId(1));
            assert_eq!(dim, usize::MAX);
        }
        other => panic!("expected DimensionOverflow, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Reference interpreter
// ---------------------------------------------------------------------------

#[test]
fn reference_interpreter_dense_matmul_matches_naive_oracle() {
    // 2x3 * 3x2 = 2x2.
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let expected = naive_matmul(&a, &b, 2, 3, 2);

    let plan = single_matmul_plan(2, 3, 2);
    let interp = ReferenceInterpreter::new(plan);
    let mut inputs = std::collections::HashMap::new();
    inputs.insert("a".to_string(), a);
    inputs.insert("b".to_string(), b);
    let outputs = interp.run(&inputs).expect("run succeeds");

    let c = outputs.outputs.get("c").expect("output c present");
    assert_eq!(c.len(), 4);
    for (got, want) in c.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-5,
            "matmul mismatch: got {} want {}",
            got,
            want
        );
    }
}

#[test]
fn reference_interpreter_swiglu_matches_naive_oracle() {
    let op_id = OperatorId(1);
    let gate = TensorRef {
        name: "gate".into(),
        shape: vec![4],
        dtype: DType::F32,
        state_id: None,
    };
    let up = TensorRef {
        name: "up".into(),
        shape: vec![4],
        dtype: DType::F32,
        state_id: None,
    };
    let out = TensorRef {
        name: "y".into(),
        shape: vec![4],
        dtype: DType::F32,
        state_id: None,
    };
    let plan = ModelPlan {
        id: ModelId(2),
        name: "swiglu-toy".into(),
        model_family: "test".into(),
        operators: vec![OperatorPlan {
            id: op_id,
            kind: OperatorKind::SwiGLU,
            attention: None,
            inputs: vec![gate.clone(), up.clone()],
            outputs: vec![out.clone()],
            precision: Precision::Fp32,
            quant: QuantizationPolicy::Dense,
            deps: vec![],
        }],
        states: vec![],
        scheduler: SchedulerPolicy::Eager,
        max_seq_len: 16,
        vocab_size: 1,
    };
    plan.validate().expect("plan validates");

    let gate_vals = vec![-1.0, 0.0, 1.0, 2.0];
    let up_vals = vec![0.5, 1.5, -0.25, 4.0];
    let expected = naive_swiglu(&gate_vals, &up_vals);

    let interp = ReferenceInterpreter::new(plan);
    let mut inputs = std::collections::HashMap::new();
    inputs.insert("gate".to_string(), gate_vals);
    inputs.insert("up".to_string(), up_vals);
    let outputs = interp.run(&inputs).expect("run succeeds");

    let y = outputs.outputs.get("y").expect("output y present");
    assert_eq!(y.len(), 4);
    for (got, want) in y.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-5,
            "swiglu mismatch: got {} want {}",
            got,
            want
        );
    }
}

#[test]
fn reference_interpreter_rejects_unsupported_operator_kind() {
    // Embedding is a real operator kind but not implemented in the slow
    // interpreter (intentionally — only the minimal subset is supported).
    let op_id = OperatorId(1);
    let idx = TensorRef {
        name: "idx".into(),
        shape: vec![3],
        dtype: DType::I8,
        state_id: None,
    };
    let weight = TensorRef {
        name: "weight".into(),
        shape: vec![3, 4],
        dtype: DType::F32,
        state_id: None,
    };
    let out = TensorRef {
        name: "y".into(),
        shape: vec![3, 4],
        dtype: DType::F32,
        state_id: None,
    };
    let plan = ModelPlan {
        id: ModelId(3),
        name: "embed-toy".into(),
        model_family: "test".into(),
        operators: vec![OperatorPlan {
            id: op_id,
            kind: OperatorKind::Embedding,
            attention: None,
            inputs: vec![idx, weight],
            outputs: vec![out],
            precision: Precision::Fp32,
            quant: QuantizationPolicy::Dense,
            deps: vec![],
        }],
        states: vec![],
        scheduler: SchedulerPolicy::Eager,
        max_seq_len: 16,
        vocab_size: 1,
    };
    plan.validate().expect("plan validates");
    let interp = ReferenceInterpreter::new(plan);
    let mut inputs = std::collections::HashMap::new();
    inputs.insert("idx".to_string(), vec![0.0, 1.0, 2.0]);
    let err = interp.run(&inputs).unwrap_err();
    match err {
        InterpreterError::UnsupportedOperator { operator, kind } => {
            assert_eq!(operator, OperatorId(1));
            assert!(matches!(kind, OperatorKind::Embedding));
        }
        other => panic!("expected UnsupportedOperator, got {:?}", other),
    }
}

#[test]
fn reference_interpreter_zero_input_is_well_defined() {
    // All-zero inputs must produce a finite, well-defined output.
    let a = vec![0.0f32; 4];
    let b = vec![0.0f32; 4];
    let expected = naive_matmul(&a, &b, 2, 2, 2);

    let plan = single_matmul_plan(2, 2, 2);
    let interp = ReferenceInterpreter::new(plan);
    let mut inputs = std::collections::HashMap::new();
    inputs.insert("a".to_string(), a);
    inputs.insert("b".to_string(), b);
    let outputs = interp.run(&inputs).expect("run succeeds");
    let c = outputs.outputs.get("c").expect("output c present");
    assert_eq!(c, &expected);
}

#[test]
fn reference_interpreter_odd_dimension_is_well_defined() {
    // seq_len=3 → matmul on 3x2 * 2x1 = 3x1.
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b: Vec<f32> = vec![0.5, -1.0];
    let expected = naive_matmul(&a, &b, 3, 2, 1);

    let a_ref = TensorRef {
        name: "a".into(),
        shape: vec![3, 2],
        dtype: DType::F32,
        state_id: None,
    };
    let b_ref = TensorRef {
        name: "b".into(),
        shape: vec![2, 1],
        dtype: DType::F32,
        state_id: None,
    };
    let c_ref = TensorRef {
        name: "c".into(),
        shape: vec![3, 1],
        dtype: DType::F32,
        state_id: None,
    };
    let plan = ModelPlan {
        id: ModelId(4),
        name: "odd".into(),
        model_family: "test".into(),
        operators: vec![OperatorPlan {
            id: OperatorId(1),
            kind: OperatorKind::DenseMatmul,
            attention: None,
            inputs: vec![a_ref, b_ref],
            outputs: vec![c_ref],
            precision: Precision::Fp32,
            quant: QuantizationPolicy::Dense,
            deps: vec![],
        }],
        states: vec![],
        scheduler: SchedulerPolicy::Eager,
        max_seq_len: 16,
        vocab_size: 1,
    };
    plan.validate().expect("plan validates");
    let interp = ReferenceInterpreter::new(plan);
    let mut inputs = std::collections::HashMap::new();
    inputs.insert("a".to_string(), a);
    inputs.insert("b".to_string(), b);
    let outputs = interp.run(&inputs).expect("run succeeds");
    let c = outputs.outputs.get("c").expect("output c present");
    assert_eq!(c.len(), 3);
    for (got, want) in c.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-5);
    }
}

// ---------------------------------------------------------------------------
// AttentionKind sanity (smoke test that the attention variants serialize).
// ---------------------------------------------------------------------------

#[test]
fn attention_kind_variants_serialize() {
    let variants = [
        AttentionKind::Gqa { kv_heads: 4 },
        AttentionKind::Mla { d_latent: 64, d_rope: 16 },
        AttentionKind::Cca { compressed_factor: 4 },
        AttentionKind::Paged { block_size: 16 },
        AttentionKind::Tree {
            width: 4,
            depth: 3,
        },
        AttentionKind::Dense,
    ];
    for v in variants {
        let s = serde_json::to_string(&v).expect("serialize");
        let back: AttentionKind = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(back, v);
    }
}

// Silence unused warnings for items that some tests import but may not all
// use; keeps the imports stable as the suite evolves.
#[allow(dead_code)]
fn _keep_imports_used(_: StateKind, _: QuantizationPolicy) {}