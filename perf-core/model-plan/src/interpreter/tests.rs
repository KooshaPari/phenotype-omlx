//! Unit tests for [`crate::interpreter`].
//!
//! Extracted to a sibling file so the implementation module stays under
//! the 500-line cap while tests remain close to the code they cover.

use std::collections::HashMap;

use crate::dtype::DType;
use crate::error::InterpreterError;
use crate::interpreter::ReferenceInterpreter;
use crate::operator::{OperatorId, OperatorKind, OperatorPlan};
use crate::plan::{ModelId, ModelPlan};
use crate::precision::Precision;
use crate::quantization::QuantizationPolicy;
use crate::scheduler::SchedulerPolicy;
use crate::tensor::TensorRef;

fn matmul_plan(m: usize, k: usize, n: usize) -> ModelPlan {
    let op = OperatorPlan {
        id: OperatorId(1),
        kind: OperatorKind::DenseMatmul,
        attention: None,
        inputs: vec![
            TensorRef {
                name: "a".into(),
                shape: vec![m, k],
                dtype: DType::F32,
                state_id: None,
            },
            TensorRef {
                name: "b".into(),
                shape: vec![k, n],
                dtype: DType::F32,
                state_id: None,
            },
        ],
        outputs: vec![TensorRef {
            name: "c".into(),
            shape: vec![m, n],
            dtype: DType::F32,
            state_id: None,
        }],
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps: vec![],
    };
    ModelPlan::new_unchecked(
        ModelId(1),
        "toy",
        "test",
        vec![op],
        vec![],
        SchedulerPolicy::Eager,
        16,
        1,
    )
    .validate_for_test()
}

#[test]
fn dense_matmul_two_by_two() {
    let plan = matmul_plan(2, 2, 2);
    let interp = ReferenceInterpreter::new(plan);
    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
    inputs.insert("b".to_string(), vec![5.0, 6.0, 7.0, 8.0]);
    let out = interp.run(&inputs).unwrap();
    let c = &out.outputs["c"];
    // 1*5+2*7=19, 1*6+2*8=22, 3*5+4*7=43, 3*6+4*8=50
    assert_eq!(c, &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn add_elementwise() {
    let op = OperatorPlan {
        id: OperatorId(1),
        kind: OperatorKind::Add,
        attention: None,
        inputs: vec![
            TensorRef {
                name: "a".into(),
                shape: vec![3],
                dtype: DType::F32,
                state_id: None,
            },
            TensorRef {
                name: "b".into(),
                shape: vec![3],
                dtype: DType::F32,
                state_id: None,
            },
        ],
        outputs: vec![TensorRef {
            name: "c".into(),
            shape: vec![3],
            dtype: DType::F32,
            state_id: None,
        }],
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps: vec![],
    };
    let plan = ModelPlan::new_unchecked(
        ModelId(1),
        "toy",
        "test",
        vec![op],
        vec![],
        SchedulerPolicy::Eager,
        16,
        1,
    )
    .validate_for_test();
    let interp = ReferenceInterpreter::new(plan);
    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), vec![1.0, 2.0, 3.0]);
    inputs.insert("b".to_string(), vec![10.0, 20.0, 30.0]);
    let out = interp.run(&inputs).unwrap();
    assert_eq!(out.outputs["c"], vec![11.0, 22.0, 33.0]);
}

#[test]
fn rms_norm_produces_unit_norm_squared_mean_plus_eps() {
    let op = OperatorPlan {
        id: OperatorId(1),
        kind: OperatorKind::RmsNorm,
        attention: None,
        inputs: vec![TensorRef {
            name: "x".into(),
            shape: vec![4],
            dtype: DType::F32,
            state_id: None,
        }],
        outputs: vec![TensorRef {
            name: "y".into(),
            shape: vec![4],
            dtype: DType::F32,
            state_id: None,
        }],
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps: vec![],
    };
    let plan = ModelPlan::new_unchecked(
        ModelId(1),
        "toy",
        "test",
        vec![op],
        vec![],
        SchedulerPolicy::Eager,
        16,
        1,
    )
    .validate_for_test();
    let interp = ReferenceInterpreter::new(plan);
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
    let out = interp.run(&inputs).unwrap();
    // mean(x^2) = (1+4+9+16)/4 = 7.5; rms = sqrt(7.5 + 1e-5) ~ 2.7386
    let y = &out.outputs["y"];
    assert!((y[0] - 1.0 / 2.7386).abs() < 1e-3);
    assert!((y[3] - 4.0 / 2.7386).abs() < 1e-3);
}

#[test]
fn softmax_sums_to_one_and_is_max_at_largest_input() {
    let op = OperatorPlan {
        id: OperatorId(1),
        kind: OperatorKind::Softmax,
        attention: None,
        inputs: vec![TensorRef {
            name: "x".into(),
            shape: vec![3],
            dtype: DType::F32,
            state_id: None,
        }],
        outputs: vec![TensorRef {
            name: "y".into(),
            shape: vec![3],
            dtype: DType::F32,
            state_id: None,
        }],
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps: vec![],
    };
    let plan = ModelPlan::new_unchecked(
        ModelId(1),
        "toy",
        "test",
        vec![op],
        vec![],
        SchedulerPolicy::Eager,
        16,
        1,
    )
    .validate_for_test();
    let interp = ReferenceInterpreter::new(plan);
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vec![0.0, 1.0, 2.0]);
    let out = interp.run(&inputs).unwrap();
    let y = &out.outputs["y"];
    let sum: f32 = y.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    let argmax = y
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(argmax, 2);
}

#[test]
fn swiglu_matches_formula() {
    let op = OperatorPlan {
        id: OperatorId(1),
        kind: OperatorKind::SwiGLU,
        attention: None,
        inputs: vec![
            TensorRef {
                name: "g".into(),
                shape: vec![2],
                dtype: DType::F32,
                state_id: None,
            },
            TensorRef {
                name: "u".into(),
                shape: vec![2],
                dtype: DType::F32,
                state_id: None,
            },
        ],
        outputs: vec![TensorRef {
            name: "y".into(),
            shape: vec![2],
            dtype: DType::F32,
            state_id: None,
        }],
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps: vec![],
    };
    let plan = ModelPlan::new_unchecked(
        ModelId(1),
        "toy",
        "test",
        vec![op],
        vec![],
        SchedulerPolicy::Eager,
        16,
        1,
    )
    .validate_for_test();
    let interp = ReferenceInterpreter::new(plan);
    let mut inputs = HashMap::new();
    inputs.insert("g".to_string(), vec![0.0, 1.0]);
    inputs.insert("u".to_string(), vec![2.0, 3.0]);
    let out = interp.run(&inputs).unwrap();
    // silu(0) = 0, silu(1) = 1/(1+e^-1) ~ 0.7311
    let y = &out.outputs["y"];
    assert!((y[0] - 0.0).abs() < 1e-5);
    assert!((y[1] - 0.7311 * 3.0).abs() < 1e-3);
}

#[test]
fn copy_returns_same_values() {
    let op = OperatorPlan {
        id: OperatorId(1),
        kind: OperatorKind::Copy,
        attention: None,
        inputs: vec![TensorRef {
            name: "x".into(),
            shape: vec![3],
            dtype: DType::F32,
            state_id: None,
        }],
        outputs: vec![TensorRef {
            name: "y".into(),
            shape: vec![3],
            dtype: DType::F32,
            state_id: None,
        }],
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps: vec![],
    };
    let plan = ModelPlan::new_unchecked(
        ModelId(1),
        "toy",
        "test",
        vec![op],
        vec![],
        SchedulerPolicy::Eager,
        16,
        1,
    )
    .validate_for_test();
    let interp = ReferenceInterpreter::new(plan);
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vec![7.0, 8.0, 9.0]);
    let out = interp.run(&inputs).unwrap();
    assert_eq!(out.outputs["y"], vec![7.0, 8.0, 9.0]);
}

#[test]
fn unsupported_operator_returns_typed_error() {
    let op = OperatorPlan {
        id: OperatorId(1),
        kind: OperatorKind::Embedding,
        attention: None,
        // Embedding expects (idx, weight); we pass both but the
        // reference interpreter does not implement the kind itself,
        // so we should get UnsupportedOperator at run time.
        inputs: vec![
            TensorRef {
                name: "idx".into(),
                shape: vec![2],
                dtype: DType::I8,
                state_id: None,
            },
            TensorRef {
                name: "weight".into(),
                shape: vec![4, 3],
                dtype: DType::F32,
                state_id: None,
            },
        ],
        outputs: vec![TensorRef {
            name: "y".into(),
            shape: vec![2, 3],
            dtype: DType::F32,
            state_id: None,
        }],
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps: vec![],
    };
    let plan = ModelPlan::new_unchecked(
        ModelId(1),
        "toy",
        "test",
        vec![op],
        vec![],
        SchedulerPolicy::Eager,
        16,
        1,
    )
    .validate_for_test();
    let interp = ReferenceInterpreter::new(plan);
    let mut inputs = HashMap::new();
    inputs.insert("idx".to_string(), vec![0.0, 1.0]);
    inputs.insert("weight".to_string(), vec![0.0; 12]);
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
fn missing_input_is_reported() {
    let plan = matmul_plan(2, 2, 2);
    let interp = ReferenceInterpreter::new(plan);
    let inputs = HashMap::new(); // empty
    let err = interp.run(&inputs).unwrap_err();
    match err {
        InterpreterError::MissingInput { operator, name } => {
            assert_eq!(operator, OperatorId(1));
            assert_eq!(name, "a");
        }
        other => panic!("expected MissingInput, got {:?}", other),
    }
}

#[test]
fn shape_mismatch_is_reported() {
    let plan = matmul_plan(2, 2, 2);
    let interp = ReferenceInterpreter::new(plan);
    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), vec![1.0, 2.0, 3.0]); // 3 != 4
    inputs.insert("b".to_string(), vec![4.0, 5.0, 6.0, 7.0]);
    let err = interp.run(&inputs).unwrap_err();
    match err {
        InterpreterError::ShapeMismatch { .. } => {}
        other => panic!("expected ShapeMismatch, got {:?}", other),
    }
}

#[test]
fn chained_plan_runs_in_dependency_order() {
    let op1 = OperatorPlan {
        id: OperatorId(1),
        kind: OperatorKind::Add,
        attention: None,
        inputs: vec![
            TensorRef {
                name: "x".into(),
                shape: vec![2],
                dtype: DType::F32,
                state_id: None,
            },
            TensorRef {
                name: "y".into(),
                shape: vec![2],
                dtype: DType::F32,
                state_id: None,
            },
        ],
        outputs: vec![TensorRef {
            name: "z".into(),
            shape: vec![2],
            dtype: DType::F32,
            state_id: None,
        }],
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps: vec![],
    };
    let op2 = OperatorPlan {
        id: OperatorId(2),
        kind: OperatorKind::Copy,
        attention: None,
        inputs: vec![TensorRef {
            name: "z".into(),
            shape: vec![2],
            dtype: DType::F32,
            state_id: None,
        }],
        outputs: vec![TensorRef {
            name: "w".into(),
            shape: vec![2],
            dtype: DType::F32,
            state_id: None,
        }],
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps: vec![OperatorId(1)],
    };
    let plan = ModelPlan::new_unchecked(
        ModelId(1),
        "toy",
        "test",
        vec![op1, op2],
        vec![],
        SchedulerPolicy::Eager,
        16,
        1,
    )
    .validate_for_test();
    let interp = ReferenceInterpreter::new(plan);
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), vec![1.0, 2.0]);
    inputs.insert("y".to_string(), vec![3.0, 4.0]);
    let out = interp.run(&inputs).unwrap();
    assert_eq!(out.outputs["w"], vec![4.0, 6.0]);
    assert_eq!(out.execution_order, vec![OperatorId(1), OperatorId(2)]);
}