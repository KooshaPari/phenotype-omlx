use model_plan::{
    DType, InterpreterError, ModelId, ModelPlan, OperatorId, OperatorKind, OperatorPlan, Precision,
    QuantizationPolicy, ReferenceInterpreter, SchedulerPolicy, TensorRef,
};

use super::{naive_matmul, naive_swiglu, single_matmul_plan};

#[test]
fn reference_interpreter_dense_matmul_matches_naive_oracle() {
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
