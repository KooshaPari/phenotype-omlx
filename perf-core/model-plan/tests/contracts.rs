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

#[path = "contracts/contracts_interpreter.rs"]
mod contracts_interpreter;
#[path = "contracts/contracts_planner.rs"]
mod contracts_planner;

use model_plan::{
    DType, ModelId, ModelPlan, OperatorId, OperatorKind, OperatorPlan, Precision,
    QuantizationPolicy, SchedulerPolicy, TensorRef,
};

/// Naive f32 dense matmul oracle: `c[i,j] = sum_k a[i,k] * b[k,j]`.
/// Used as ground truth for `ReferenceInterpreter::run` correctness tests.
pub(crate) fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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
pub(crate) fn naive_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
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
pub(crate) fn single_matmul_plan(m: usize, k: usize, n: usize) -> ModelPlan {
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

#[allow(dead_code)]
fn _keep_imports_used(_: model_plan::StateKind) {}
