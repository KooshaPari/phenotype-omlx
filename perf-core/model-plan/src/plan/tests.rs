//! Unit tests for [`crate::plan`].
//!
//! Extracted into a sibling file so [`crate::plan::plan`] stays under the
//! 500-line module cap while tests remain close to the code they cover.

use crate::attention::AttentionKind;
use crate::dtype::DType;
use crate::error::PlanError;
use crate::operator::{OperatorId, OperatorKind, OperatorPlan};
use crate::plan::{ModelId, ModelPlan};
use crate::precision::Precision;
use crate::quantization::QuantizationPolicy;
use crate::scheduler::SchedulerPolicy;
use crate::state::{StateId, StateKind, StatePlan};
use crate::tensor::TensorRef;

fn matmul_op(id: u64, m: usize, k: usize, n: usize) -> OperatorPlan {
    OperatorPlan {
        id: OperatorId(id),
        kind: OperatorKind::DenseMatmul,
        attention: None,
        inputs: vec![
            TensorRef {
                name: format!("a{}", id),
                shape: vec![m, k],
                dtype: DType::F32,
                state_id: None,
            },
            TensorRef {
                name: format!("b{}", id),
                shape: vec![k, n],
                dtype: DType::F32,
                state_id: None,
            },
        ],
        outputs: vec![TensorRef {
            name: format!("c{}", id),
            shape: vec![m, n],
            dtype: DType::F32,
            state_id: None,
        }],
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps: vec![],
    }
}

fn two_matmul_plan() -> ModelPlan {
    let mut p = ModelPlan::new_unchecked(
        ModelId(1),
        "toy",
        "test",
        vec![matmul_op(1, 2, 2, 2), matmul_op(2, 2, 2, 2)],
        vec![],
        SchedulerPolicy::Eager,
        16,
        1,
    );
    // Second operator consumes the first's output.
    p.operators[1].inputs[0] = p.operators[0].outputs[0].clone();
    p.operators[1].deps.push(OperatorId(1));
    p
}

#[test]
fn validate_accepts_well_formed_plan() {
    let p = two_matmul_plan();
    assert!(p.validate().is_ok());
    assert_eq!(p.topo_sort().unwrap().len(), 2);
}

#[test]
fn validate_detects_duplicate_operator_ids() {
    let mut p = two_matmul_plan();
    p.operators.push(p.operators[0].clone());
    assert!(matches!(
        p.validate(),
        Err(PlanError::DuplicateOperatorId { .. })
    ));
}

#[test]
fn validate_detects_unknown_dependency() {
    let mut p = two_matmul_plan();
    p.operators[0].deps.push(OperatorId(999));
    assert!(matches!(
        p.validate(),
        Err(PlanError::UnknownDependency { .. })
    ));
}

#[test]
fn validate_detects_unknown_state_owner() {
    let mut p = two_matmul_plan();
    p.states.push(StatePlan {
        id: StateId(7),
        kind: StateKind::KvCache,
        persistent: true,
        shape: vec![2, 2],
        dtype: DType::F32,
        owner_operator: OperatorId(999),
        max_versions: 1,
    });
    assert!(matches!(
        p.validate(),
        Err(PlanError::UnknownStateOwner { .. })
    ));
}

#[test]
fn validate_detects_moe_top_k_above_num_experts() {
    let mut p = two_matmul_plan();
    p.scheduler = SchedulerPolicy::Moe {
        capacity_factor: 1.25,
        top_k: 16,
        num_experts: 8,
    };
    assert!(matches!(
        p.validate(),
        Err(PlanError::InvalidScheduler { .. })
    ));
}

#[test]
fn validate_detects_invalid_quant_policy() {
    let mut p = two_matmul_plan();
    p.operators[0].quant = QuantizationPolicy::Ternary {
        group_size: 0,
        bits: 2,
    };
    assert!(matches!(
        p.validate(),
        Err(PlanError::InvalidQuantPolicy { .. })
    ));
}

#[test]
fn validate_detects_dimension_overflow() {
    let mut p = two_matmul_plan();
    p.operators[0].inputs[0].shape = vec![usize::MAX];
    assert!(matches!(
        p.validate(),
        Err(PlanError::DimensionOverflow { .. })
    ));
}

#[test]
fn validate_detects_arity_mismatch() {
    let mut p = two_matmul_plan();
    // DenseMatmul wants 2 inputs; drop one.
    p.operators[0].inputs.pop();
    assert!(matches!(
        p.validate(),
        Err(PlanError::MalformedOperator { .. })
    ));
}

#[test]
fn validate_detects_binary_op_dtype_mismatch() {
    let mut p = two_matmul_plan();
    // Make operator 0 an Add with mismatched dtypes.
    p.operators[0].kind = OperatorKind::Add;
    p.operators[0].inputs[0].dtype = DType::F32;
    p.operators[0].inputs[1].dtype = DType::I8;
    assert!(matches!(p.validate(), Err(PlanError::DtypeMismatch { .. })));
}

#[test]
fn validate_detects_unknown_tensor_state_ref() {
    let mut p = two_matmul_plan();
    p.operators[0].inputs[0].state_id = Some(StateId(123));
    assert!(matches!(
        p.validate(),
        Err(PlanError::MalformedOperator { .. })
    ));
}

#[test]
fn topo_sort_orders_by_deps() {
    let p = two_matmul_plan();
    let order = p.topo_sort().unwrap();
    assert_eq!(order[0].id, OperatorId(1));
    assert_eq!(order[1].id, OperatorId(2));
}

#[test]
fn attention_field_is_optional_in_serde() {
    let op = matmul_op(1, 2, 2, 2);
    let s = serde_json::to_string(&op).unwrap();
    // attention is None and skipped in serialization.
    assert!(!s.contains("attention"));
    let mut op2 = op.clone();
    op2.attention = Some(AttentionKind::Dense);
    let s2 = serde_json::to_string(&op2).unwrap();
    assert!(s2.contains("attention"));
}

#[test]
fn model_plan_round_trip_no_unknown_fields() {
    let p = two_matmul_plan();
    let s = serde_json::to_string(&p).unwrap();
    let back: ModelPlan = serde_json::from_str(&s).unwrap();
    assert_eq!(back, p);
}

#[test]
fn model_plan_rejects_unknown_field() {
    let s = r#"{
        "id": 1,
        "name": "x",
        "model_family": "y",
        "operators": [],
        "states": [],
        "scheduler": {"policy":"eager"},
        "max_seq_len": 16,
        "vocab_size": 1,
        "mystery": 1
    }"#;
    let err = serde_json::from_str::<ModelPlan>(s).unwrap_err();
    assert!(err.to_string().contains("unknown field"));
}
