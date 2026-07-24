use model_plan::{
    AttentionKind, DType, ModelId, ModelPlan, OperatorId, OperatorKind, OperatorPlan, Precision,
    QuantizationPolicy, SchedulerPolicy, StateId, StateKind, StatePlan, TensorRef,
};
use serde_json::{json, Value};

use super::single_matmul_plan;

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
    let bad = QuantizationPolicy::Ternary {
        group_size: 0,
        bits: 2,
    };
    let err = ModelPlan::validate_quant(&bad).unwrap_err();
    assert!(matches!(
        err,
        model_plan::PlanError::InvalidQuantPolicy { .. }
    ));
}

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
        model_plan::PlanError::UnknownStateOwner { state, operator } => {
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

#[test]
fn model_plan_rejects_dimension_overflow() {
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

#[test]
fn attention_kind_variants_serialize() {
    let variants = [
        AttentionKind::Gqa { kv_heads: 4 },
        AttentionKind::Mla {
            d_latent: 64,
            d_rope: 16,
        },
        AttentionKind::Cca {
            compressed_factor: 4,
        },
        AttentionKind::Paged { block_size: 16 },
        AttentionKind::Tree { width: 4, depth: 3 },
        AttentionKind::Dense,
    ];
    for v in variants {
        let s = serde_json::to_string(&v).expect("serialize");
        let back: AttentionKind = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(back, v);
    }
}
