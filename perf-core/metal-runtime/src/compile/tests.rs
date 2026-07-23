//! Tests for [`super`]. Moved verbatim from the original `compile.rs`
//! `#[cfg(test)] mod tests` block during the per-topic module split
//! (turn-13 module-size sweep). Behavior is unchanged.

use super::{plan_revision, BoundedCompiler, CompileBudget};
use model_plan::{
    DType, ModelId, ModelPlan, OperatorId, OperatorKind, OperatorPlan, Precision,
    QuantizationPolicy, SchedulerPolicy, TensorRef,
};

use crate::compile::msl_stub::emit_msl_stub;
use crate::error::CompileError;
use crate::fingerprint::DeviceFingerprint;
use crate::RuntimeMode;

fn small_plan() -> ModelPlan {
    let op = |id: u64, kind: OperatorKind, ins: usize, outs: usize| OperatorPlan {
        id: OperatorId(id),
        kind,
        attention: None,
        inputs: (0..ins)
            .map(|i| TensorRef {
                name: format!("i{}", i),
                shape: vec![1],
                dtype: DType::F32,
                state_id: None,
            })
            .collect(),
        outputs: (0..outs)
            .map(|i| TensorRef {
                name: format!("o{}", i),
                shape: vec![1],
                dtype: DType::F32,
                state_id: None,
            })
            .collect(),
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps: vec![],
    };
    let plan = ModelPlan::new_unchecked(
        ModelId(1),
        "tiny",
        "test",
        vec![op(1, OperatorKind::Copy, 1, 1)],
        vec![],
        SchedulerPolicy::Eager,
        4,
        8,
    );
    plan.validate().expect("tiny plan must validate");
    plan
}

#[test]
fn budget_default_is_generous() {
    let b = CompileBudget::DEFAULT;
    assert!(b.max_ms >= 100);
    assert!(b.max_shader_bytes >= 4096);
}

#[test]
fn production_mode_rejects_source_compilation() {
    let compiler = BoundedCompiler::new(CompileBudget::DEFAULT);
    let error = compiler
        .compile_with_mode(
            &small_plan(),
            &DeviceFingerprint::compute_software(),
            RuntimeMode::Production,
        )
        .expect_err("production must never compile shader source");
    assert_eq!(error, CompileError::SourceCompilationForbidden);
}

#[test]
fn plan_revision_changes_when_op_added() {
    let plan = small_plan();
    let r0 = plan_revision(&plan);
    use model_plan::{
        DType, ModelId, OperatorId, OperatorKind, OperatorPlan, Precision, QuantizationPolicy,
        SchedulerPolicy, TensorRef,
    };
    let bigger = ModelPlan::new_unchecked(
        ModelId(1),
        "tiny",
        "test",
        vec![
            OperatorPlan {
                id: OperatorId(1),
                kind: OperatorKind::Copy,
                attention: None,
                inputs: vec![TensorRef {
                    name: "i0".into(),
                    shape: vec![1],
                    dtype: DType::F32,
                    state_id: None,
                }],
                outputs: vec![TensorRef {
                    name: "o0".into(),
                    shape: vec![1],
                    dtype: DType::F32,
                    state_id: None,
                }],
                precision: Precision::Fp32,
                quant: QuantizationPolicy::Dense,
                deps: vec![],
            },
            OperatorPlan {
                id: OperatorId(2),
                kind: OperatorKind::Copy,
                attention: None,
                inputs: vec![TensorRef {
                    name: "i1".into(),
                    shape: vec![1],
                    dtype: DType::F32,
                    state_id: None,
                }],
                outputs: vec![TensorRef {
                    name: "o1".into(),
                    shape: vec![1],
                    dtype: DType::F32,
                    state_id: None,
                }],
                precision: Precision::Fp32,
                quant: QuantizationPolicy::Dense,
                deps: vec![],
            },
        ],
        vec![],
        SchedulerPolicy::Eager,
        4,
        8,
    );
    let r1 = plan_revision(&bigger);
    assert_ne!(r0, r1);
}

/// Regression test for the metal-runtime <-> kernel-registry
/// dispatch bridge: a plan carrying a sliding-window attention
/// operator must produce an MSL stub line tagged
/// `sliding_window_attention` (the kernel-registry dispatch tag)
/// so trace audits can correlate the plan-side and kernel-side
/// namespaces.
#[test]
fn emit_msl_stub_includes_kernel_tag_for_sliding_window() {
    use model_plan::attention::AttentionKind;
    let op_sw = OperatorPlan {
        id: OperatorId(1),
        kind: OperatorKind::Rope,
        attention: Some(AttentionKind::SlidingWindow { window_size: 256 }),
        inputs: vec![TensorRef {
            name: "q".into(),
            shape: vec![1],
            dtype: DType::F32,
            state_id: None,
        }],
        outputs: vec![TensorRef {
            name: "o".into(),
            shape: vec![1],
            dtype: DType::F32,
            state_id: None,
        }],
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps: vec![],
    };
    let plan = ModelPlan::new_unchecked(
        ModelId(1),
        "qwen3-next",
        "qwen",
        vec![op_sw],
        vec![],
        SchedulerPolicy::Eager,
        4,
        8,
    );
    plan.validate().expect("sw plan must validate");
    let fp = DeviceFingerprint::compute_software();
    let stub = emit_msl_stub(&plan, &fp);
    assert!(
        stub.contains("[kernel=sliding_window_attention"),
        "shader stub must carry sliding_window_attention tag; got:\n{stub}"
    );
    assert!(
        stub.contains("from-plan=rope"),
        "shader stub must carry the plan-side from-plan tag; got:\n{stub}"
    );
}

/// Sibling test for [`emit_msl_stub_includes_kernel_tag_for_sliding_window`]
/// covering GQA: confirms the bridge produces a per-op kernel-tag
/// line for the Qwen3-Coder-Next GQA layers that run alongside
/// the sliding-window ones.
#[test]
fn emit_msl_stub_includes_kernel_tag_for_gqa() {
    use model_plan::attention::AttentionKind;
    let op_gqa = OperatorPlan {
        id: OperatorId(7),
        kind: OperatorKind::Rope,
        attention: Some(AttentionKind::Gqa { kv_heads: 2 }),
        inputs: vec![TensorRef {
            name: "q".into(),
            shape: vec![1],
            dtype: DType::F32,
            state_id: None,
        }],
        outputs: vec![TensorRef {
            name: "o".into(),
            shape: vec![1],
            dtype: DType::F32,
            state_id: None,
        }],
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps: vec![],
    };
    let plan = ModelPlan::new_unchecked(
        ModelId(1),
        "qwen3-next",
        "qwen",
        vec![op_gqa],
        vec![],
        SchedulerPolicy::Eager,
        4,
        8,
    );
    plan.validate().expect("gqa plan must validate");
    let fp = DeviceFingerprint::compute_software();
    let stub = emit_msl_stub(&plan, &fp);
    assert!(
        stub.contains("[kernel=gqa_attention"),
        "shader stub must carry gqa_attention tag; got:\n{stub}"
    );
}

/// Confirms the stub marks operators without a kernel-registry
/// mapping with the literal `no-kernel-tag` token so audits can
/// distinguish deliberate absence from a missing case.
#[test]
fn emit_msl_stub_marks_unmapped_operators_as_no_kernel_tag() {
    let op_softmax = OperatorPlan {
        id: OperatorId(1),
        kind: OperatorKind::Softmax,
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
        deps: vec![],
    };
    let plan = ModelPlan::new_unchecked(
        ModelId(1),
        "smoke",
        "test",
        vec![op_softmax],
        vec![],
        SchedulerPolicy::Eager,
        4,
        8,
    );
    plan.validate().expect("smoke plan must validate");
    let fp = DeviceFingerprint::compute_software();
    let stub = emit_msl_stub(&plan, &fp);
    assert!(
        stub.contains("[kernel=no-kernel-tag"),
        "unmapped operator must be marked no-kernel-tag; got:\n{stub}"
    );
}
