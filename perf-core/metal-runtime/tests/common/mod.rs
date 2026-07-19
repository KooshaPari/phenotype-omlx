//! Shared test helpers for the `metal-runtime` integration tests.
//!
//! Cargo treats every `.rs` file directly under `tests/` as a separate
//! integration-test binary; files under `tests/common/` are *not* treated
//! as test binaries (the directory name starts with `common`). Each test
//! binary pulls these helpers in via `mod common;`.

#![allow(dead_code)]

use std::time::{SystemTime, UNIX_EPOCH};

use model_plan::{
    DType, ModelId, ModelPlan, OperatorId, OperatorKind, OperatorPlan, Precision,
    QuantizationPolicy, SchedulerPolicy, TensorRef,
};
use metal_runtime::{DeviceFingerprint, GpuFamily};

/// Current unix epoch in milliseconds. Used to bound fingerprint timestamps.
pub fn tnow_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Validate a plan and panic if it fails. Mirrors the in-crate
/// `validate_for_test` helper but is visible to integration tests.
pub fn validate(plan: ModelPlan) -> ModelPlan {
    plan.validate().expect("contracts.rs: plan must validate");
    plan
}

/// Build a `TensorRef` with `state_id = None`.
pub fn tensor(name: &str, shape: Vec<usize>, dtype: DType) -> TensorRef {
    TensorRef {
        name: name.to_string(),
        shape,
        dtype,
        state_id: None,
    }
}

/// Build a `DenseMatmul` operator.
pub fn op_dense_matmul(
    id: u64,
    inputs: Vec<TensorRef>,
    outputs: Vec<TensorRef>,
    deps: Vec<OperatorId>,
) -> OperatorPlan {
    OperatorPlan {
        id: OperatorId(id),
        kind: OperatorKind::DenseMatmul,
        attention: None,
        inputs,
        outputs,
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps,
    }
}

/// Build an `Add` operator (kept for future tests).
pub fn op_add(
    id: u64,
    inputs: Vec<TensorRef>,
    outputs: Vec<TensorRef>,
    deps: Vec<OperatorId>,
) -> OperatorPlan {
    OperatorPlan {
        id: OperatorId(id),
        kind: OperatorKind::Add,
        attention: None,
        inputs,
        outputs,
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps,
    }
}

/// Build a `Copy` operator.
pub fn op_copy(
    id: u64,
    inputs: Vec<TensorRef>,
    outputs: Vec<TensorRef>,
    deps: Vec<OperatorId>,
) -> OperatorPlan {
    OperatorPlan {
        id: OperatorId(id),
        kind: OperatorKind::Copy,
        attention: None,
        inputs,
        outputs,
        precision: Precision::Fp32,
        quant: QuantizationPolicy::Dense,
        deps,
    }
}

/// A simple 2-op plan: copy -> matmul.
pub fn two_op_plan() -> ModelPlan {
    let plan = ModelPlan::new_unchecked(
        ModelId(1),
        "two-op-plan",
        "test",
        vec![
            op_copy(
                1,
                vec![tensor("x", vec![2, 2], DType::F32)],
                vec![tensor("y", vec![2, 2], DType::F32)],
                vec![],
            ),
            op_dense_matmul(
                2,
                vec![
                    tensor("y", vec![2, 2], DType::F32),
                    tensor("w", vec![2, 2], DType::F32),
                ],
                vec![tensor("z", vec![2, 2], DType::F32)],
                vec![OperatorId(1)],
            ),
        ],
        vec![],
        SchedulerPolicy::Eager,
        16,
        32,
    );
    validate(plan)
}

/// A plan with a self-cycle (op 1 depends on itself). Used to drive a
/// pipeline error.
pub fn self_cycle_plan() -> ModelPlan {
    ModelPlan::new_unchecked(
        ModelId(2),
        "cycle",
        "test",
        vec![op_copy(
            1,
            vec![tensor("x", vec![2], DType::F32)],
            vec![tensor("y", vec![2], DType::F32)],
            vec![OperatorId(1)], // self-cycle
        )],
        vec![],
        SchedulerPolicy::Eager,
        4,
        8,
    )
}

/// A plan referencing a non-existent operator id in `deps`. `validate()`
/// rejects this so we have to use `new_unchecked` and skip validation.
pub fn missing_op_plan() -> ModelPlan {
    ModelPlan::new_unchecked(
        ModelId(3),
        "missing",
        "test",
        vec![op_copy(
            1,
            vec![tensor("x", vec![2], DType::F32)],
            vec![tensor("y", vec![2], DType::F32)],
            vec![OperatorId(99)], // missing
        )],
        vec![],
        SchedulerPolicy::Eager,
        4,
        8,
    )
}

/// Build a synthetic fingerprint with the given GPU family for tests that
/// want host-independent values.
pub fn identity_fp(family: GpuFamily) -> DeviceFingerprint {
    DeviceFingerprint {
        device_name: "test-device".to_string(),
        os: "test-os".to_string(),
        arch: "test-arch".to_string(),
        simd_bit_width: 128,
        total_memory_bytes: 1024 * 1024 * 1024,
        gpu_family: family,
        sysctl_cached: false,
        captured_at_unix_ms: 0,
    }
}