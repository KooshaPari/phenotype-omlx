//! Operator kinds, identifiers, and operator plans.
//!
//! An `OperatorPlan` is the smallest unit of work the kernel registry
//! schedules. It is *pure description*: no buffer pointers, no event
//! handles, no FFI state. The interpreter and the kernel selector both
//! consume these records and derive their own concrete resources.

use serde::{Deserialize, Serialize};

use crate::attention::AttentionKind;
use crate::precision::Precision;
use crate::quantization::QuantizationPolicy;
use crate::tensor::TensorRef;

/// Stable identifier for an [`OperatorPlan`] within a [`crate::ModelPlan`].
///
/// Newtype around `u64` to keep room for future typed identifiers
/// (uuids, namespaced ids) without changing call sites.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct OperatorId(pub u64);

impl std::fmt::Display for OperatorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "op#{}", self.0)
    }
}

/// What an operator *does*.
///
/// Each variant carries the small structural parameters the kernel
/// selector needs to dispatch to a candidate family. Heavy state (KV
/// caches, expert weights, masks) lives in [`crate::state::StatePlan`]
/// records referenced by [`crate::tensor::TensorRef::state_id`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "op")]
pub enum OperatorKind {
    /// Standard dense matmul: `c[m,n] = a[m,k] @ b[k,n]`.
    DenseMatmul,
    /// Grouped matmul over `groups` independent dense matmuls.
    GroupedMatmul {
        /// Number of independent groups.
        groups: usize,
    },
    /// Rotary position embedding.
    Rope,
    /// RMSNorm: `y = x / sqrt(mean(x^2) + eps) * weight`.
    RmsNorm,
    /// LayerNorm: standard `mean`/`var`/`normalize` with affine.
    LayerNorm,
    /// Softmax along the last axis.
    Softmax,
    /// SwiGLU: `y = silu(gate) * up`.
    SwiGLU,
    /// Gaussian Error Linear Unit activation.
    GeLU,
    /// SiLU (a.k.a. Swish) activation.
    SilU,
    /// Embedding lookup.
    Embedding,
    /// Token sampling (argmax, top-k, top-p, etc.).
    Sampling,
    /// Range construction (e.g. position ids).
    Arange,
    /// Elementwise / structural copy.
    Copy,
    /// Elementwise add (broadcasting).
    Add,
    /// Elementwise multiply (broadcasting).
    Mul,
    /// Scatter write (index, src, dst).
    Scatter,
    /// Gather read (index, src, dst).
    Gather,
}

impl OperatorKind {
    /// Short lowercase tag for selector logs.
    pub fn tag(&self) -> &'static str {
        match self {
            OperatorKind::DenseMatmul => "dense_matmul",
            OperatorKind::GroupedMatmul { .. } => "grouped_matmul",
            OperatorKind::Rope => "rope",
            OperatorKind::RmsNorm => "rms_norm",
            OperatorKind::LayerNorm => "layer_norm",
            OperatorKind::Softmax => "softmax",
            OperatorKind::SwiGLU => "swiglu",
            OperatorKind::GeLU => "gelu",
            OperatorKind::SilU => "silu",
            OperatorKind::Embedding => "embedding",
            OperatorKind::Sampling => "sampling",
            OperatorKind::Arange => "arange",
            OperatorKind::Copy => "copy",
            OperatorKind::Add => "add",
            OperatorKind::Mul => "mul",
            OperatorKind::Scatter => "scatter",
            OperatorKind::Gather => "gather",
        }
    }
}

/// A single operator in a [`crate::ModelPlan`].
///
/// `deps` references other operators by [`OperatorId`]; the runtime
/// topologically sorts operators before execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorPlan {
    /// Unique id within the enclosing plan.
    pub id: OperatorId,
    /// What the operator computes.
    pub kind: OperatorKind,
    /// Attention family for attention-shaped operators (set for `Rope`,
    /// attention variants modeled as their own operator kind in later
    /// tasks, or any operator that should be dispatched through the
    /// attention selector).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention: Option<AttentionKind>,
    /// Named input tensors.
    pub inputs: Vec<TensorRef>,
    /// Named output tensors.
    pub outputs: Vec<TensorRef>,
    /// Logical precision policy.
    pub precision: Precision,
    /// Quantization policy.
    pub quant: QuantizationPolicy,
    /// Operator ids that must run before this one. The runtime computes
    /// a topological order from these edges.
    #[serde(default)]
    pub deps: Vec<OperatorId>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    #[test]
    fn tag_for_each_variant() {
        assert_eq!(OperatorKind::DenseMatmul.tag(), "dense_matmul");
        assert_eq!(
            OperatorKind::GroupedMatmul { groups: 4 }.tag(),
            "grouped_matmul"
        );
        assert_eq!(OperatorKind::Rope.tag(), "rope");
        assert_eq!(OperatorKind::SwiGLU.tag(), "swiglu");
        assert_eq!(OperatorKind::Arange.tag(), "arange");
        assert_eq!(OperatorKind::Copy.tag(), "copy");
        assert_eq!(OperatorKind::Add.tag(), "add");
        assert_eq!(OperatorKind::Mul.tag(), "mul");
    }

    #[test]
    fn serde_round_trip_for_kind() {
        let variants = vec![
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
        for v in variants {
            let s = serde_json::to_string(&v).unwrap();
            let back: OperatorKind = serde_json::from_str(&s).unwrap();
            assert_eq!(back, v);
        }
    }

    #[test]
    fn serde_rejects_unknown_field_on_operator_plan() {
        let s = r#"{
            "id": 1,
            "op": "dense_matmul",
            "inputs": [],
            "outputs": [],
            "precision": "fp32",
            "quant": {"scheme":"dense"},
            "deps": [],
            "mystery": 1
        }"#;
        let err = serde_json::from_str::<OperatorPlan>(s).unwrap_err();
        assert!(err.to_string().contains("unknown field"));
    }

    #[test]
    fn operator_plan_round_trips() {
        let op = OperatorPlan {
            id: OperatorId(1),
            kind: OperatorKind::DenseMatmul,
            attention: None,
            inputs: vec![TensorRef {
                name: "a".into(),
                shape: vec![2, 3],
                dtype: DType::F32,
                state_id: None,
            }],
            outputs: vec![],
            precision: Precision::Fp32,
            quant: QuantizationPolicy::Dense,
            deps: vec![],
        };
        let s = serde_json::to_string(&op).unwrap();
        let back: OperatorPlan = serde_json::from_str(&s).unwrap();
        assert_eq!(back, op);
    }
}