//! A `TensorRef` is a named slot in a [`crate::ModelPlan`] describing the
//! shape, dtype, and optional backing state of a tensor.
//!
//! `TensorRef` is a *contract*, not a buffer. Buffers live elsewhere (in
//! the interpreter's arena or in the Metal runtime); this struct just
//! pins the shape and dtype that the kernel is allowed to assume.

use serde::{Deserialize, Serialize};

use crate::dtype::DType;
use crate::state::StateId;

/// Logical reference to a tensor slot in a [`crate::ModelPlan`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorRef {
    /// Stable, plan-unique name. Operators reference tensors by name when
    /// binding to runtime buffers.
    pub name: String,
    /// Shape. May be empty for scalar tensors.
    pub shape: Vec<usize>,
    /// Runtime dtype of the storage.
    pub dtype: DType,
    /// If this tensor is backed by a [`StatePlan`] slot (KV cache, RNN
    /// state, etc.), the id of that state. `None` for ephemeral tensors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_id: Option<StateId>,
}

impl TensorRef {
    /// Total element count of the tensor. Returns `1` for scalar (empty
    /// shape) tensors and `0` for any zero-sized dimension.
    pub fn element_count(&self) -> usize {
        if self.shape.is_empty() {
            return 1;
        }
        let mut acc = 1usize;
        for d in &self.shape {
            if *d == 0 {
                return 0;
            }
            // Saturating to avoid panic; callers reject overflow at validate
            // time via [`crate::PlanError::DimensionOverflow`].
            acc = acc.saturating_mul(*d);
        }
        acc
    }

    /// True when `dim` exceeds a sane upper bound for plan-time validation.
    /// `usize::MAX` is the obvious reject; anything above 2^40 is implausible
    /// for the kinds of tensors this plan describes.
    pub fn is_dim_overflowing(&self, dim: usize) -> bool {
        dim >= (1usize << 40)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_count_for_scalar_is_one() {
        let t = TensorRef {
            name: "s".into(),
            shape: vec![],
            dtype: DType::F32,
            state_id: None,
        };
        assert_eq!(t.element_count(), 1);
    }

    #[test]
    fn element_count_for_2d_shape() {
        let t = TensorRef {
            name: "m".into(),
            shape: vec![3, 4],
            dtype: DType::F32,
            state_id: None,
        };
        assert_eq!(t.element_count(), 12);
    }

    #[test]
    fn element_count_for_zero_dim_is_zero() {
        let t = TensorRef {
            name: "z".into(),
            shape: vec![2, 0, 3],
            dtype: DType::F32,
            state_id: None,
        };
        assert_eq!(t.element_count(), 0);
    }

    #[test]
    fn serde_rejects_unknown_field() {
        let s = r#"{"name":"x","shape":[1],"dtype":"f32","weird":true}"#;
        let err = serde_json::from_str::<TensorRef>(s).unwrap_err();
        assert!(err.to_string().contains("unknown field"));
    }
}
