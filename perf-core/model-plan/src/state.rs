//! State slots carried across steps.
//!
//! `StatePlan` describes persistent runtime state (KV caches, RNN state,
//! MoE router scratch, diffusion masks, sparse slot maps). The state is
//! identified by [`StateId`], owned by exactly one operator (the one that
//! mutates it), and optionally carries `max_versions` for cache-eviction
//! style uses (e.g. a Speculative decoding draft buffer).

use serde::{Deserialize, Serialize};

use crate::dtype::DType;
use crate::operator::OperatorId;

/// Stable identifier for a [`StatePlan`] within a [`crate::ModelPlan`].
///
/// Newtype around `u64` so `serde` derives are ergonomic and to keep room
/// for future typed identifiers (e.g. namespaced uuids).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct StateId(pub u64);

impl std::fmt::Display for StateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "state#{}", self.0)
    }
}

/// What kind of state the slot represents.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum StateKind {
    /// Standard KV cache for autoregressive attention.
    KvCache,

    /// Compressed-latent cache for MLA.
    MlaCache,

    /// Compressed-context cache for CCA.
    CcaCache,

    /// DeltaNet-style linear-attention state with a fixed state dimension.
    DeltaNetState {
        /// State dimension per head.
        state_dim: usize,
    },

    /// Generic RNN state (Mamba, RWKV, Jamba recurrent block).
    RnnState,

    /// Router scratch for MoE (load stats, expert assignment buffers).
    MoERouterState,

    /// Diffusion active-token mask and confidence buffer.
    DiffusionMask,

    /// Sparse slot map (block-sparse MoE activations, etc.).
    SparseSlotMap,
}

impl StateKind {
    /// Short lowercase tag for logs and selector keys.
    pub fn tag(&self) -> &'static str {
        match self {
            StateKind::KvCache => "kv_cache",
            StateKind::MlaCache => "mla_cache",
            StateKind::CcaCache => "cca_cache",
            StateKind::DeltaNetState { .. } => "deltanet_state",
            StateKind::RnnState => "rnn_state",
            StateKind::MoERouterState => "moe_router_state",
            StateKind::DiffusionMask => "diffusion_mask",
            StateKind::SparseSlotMap => "sparse_slot_map",
        }
    }
}

/// Description of a persistent state slot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatePlan {
    /// Stable id within the enclosing [`crate::ModelPlan`].
    pub id: StateId,
    /// What kind of state this slot represents.
    pub kind: StateKind,
    /// If `true`, the slot survives across requests; otherwise it is
    /// freed at the end of a single generation.
    pub persistent: bool,
    /// Logical shape of the slot.
    pub shape: Vec<usize>,
    /// Storage dtype.
    pub dtype: DType,
    /// Operator that writes this slot. Must exist in the same plan.
    pub owner_operator: OperatorId,
    /// Maximum number of versions to retain (1 for stateless slots).
    /// Used by Speculative draft buffers and paged KV caches.
    pub max_versions: usize,
}

impl StatePlan {
    /// Validate the state record. Does not check `owner_operator`
    /// presence in the plan — that's a plan-level check.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_versions == 0 {
            return Err(format!(
                "state {} max_versions must be >= 1",
                self.id.0
            ));
        }
        for dim in &self.shape {
            if *dim >= (1usize << 40) {
                return Err(format!(
                    "state {} has unreasonable dimension {}",
                    self.id.0, dim
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_plan_rejects_zero_max_versions() {
        let s = StatePlan {
            id: StateId(1),
            kind: StateKind::KvCache,
            persistent: true,
            shape: vec![2, 2],
            dtype: DType::F32,
            owner_operator: OperatorId(1),
            max_versions: 0,
        };
        assert!(s.validate().is_err());
    }

    #[test]
    fn state_plan_accepts_well_formed() {
        let s = StatePlan {
            id: StateId(1),
            kind: StateKind::KvCache,
            persistent: true,
            shape: vec![2, 2],
            dtype: DType::F32,
            owner_operator: OperatorId(1),
            max_versions: 1,
        };
        assert!(s.validate().is_ok());
    }

    #[test]
    fn state_kind_tag_for_each_variant() {
        assert_eq!(StateKind::KvCache.tag(), "kv_cache");
        assert_eq!(StateKind::MlaCache.tag(), "mla_cache");
        assert_eq!(StateKind::CcaCache.tag(), "cca_cache");
        assert_eq!(
            StateKind::DeltaNetState { state_dim: 64 }.tag(),
            "deltanet_state"
        );
        assert_eq!(StateKind::RnnState.tag(), "rnn_state");
        assert_eq!(StateKind::MoERouterState.tag(), "moe_router_state");
        assert_eq!(StateKind::DiffusionMask.tag(), "diffusion_mask");
        assert_eq!(StateKind::SparseSlotMap.tag(), "sparse_slot_map");
    }

    #[test]
    fn serde_rejects_unknown_field() {
        let s = r#"{
            "id": 1,
            "kind": {"kind": "kv_cache"},
            "persistent": true,
            "shape": [2],
            "dtype": "f32",
            "owner_operator": 1,
            "max_versions": 1,
            "extra": true
        }"#;
        let err = serde_json::from_str::<StatePlan>(s).unwrap_err();
        assert!(err.to_string().contains("unknown field"));
    }
}