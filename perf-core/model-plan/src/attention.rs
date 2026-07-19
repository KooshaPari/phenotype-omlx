//! Attention variants modeled by the plan.
//!
//! Each variant is a *family* descriptor: it says which attention family
//! the operator uses (and the small structural parameters) without
//! encoding full kernel selection state. The kernel registry derives a
//! quantization- and policy-aware selection key from the operator +
//! [`AttentionKind`] in a later task.

use serde::{Deserialize, Serialize};

/// Family of attention used by an attention-shaped operator.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AttentionKind {
    /// Grouped-Query Attention: `kv_heads` shared across query heads.
    Gqa {
        /// Number of key/value heads.
        kv_heads: usize,
    },

    /// Multi-Latent Attention: compressed KV latent + a rope sub-vector.
    Mla {
        /// Compressed latent dimension (per head).
        d_latent: usize,
        /// Dimension of the rope sub-vector.
        d_rope: usize,
    },

    /// Compressed-Context Attention: aggressive KV compression.
    Cca {
        /// Compression factor applied to the KV cache (e.g. 4 → 4x smaller).
        compressed_factor: usize,
    },

    /// Paged attention: KV cache split into fixed-size pages.
    Paged {
        /// Page size in tokens.
        block_size: usize,
    },

    /// Tree attention (speculative verification): tree-shaped causal mask.
    Tree {
        /// Branching width of the speculative tree.
        width: usize,
        /// Depth of the speculative tree.
        depth: usize,
    },

    /// Vanilla dense attention. All heads share KV.
    Dense,
}

impl AttentionKind {
    /// Short lowercase tag used in selector logs and cache keys.
    pub fn tag(&self) -> &'static str {
        match self {
            AttentionKind::Gqa { .. } => "gqa",
            AttentionKind::Mla { .. } => "mla",
            AttentionKind::Cca { .. } => "cca",
            AttentionKind::Paged { .. } => "paged",
            AttentionKind::Tree { .. } => "tree",
            AttentionKind::Dense => "dense",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_for_each_variant() {
        assert_eq!(AttentionKind::Dense.tag(), "dense");
        assert_eq!(
            AttentionKind::Gqa { kv_heads: 4 }.tag(),
            "gqa"
        );
        assert_eq!(
            AttentionKind::Mla {
                d_latent: 64,
                d_rope: 16
            }
            .tag(),
            "mla"
        );
        assert_eq!(
            AttentionKind::Cca {
                compressed_factor: 4
            }
            .tag(),
            "cca"
        );
        assert_eq!(
            AttentionKind::Paged { block_size: 16 }.tag(),
            "paged"
        );
        assert_eq!(
            AttentionKind::Tree {
                width: 4,
                depth: 3
            }
            .tag(),
            "tree"
        );
    }

    #[test]
    fn serde_round_trip() {
        let variants = vec![
            AttentionKind::Gqa { kv_heads: 4 },
            AttentionKind::Mla {
                d_latent: 64,
                d_rope: 16,
            },
            AttentionKind::Cca {
                compressed_factor: 4,
            },
            AttentionKind::Paged { block_size: 16 },
            AttentionKind::Tree {
                width: 4,
                depth: 3,
            },
            AttentionKind::Dense,
        ];
        for v in variants {
            let s = serde_json::to_string(&v).unwrap();
            let back: AttentionKind = serde_json::from_str(&s).unwrap();
            assert_eq!(back, v);
        }
    }
}