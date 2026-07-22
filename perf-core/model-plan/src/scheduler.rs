//! Scheduler policy describing how a model's operators are executed.
//!
//! The scheduler policy is metadata for the runtime's kernel selector and
//! resource planner. It does not encode schedule timestamps; it encodes
//! *what kind* of execution pattern the model uses so the planner can
//! pick the right batching, pipelining, and routing strategies.

use serde::{Deserialize, Serialize};

/// How the model is executed.
///
/// `Eq` is intentionally *not* derived: [`SchedulerPolicy::Moe`] carries
/// an `f32` capacity factor that has no total equality relation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "policy")]
pub enum SchedulerPolicy {
    /// Eager single-token decode / standard autoregressive.
    Eager,

    /// Pipeline-parallel execution with `stages` stages. Stages are
    /// typically mapped to model layers; the runtime fuses cross-stage
    /// communication with intra-stage compute.
    Pipeline {
        /// Number of pipeline stages. Must be `>= 1`.
        stages: usize,
    },

    /// Recurrent execution (Mamba, RWKV, Jamba hybrid recurrent blocks).
    Recurrent,

    /// Masked-diffusion denoising with `denoise_steps` steps and optional
    /// remasking after each step.
    Diffusion {
        /// Number of denoising steps per generation. Must be `>= 1`.
        denoise_steps: usize,
        /// Whether the scheduler remasks low-confidence tokens between steps.
        remask: bool,
    },

    /// Speculative decoding with a draft model and tree verification.
    Speculative {
        /// Maximum draft tokens per verification round.
        max_draft: usize,
        /// Branching width of the speculative tree.
        tree_width: usize,
        /// Depth of the speculative tree.
        tree_depth: usize,
    },

    /// Sparse Mixture-of-Experts scheduling with capacity and top-k
    /// routing. `top_k` must be `<= num_experts`.
    Moe {
        /// Capacity factor relative to uniform load.
        capacity_factor: f32,
        /// Number of experts selected per token. Must be `<= num_experts`.
        top_k: usize,
        /// Total expert count available for routing. Must be `>= 1`.
        num_experts: usize,
    },
}

impl SchedulerPolicy {
    /// Validate the policy. Returns `Err(reason)` on structural problems.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            SchedulerPolicy::Eager => Ok(()),
            SchedulerPolicy::Recurrent => Ok(()),
            SchedulerPolicy::Pipeline { stages } => {
                if *stages == 0 {
                    return Err("pipeline.stages must be >= 1".to_string());
                }
                Ok(())
            }
            SchedulerPolicy::Diffusion {
                denoise_steps,
                remask: _,
            } => {
                if *denoise_steps == 0 {
                    return Err("diffusion.denoise_steps must be >= 1".to_string());
                }
                Ok(())
            }
            SchedulerPolicy::Speculative {
                max_draft,
                tree_width,
                tree_depth,
            } => {
                if *max_draft == 0 {
                    return Err("speculative.max_draft must be >= 1".to_string());
                }
                if *tree_width == 0 {
                    return Err("speculative.tree_width must be >= 1".to_string());
                }
                if *tree_depth == 0 {
                    return Err("speculative.tree_depth must be >= 1".to_string());
                }
                Ok(())
            }
            SchedulerPolicy::Moe {
                capacity_factor,
                top_k,
                num_experts,
            } => {
                if *num_experts == 0 {
                    return Err("moe.num_experts must be >= 1".to_string());
                }
                if *top_k == 0 {
                    return Err("moe.top_k must be >= 1".to_string());
                }
                if top_k > num_experts {
                    return Err(format!(
                        "moe.top_k ({}) must be <= num_experts ({})",
                        top_k, num_experts
                    ));
                }
                if !capacity_factor.is_finite() || *capacity_factor <= 0.0 {
                    return Err("moe.capacity_factor must be a positive finite f32".to_string());
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eager_and_recurrent_validate() {
        assert!(SchedulerPolicy::Eager.validate().is_ok());
        assert!(SchedulerPolicy::Recurrent.validate().is_ok());
    }

    #[test]
    fn pipeline_rejects_zero_stages() {
        let p = SchedulerPolicy::Pipeline { stages: 0 };
        assert!(p.validate().is_err());
    }

    #[test]
    fn diffusion_rejects_zero_steps() {
        let p = SchedulerPolicy::Diffusion {
            denoise_steps: 0,
            remask: false,
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn moe_rejects_top_k_above_num_experts() {
        let p = SchedulerPolicy::Moe {
            capacity_factor: 1.25,
            top_k: 16,
            num_experts: 8,
        };
        let err = p.validate().unwrap_err();
        assert!(err.contains("top_k") && err.contains("num_experts"));
    }

    #[test]
    fn moe_rejects_zero_top_k() {
        let p = SchedulerPolicy::Moe {
            capacity_factor: 1.25,
            top_k: 0,
            num_experts: 8,
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn moe_rejects_zero_num_experts() {
        let p = SchedulerPolicy::Moe {
            capacity_factor: 1.25,
            top_k: 1,
            num_experts: 0,
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn moe_rejects_non_positive_capacity_factor() {
        let p = SchedulerPolicy::Moe {
            capacity_factor: 0.0,
            top_k: 1,
            num_experts: 4,
        };
        assert!(p.validate().is_err());
        let p = SchedulerPolicy::Moe {
            capacity_factor: f32::NAN,
            top_k: 1,
            num_experts: 4,
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn speculative_rejects_zero_max_draft() {
        let p = SchedulerPolicy::Speculative {
            max_draft: 0,
            tree_width: 2,
            tree_depth: 2,
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn serde_round_trip() {
        let variants = vec![
            SchedulerPolicy::Eager,
            SchedulerPolicy::Pipeline { stages: 2 },
            SchedulerPolicy::Recurrent,
            SchedulerPolicy::Diffusion {
                denoise_steps: 4,
                remask: true,
            },
            SchedulerPolicy::Speculative {
                max_draft: 4,
                tree_width: 3,
                tree_depth: 2,
            },
            SchedulerPolicy::Moe {
                capacity_factor: 1.25,
                top_k: 2,
                num_experts: 8,
            },
        ];
        for v in variants {
            let s = serde_json::to_string(&v).unwrap();
            let back: SchedulerPolicy = serde_json::from_str(&s).unwrap();
            assert_eq!(back, v);
        }
    }
}