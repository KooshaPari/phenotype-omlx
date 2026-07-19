//! `ModelPlan`: the top-level domain record describing a model.
//!
//! A `ModelPlan` is a *description*, not a runtime. It contains:
//! - the operators that compose the model, with their dependencies,
//!   precision policy, and quantization layout;
//! - persistent state slots owned by operators (KV caches, RNN state,
//!   MoE scratch);
//! - a scheduler policy describing the execution pattern.
//!
//! Validation is pure and eager: [`ModelPlan::validate`] returns
//! `Err(PlanError)` for any structural defect the runtime cannot
//! silently recover from. The reference interpreter additionally
//! re-validates on construction so ad-hoc test plans are caught even
//! when callers skip the explicit validate call.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::error::{PlanError, PlanResult};
use crate::operator::{OperatorId, OperatorKind, OperatorPlan};
use crate::quantization::QuantizationPolicy;
use crate::scheduler::SchedulerPolicy;
use crate::state::{StateId, StatePlan};
use crate::tensor::TensorRef;

/// Stable identifier for a [`ModelPlan`].
///
/// Newtype around `u64`; future revisions can swap in a uuid without
/// changing call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(pub u64);

/// A complete model description.
///
/// `Eq` is intentionally *not* derived: [`ModelPlan::scheduler`] is a
/// [`SchedulerPolicy`] which carries an `f32` (no total equality). Use
/// [`ModelPlan::validate`] for structural equality checks where it
/// matters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPlan {
    /// Stable id for the plan.
    pub id: ModelId,
    /// Human-readable name (e.g. `"Qwen3-Coder-Next-Instruct"`).
    pub name: String,
    /// Family tag (e.g. `"qwen"`, `"deepseek"`, `"bonsai"`). Used by the
    /// kernel selector to bias candidate selection.
    pub model_family: String,
    /// All operators in the model. Order is irrelevant; the runtime
    /// topologically sorts from [`OperatorPlan::deps`].
    pub operators: Vec<OperatorPlan>,
    /// Persistent state slots.
    pub states: Vec<StatePlan>,
    /// Execution pattern (eager, pipeline, MoE, diffusion, ...).
    pub scheduler: SchedulerPolicy,
    /// Maximum sequence length the model supports.
    pub max_seq_len: usize,
    /// Vocabulary size of the tokenizer.
    pub vocab_size: usize,
}

impl ModelPlan {
    /// Construct a model plan without validating. Use [`ModelPlan::validate`]
    /// before relying on the plan's structural integrity.
    pub fn new_unchecked(
        id: ModelId,
        name: impl Into<String>,
        model_family: impl Into<String>,
        operators: Vec<OperatorPlan>,
        states: Vec<StatePlan>,
        scheduler: SchedulerPolicy,
        max_seq_len: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            model_family: model_family.into(),
            operators,
            states,
            scheduler,
            max_seq_len,
            vocab_size,
        }
    }

    /// Validate the plan and panic if it is malformed. Intended for
    /// in-crate tests that build ad-hoc plans; production code should
    /// call [`ModelPlan::validate`] explicitly and handle the error.
    #[cfg(test)]
    pub fn validate_for_test(self) -> Self {
        self.validate()
            .expect("ModelPlan::validate_for_test: plan must validate");
        self
    }

    /// Validate a single quantization policy against an operator. Used by
    /// [`ModelPlan::validate`] and exposed publicly so callers that build
    /// operator records incrementally can check before assembling the plan.
    pub fn validate_quant(q: &QuantizationPolicy) -> PlanResult<()> {
        q.validate().map_err(|reason| PlanError::InvalidQuantPolicy {
            operator: OperatorId(0),
            reason,
        })
    }

    /// Validate the plan. Returns `Err(PlanError)` on the first defect
    /// found. The order of checks is deterministic so test failures
    /// attribute cleanly.
    pub fn validate(&self) -> PlanResult<()> {
        // 1. Operator id uniqueness.
        let mut seen_ops: HashSet<OperatorId> = HashSet::with_capacity(self.operators.len());
        for op in &self.operators {
            if !seen_ops.insert(op.id) {
                return Err(PlanError::DuplicateOperatorId { id: op.id });
            }
        }

        // 2. Per-operator: quant policy, dimension overflow, malformed arity.
        for op in &self.operators {
            op.quant
                .validate()
                .map_err(|reason| PlanError::InvalidQuantPolicy {
                    operator: op.id,
                    reason,
                })?;
            for t in op.inputs.iter().chain(op.outputs.iter()) {
                for &d in &t.shape {
                    if d >= (1usize << 40) {
                        return Err(PlanError::DimensionOverflow {
                            operator: op.id,
                            dim: d,
                        });
                    }
                }
            }
            Self::check_operator_arity(op)?;
            Self::check_operator_dtype(op)?;
        }

        // 3. Dependency references resolve.
        for op in &self.operators {
            for d in &op.deps {
                if !seen_ops.contains(d) {
                    return Err(PlanError::UnknownDependency {
                        operator: op.id,
                        dependency: *d,
                    });
                }
            }
        }

        // 4. State owner operators exist.
        for s in &self.states {
            s.validate().map_err(|reason| PlanError::MalformedOperator {
                operator: s.owner_operator,
                reason: format!("state {} invalid: {}", s.id.0, reason),
            })?;
            if !seen_ops.contains(&s.owner_operator) {
                return Err(PlanError::UnknownStateOwner {
                    state: s.id,
                    operator: s.owner_operator,
                });
            }
        }

        // 5. State id uniqueness.
        let mut seen_states: HashSet<StateId> = HashSet::with_capacity(self.states.len());
        for s in &self.states {
            if !seen_states.insert(s.id) {
                return Err(PlanError::MalformedOperator {
                    operator: s.owner_operator,
                    reason: format!("duplicate state id {}", s.id.0),
                });
            }
        }

        // 6. Tensor references to states point at real states.
        let state_set: HashSet<StateId> = self.states.iter().map(|s| s.id).collect();
        for op in &self.operators {
            for t in op.inputs.iter().chain(op.outputs.iter()) {
                if let Some(sid) = t.state_id {
                    if !state_set.contains(&sid) {
                        return Err(PlanError::MalformedOperator {
                            operator: op.id,
                            reason: format!(
                                "tensor '{}' references unknown state {}",
                                t.name, sid.0
                            ),
                        });
                    }
                }
            }
        }

        // 7. Scheduler policy structural validity.
        self.scheduler
            .validate()
            .map_err(|reason| PlanError::InvalidScheduler { reason })?;

        Ok(())
    }

    /// Validate the input/output arity for the operator's kind.
    ///
    /// Only operators with strict arity rules are checked here; permissive
    /// kinds (Rope, Embedding, Sampling, ...) accept any non-zero tensor
    /// count as long as tensors are well-formed.
    fn check_operator_arity(op: &OperatorPlan) -> PlanResult<()> {
        let (want_in, want_out): (usize, usize) = match op.kind {
            OperatorKind::DenseMatmul => (2, 1),
            OperatorKind::GroupedMatmul { .. } => (2, 1),
            OperatorKind::RmsNorm | OperatorKind::LayerNorm => (1, 1),
            OperatorKind::Softmax => (1, 1),
            OperatorKind::SwiGLU => (2, 1),
            OperatorKind::GeLU | OperatorKind::SilU => (1, 1),
            OperatorKind::Copy => (1, 1),
            OperatorKind::Add | OperatorKind::Mul => (2, 1),
            OperatorKind::Embedding => (2, 1),
            // Arange, Rope, Sampling, Scatter, Gather have flexible arity
            // that depends on configuration; we do not enforce here.
            _ => return Ok(()),
        };
        if op.inputs.len() != want_in || op.outputs.len() != want_out {
            return Err(PlanError::MalformedOperator {
                operator: op.id,
                reason: format!(
                    "{:?} expects {} input(s) and {} output(s), got {} and {}",
                    op.kind,
                    want_in,
                    want_out,
                    op.inputs.len(),
                    op.outputs.len()
                ),
            });
        }
        Ok(())
    }

    /// Validate dtype compatibility for binary ops. For Add/Mul both inputs
    /// must have the same dtype (no implicit promotion in this plan).
    fn check_operator_dtype(op: &OperatorPlan) -> PlanResult<()> {
        match op.kind {
            OperatorKind::Add | OperatorKind::Mul => {
                if op.inputs.len() == 2 {
                    let a = &op.inputs[0];
                    let b = &op.inputs[1];
                    if a.dtype != b.dtype {
                        return Err(PlanError::DtypeMismatch {
                            operator: op.id,
                            reason: format!(
                                "{:?} inputs must share dtype, got {:?} and {:?}",
                                op.kind, a.dtype, b.dtype
                            ),
                        });
                    }
                }
            }
            OperatorKind::DenseMatmul | OperatorKind::GroupedMatmul { .. } => {
                if op.inputs.len() >= 2 && op.inputs[0].dtype != op.inputs[1].dtype {
                    return Err(PlanError::DtypeMismatch {
                        operator: op.id,
                        reason: format!(
                            "{:?} input dtypes must match, got {:?} and {:?}",
                            op.kind,
                            op.inputs[0].dtype,
                            op.inputs[1].dtype
                        ),
                    });
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Topologically sort operators by [`OperatorPlan::deps`]. Returns
    /// `Err(PlanError::MalformedOperator { .. cycle .. })` when a cycle
    /// is present. Stable: ties broken by [`OperatorId`] order.
    pub fn topo_sort(&self) -> PlanResult<Vec<&OperatorPlan>> {
        self.validate()?; // ensures deps are well-formed first
        let mut in_degree: std::collections::HashMap<OperatorId, usize> =
            self.operators.iter().map(|o| (o.id, o.deps.len())).collect();
        let mut reverse: std::collections::HashMap<OperatorId, Vec<OperatorId>> =
            std::collections::HashMap::new();
        for op in &self.operators {
            for d in &op.deps {
                reverse.entry(*d).or_default().push(op.id);
            }
        }
        let mut ready: std::collections::BTreeSet<OperatorId> = self
            .operators
            .iter()
            .filter(|o| o.deps.is_empty())
            .map(|o| o.id)
            .collect();
        let by_id: std::collections::HashMap<OperatorId, &OperatorPlan> =
            self.operators.iter().map(|o| (o.id, o)).collect();
        let mut out: Vec<&OperatorPlan> = Vec::with_capacity(self.operators.len());
        while let Some(&next) = ready.iter().next() {
            ready.remove(&next);
            out.push(by_id[&next]);
            if let Some(children) = reverse.get(&next) {
                let mut sorted_children: Vec<OperatorId> = children.clone();
                sorted_children.sort();
                for c in sorted_children {
                    if let Some(deg) = in_degree.get_mut(&c) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            ready.insert(c);
                        }
                    }
                }
            }
        }
        if out.len() != self.operators.len() {
            return Err(PlanError::MalformedOperator {
                operator: OperatorId(0),
                reason: "operator graph contains a cycle".to_string(),
            });
        }
        Ok(out)
    }

    /// Look up an operator by id. Returns `None` if absent.
    pub fn operator(&self, id: OperatorId) -> Option<&OperatorPlan> {
        self.operators.iter().find(|o| o.id == id)
    }

    /// Look up a state by id. Returns `None` if absent.
    pub fn state(&self, id: StateId) -> Option<&StatePlan> {
        self.states.iter().find(|s| s.id == id)
    }

    /// Look up a tensor by name across all operators. Returns the first
    /// match in declaration order. Useful for binding runtime buffers.
    pub fn tensor(&self, name: &str) -> Option<&TensorRef> {
        for op in &self.operators {
            for t in op.inputs.iter().chain(op.outputs.iter()) {
                if t.name == name {
                    return Some(t);
                }
            }
        }
        None
    }
}

#[cfg(test)]
#[path = "plan/tests.rs"]
mod tests;