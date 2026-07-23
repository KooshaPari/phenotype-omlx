//! Slow reference interpreter used as the correctness oracle for fast
//! Metal/Mojo/Zig kernels.
//!
//! The interpreter runs a [`crate::ModelPlan`] on tiny deterministic
//! tensors (f32 only, ≤16 elements per tensor in practice). It is
//! deliberately naive: triple-loop matmul, scalar SwiGLU, sequential
//! execution. Performance is not the goal — *correctness* is, so that
//! optimized candidates can be diffed against this oracle during
//! selector promotion.
//!
//! Operators supported here are intentionally a small subset of
//! [`crate::operator::OperatorKind`]. Anything not in the subset
//! returns [`crate::error::InterpreterError::UnsupportedOperator`].

use std::collections::HashMap;

use crate::error::{InterpreterError, InterpreterResult};
use crate::operator::{OperatorId, OperatorKind};
use crate::plan::ModelPlan;

/// The result of a single [`crate::ReferenceInterpreter::run`] call.
///
/// `outputs` maps output tensor name to its computed values. For
/// chained plans (operator B consumes operator A's output), the
/// intermediate values are also recorded under their declared names so
/// callers can assert on intermediate state.
#[derive(Debug, Clone, PartialEq)]
pub struct StepOutputs {
    /// Map of tensor name → computed f32 buffer.
    pub outputs: HashMap<String, Vec<f32>>,
    /// Order in which operators were executed (by id). Useful for
    /// chained plans where the caller wants to assert on the
    /// intermediate execution order.
    pub execution_order: Vec<OperatorId>,
}

/// Slow reference interpreter.
///
/// Validates the plan on construction so a bad plan never silently
/// executes. Stores a topologically sorted copy of the operators so each
/// `run` is allocation-cheap.
#[derive(Debug, Clone)]
pub struct ReferenceInterpreter {
    plan: ModelPlan,
    order: Vec<OperatorId>,
}

impl ReferenceInterpreter {
    /// Construct the interpreter. Panics if the plan fails validation
    /// or cannot be topologically sorted — an unvalidated plan is a
    /// programmer error.
    pub fn new(plan: ModelPlan) -> Self {
        plan.validate()
            .expect("ReferenceInterpreter: plan must validate before construction");
        let order = plan
            .topo_sort()
            .expect("ReferenceInterpreter: validated plan must topo-sort")
            .into_iter()
            .map(|op| op.id)
            .collect();
        Self { plan, order }
    }

    /// Execute the plan with the supplied input tensors.
    ///
    /// `inputs` maps input tensor **name** to f32 values. Outputs of one
    /// operator are stored under their declared output names and become
    /// available as inputs to dependent operators (matched by name).
    pub fn run(&self, inputs: &HashMap<String, Vec<f32>>) -> InterpreterResult<StepOutputs> {
        let mut arena: HashMap<String, Vec<f32>> = inputs.clone();
        let mut execution_order = Vec::with_capacity(self.order.len());

        for op_id in &self.order {
            let op = self
                .plan
                .operator(*op_id)
                .expect("ReferenceInterpreter: operator must exist");
            execute_one(op, &mut arena)?;
            execution_order.push(*op_id);
        }

        Ok(StepOutputs {
            outputs: arena,
            execution_order,
        })
    }

    /// Borrow the plan.
    pub fn plan(&self) -> &ModelPlan {
        &self.plan
    }
}

fn execute_one(
    op: &crate::operator::OperatorPlan,
    arena: &mut HashMap<String, Vec<f32>>,
) -> InterpreterResult<()> {
    match op.kind {
        OperatorKind::DenseMatmul => run_dense_matmul(op, arena),
        OperatorKind::Add => run_add(op, arena),
        OperatorKind::RmsNorm => run_rms_norm(op, arena),
        OperatorKind::Softmax => run_softmax(op, arena),
        OperatorKind::SwiGLU => run_swiglu(op, arena),
        OperatorKind::Copy => run_copy(op, arena),
        _ => Err(InterpreterError::UnsupportedOperator {
            operator: op.id,
            kind: op.kind.clone(),
        }),
    }
}

fn fetch_input<'a>(
    op: &crate::operator::OperatorPlan,
    arena: &'a HashMap<String, Vec<f32>>,
    slot: &crate::tensor::TensorRef,
) -> InterpreterResult<&'a Vec<f32>> {
    arena
        .get(&slot.name)
        .ok_or_else(|| InterpreterError::MissingInput {
            operator: op.id,
            name: slot.name.clone(),
        })
}

fn check_shape(
    op: &crate::operator::OperatorPlan,
    name: &str,
    shape: &[usize],
    actual: &[f32],
) -> InterpreterResult<()> {
    let mut expected: usize = 1;
    for d in shape {
        expected = expected.saturating_mul(*d);
    }
    if actual.len() != expected {
        return Err(InterpreterError::ShapeMismatch {
            operator: op.id,
            name: name.to_string(),
            shape: shape.to_vec(),
            expected,
            actual: actual.len(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Operator implementations
// ---------------------------------------------------------------------------

fn run_dense_matmul(
    op: &crate::operator::OperatorPlan,
    arena: &mut HashMap<String, Vec<f32>>,
) -> InterpreterResult<()> {
    if op.inputs.len() != 2 || op.outputs.len() != 1 {
        return Err(InterpreterError::ArityMismatch {
            operator: op.id,
            expected: 2,
            actual: op.inputs.len(),
        });
    }
    let a_slot = &op.inputs[0];
    let b_slot = &op.inputs[1];
    let c_slot = &op.outputs[0];

    let a = fetch_input(op, arena, a_slot)?;
    let b = fetch_input(op, arena, b_slot)?;
    check_shape(op, &a_slot.name, &a_slot.shape, a)?;
    check_shape(op, &b_slot.name, &b_slot.shape, b)?;

    let m = a_slot.shape.first().copied().unwrap_or(1);
    let k = a_slot.shape.get(1).copied().unwrap_or(1);
    let k_b = b_slot.shape.first().copied().unwrap_or(1);
    let n = b_slot.shape.get(1).copied().unwrap_or(1);
    if k != k_b {
        return Err(InterpreterError::ShapeMismatch {
            operator: op.id,
            name: format!("{}x{}", a_slot.name, b_slot.name),
            shape: vec![m, k, n],
            expected: m * n,
            actual: a.len(),
        });
    }

    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    arena.insert(c_slot.name.clone(), c);
    Ok(())
}

fn run_add(
    op: &crate::operator::OperatorPlan,
    arena: &mut HashMap<String, Vec<f32>>,
) -> InterpreterResult<()> {
    if op.inputs.len() != 2 || op.outputs.len() != 1 {
        return Err(InterpreterError::ArityMismatch {
            operator: op.id,
            expected: 2,
            actual: op.inputs.len(),
        });
    }
    let a = fetch_input(op, arena, &op.inputs[0])?;
    let b = fetch_input(op, arena, &op.inputs[1])?;
    if a.len() != b.len() {
        return Err(InterpreterError::ShapeMismatch {
            operator: op.id,
            name: format!("{}x{}", op.inputs[0].name, op.inputs[1].name),
            shape: vec![a.len()],
            expected: a.len(),
            actual: b.len(),
        });
    }
    let c: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    arena.insert(op.outputs[0].name.clone(), c);
    Ok(())
}

fn run_rms_norm(
    op: &crate::operator::OperatorPlan,
    arena: &mut HashMap<String, Vec<f32>>,
) -> InterpreterResult<()> {
    if op.outputs.len() != 1 {
        return Err(InterpreterError::ArityMismatch {
            operator: op.id,
            expected: 1,
            actual: op.outputs.len(),
        });
    }
    let x = fetch_input(op, arena, &op.inputs[0])?;
    let eps = 1e-5f32;
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32;
    let rms = (mean_sq + eps).sqrt();
    let y: Vec<f32> = x.iter().map(|v| v / rms).collect();
    arena.insert(op.outputs[0].name.clone(), y);
    Ok(())
}

fn run_softmax(
    op: &crate::operator::OperatorPlan,
    arena: &mut HashMap<String, Vec<f32>>,
) -> InterpreterResult<()> {
    if op.outputs.len() != 1 {
        return Err(InterpreterError::ArityMismatch {
            operator: op.id,
            expected: 1,
            actual: op.outputs.len(),
        });
    }
    let x = fetch_input(op, arena, &op.inputs[0])?;
    if x.is_empty() {
        arena.insert(op.outputs[0].name.clone(), Vec::new());
        return Ok(());
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let y: Vec<f32> = exps.iter().map(|e| e / sum).collect();
    arena.insert(op.outputs[0].name.clone(), y);
    Ok(())
}

fn run_swiglu(
    op: &crate::operator::OperatorPlan,
    arena: &mut HashMap<String, Vec<f32>>,
) -> InterpreterResult<()> {
    if op.inputs.len() != 2 || op.outputs.len() != 1 {
        return Err(InterpreterError::ArityMismatch {
            operator: op.id,
            expected: 2,
            actual: op.inputs.len(),
        });
    }
    let gate = fetch_input(op, arena, &op.inputs[0])?;
    let up = fetch_input(op, arena, &op.inputs[1])?;
    if gate.len() != up.len() {
        return Err(InterpreterError::ShapeMismatch {
            operator: op.id,
            name: format!("{}x{}", op.inputs[0].name, op.inputs[1].name),
            shape: vec![gate.len()],
            expected: gate.len(),
            actual: up.len(),
        });
    }
    let y: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| {
            let sig = 1.0 / (1.0 + (-*g).exp());
            let silu = *g * sig;
            silu * *u
        })
        .collect();
    arena.insert(op.outputs[0].name.clone(), y);
    Ok(())
}

fn run_copy(
    op: &crate::operator::OperatorPlan,
    arena: &mut HashMap<String, Vec<f32>>,
) -> InterpreterResult<()> {
    if op.inputs.len() != 1 || op.outputs.len() != 1 {
        return Err(InterpreterError::ArityMismatch {
            operator: op.id,
            expected: 1,
            actual: op.inputs.len(),
        });
    }
    let x = fetch_input(op, arena, &op.inputs[0])?.clone();
    arena.insert(op.outputs[0].name.clone(), x);
    Ok(())
}

#[cfg(test)]
#[path = "interpreter/tests.rs"]
mod tests;
