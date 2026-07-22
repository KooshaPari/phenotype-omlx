//! Error types for the model-plan domain crate.
//!
//! Two error enums live here:
//! - [`PlanError`]: errors produced by [`crate::ModelPlan::validate`].
//! - [`InterpreterError`]: errors produced by the reference interpreter
//!   during [`crate::ReferenceInterpreter::run`].
//!
//! Errors carry enough structured context (operator id, tensor name,
//! offending value) that callers can surface concise, actionable messages
//! without re-deriving context from string messages.

use crate::operator::OperatorId;
use crate::operator::OperatorKind;
use crate::state::StateId;
use thiserror::Error;

/// Errors produced while validating a [`crate::ModelPlan`].
///
/// `#[non_exhaustive]` lets us add new variants without breaking
/// downstream exhaustive matches.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanError {
    /// Two operators in the same plan share an [`OperatorId`].
    #[error("duplicate operator id {id} in plan")]
    DuplicateOperatorId { id: OperatorId },

    /// An operator's `deps` entry refers to a missing operator id.
    #[error("operator {operator} depends on unknown operator {dependency}")]
    UnknownDependency {
        operator: OperatorId,
        dependency: OperatorId,
    },

    /// A state record's `owner_operator` refers to a missing operator id.
    #[error("state {state} owned by unknown operator {operator}")]
    UnknownStateOwner {
        state: StateId,
        operator: OperatorId,
    },

    /// A tensor dimension is unreasonably large (e.g. `usize::MAX`). Used
    /// as a defensive overflow gate during validation.
    #[error("operator {operator} has unreasonable tensor dimension {dim}")]
    DimensionOverflow { operator: OperatorId, dim: usize },

    /// A quantization policy was structurally invalid (e.g. `group_size = 0`).
    #[error("invalid quantization policy for operator {operator}: {reason}")]
    InvalidQuantPolicy {
        operator: OperatorId,
        reason: String,
    },

    /// A scheduler policy was structurally invalid (e.g. `top_k > num_experts`).
    #[error("invalid scheduler policy: {reason}")]
    InvalidScheduler { reason: String },

    /// An operator's inputs and outputs are inconsistent (mismatched count
    /// for the chosen `OperatorKind`, etc.).
    #[error("operator {operator} is malformed: {reason}")]
    MalformedOperator {
        operator: OperatorId,
        reason: String,
    },

    /// An operator's inputs refer to tensors with mismatched dtypes for a
    /// binary op, or otherwise incompatible element types.
    #[error("operator {operator} dtype mismatch: {reason}")]
    DtypeMismatch {
        operator: OperatorId,
        reason: String,
    },
}

/// Errors produced by [`crate::ReferenceInterpreter::run`].
///
/// Carries enough context to attribute the failure to a specific operator
/// or tensor in the plan.
#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum InterpreterError {
    /// The operator kind is recognized by the plan but the slow reference
    /// interpreter does not implement it.
    #[error("operator {operator} of kind {kind:?} is not implemented in the reference interpreter")]
    UnsupportedOperator {
        operator: OperatorId,
        kind: OperatorKind,
    },

    /// A required input tensor was not supplied by the caller.
    #[error("missing input tensor '{name}' for operator {operator}")]
    MissingInput {
        operator: OperatorId,
        name: String,
    },

    /// A tensor's runtime element count does not match its declared shape.
    #[error(
        "tensor '{name}' for operator {operator} has shape {shape:?} (={expected} elements) but {actual} values were supplied"
    )]
    ShapeMismatch {
        operator: OperatorId,
        name: String,
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },

    /// An operator requires more inputs than the plan declared.
    #[error("operator {operator} expected {expected} inputs, plan declared {actual}")]
    ArityMismatch {
        operator: OperatorId,
        expected: usize,
        actual: usize,
    },

    /// The plan contains a dependency cycle and cannot be topologically
    /// ordered.
    #[error("operator graph contains a cycle")]
    Cycle,
}

/// Crate-wide result alias for fallible plan operations.
pub type PlanResult<T> = Result<T, PlanError>;

/// Crate-wide result alias for the reference interpreter.
pub type InterpreterResult<T> = Result<T, InterpreterError>;