//! Error types for the `metal-runtime` crate.
//!
//! Two error enums live here:
//!
//! - [`CompileError`]: returned by [`crate::BoundedCompiler::compile`] when
//!   the shader template or compile time exceeds the configured budget, or
//!   when the plan itself cannot be compiled for some structural reason.
//! - [`PipelineError`]: returned by [`crate::Pipeline::compile`] and
//!   [`crate::Pipeline::step`] for everything pipeline-level: plan
//!   validation, topo sort, missing-input, runtime kernel failures.

use model_plan::OperatorId;
use thiserror::Error;

/// Errors produced by [`crate::BoundedCompiler::compile`].
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompileError {
    /// The emitted shader source exceeded `max_shader_bytes` *or* the
    /// compile time exceeded `max_ms`. Both fields are populated when
    /// both limits were violated simultaneously.
    #[error(
        "compile budget exceeded: compile_ms={compile_ms} > max_ms={max_ms}, \
         shader_bytes={shader_bytes} > max_shader_bytes={max_shader_bytes}"
    )]
    BudgetExceeded {
        /// Wall-clock compile time observed.
        compile_ms: u64,
        /// Configured wall-clock ceiling.
        max_ms: u64,
        /// Bytes of generated shader source.
        shader_bytes: usize,
        /// Configured shader-byte ceiling.
        max_shader_bytes: usize,
    },

    /// The supplied plan failed validation. Wraps the underlying
    /// `model_plan::PlanError` message so callers don't need to depend on
    /// the model-plan error enum.
    #[error("invalid plan: {0}")]
    InvalidPlan(String),

    /// The supplied plan cannot be topologically ordered (cycle detected).
    #[error("plan contains a cycle in operator deps: {0}")]
    Cycle(String),
}

/// Errors produced by [`crate::Pipeline::compile`] and
/// [`crate::Pipeline::step`].
#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum PipelineError {
    /// `Pipeline::compile` rejected the plan during validation.
    #[error("invalid plan: {0}")]
    InvalidPlan(String),

    /// `Pipeline::compile` could not produce a topological order.
    #[error("topological sort failed: {0}")]
    TopoSortFailed(String),

    /// `Pipeline::compile` could not produce a compiled pipeline (shader
    /// generation or compile error from the bounded compiler).
    #[error("compile error: {0}")]
    CompileError(#[from] CompileError),

    /// `Pipeline::step` could not find an input tensor in the supplied map.
    #[error("missing input tensor '{name}' for operator {operator}")]
    MissingInput {
        /// Operator that needed the tensor.
        operator: OperatorId,
        /// Tensor name that was not supplied.
        name: String,
    },

    /// `Pipeline::step` observed a runtime shape mismatch (e.g. matmul
    /// inner dimension disagreement).
    #[error(
        "operator {operator}: shape mismatch on tensor '{name}' \
         (expected {expected} elements, got {actual})"
    )]
    ShapeMismatch {
        operator: OperatorId,
        name: String,
        expected: usize,
        actual: usize,
    },

    /// `Pipeline::step` could not dispatch an operator because the
    /// reference interpreter does not implement it.
    #[error("unsupported operator {operator}: {reason}")]
    UnsupportedOperator {
        operator: OperatorId,
        reason: String,
    },
}