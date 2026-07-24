//! Public surface re-exports for the model-plan domain crate.
//!
//! This crate owns the pure Rust domain model for the Metal model runtime:
//! [`ModelPlan`], [`OperatorPlan`], [`StatePlan`], scheduler policy, and a
//! slow reference interpreter used as the correctness oracle for fast
//! Metal/Mojo/Zig kernel candidates.
//!
//! Every public type is `#[serde(deny_unknown_fields)]` so external schema
//! drift surfaces immediately rather than silently dropping fields.

#![deny(unsafe_code)]

pub mod attention;
pub mod dtype;
pub mod error;
pub mod interpreter;
pub mod operator;
pub mod plan;
pub mod precision;
pub mod quantization;
pub mod scheduler;
pub mod state;
pub mod tensor;

pub use attention::AttentionKind;
pub use dtype::DType;
pub use error::{InterpreterError, PlanError};
pub use interpreter::{ReferenceInterpreter, StepOutputs};
pub use operator::{OperatorId, OperatorKind, OperatorPlan};
pub use plan::{ModelId, ModelPlan};
pub use precision::Precision;
pub use quantization::QuantizationPolicy;
pub use scheduler::SchedulerPolicy;
pub use state::{StateId, StateKind, StatePlan};
pub use tensor::TensorRef;
