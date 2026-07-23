//! Bounded shader compiler.
//!
//! [`BoundedCompiler`] turns a validated [`ModelPlan`] into a
//! [`CompiledPipeline`] containing an MSL shader template (a stub string
//! for now — real codegen arrives in a future task). The compiler
//! enforces two orthogonal budgets:
//!
//! - **shader-byte budget** (`max_shader_bytes`): the generated MSL source
//!   must fit. If it doesn't, [`CompileError::BudgetExceeded`] is returned.
//! - **wall-clock budget** (`max_ms`): the simulated compile path must
//!   finish within the budget. The current implementation uses a
//!   realistic-shaped time slice proportional to the operator count so
//!   tests can dial `max_ms = 0` and reliably trip the budget.
//!
//! Both budgets are checked on every compile call; a compile that violates
//! either or both produces a single [`CompileError::BudgetExceeded`] with
//! every field populated.
//!
//! # Layout
//!
//! - [`budget`]: [`CompileBudget`] — the shader-byte + wall-clock budgets.
//! - [`compiler`]: [`BoundedCompiler`] — the bounded compiler entry point.
//! - [`msl_stub`]: internal helpers (`emit_msl_stub`, `synthesize_*`,
//!   `plan_revision`) used by [`compiler`].
//!
//! `plan_revision` is `pub(crate)` so the [`crate::pipeline`] module can
//! compute the source-revision hash used as a cache dimension without
//! re-routing through the compiler.

mod budget;
mod compiler;
mod msl_stub;

#[cfg(test)]
mod tests;

pub use budget::CompileBudget;
pub use compiler::BoundedCompiler;
// `plan_revision` stays `pub(crate)` so `pipeline::compile_with_mode` can
// reach it via `crate::compile::plan_revision(plan)` without it leaking
// into the public surface.
pub(crate) use msl_stub::plan_revision;
