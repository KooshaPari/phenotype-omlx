//! [`BoundedCompiler`] — the bounded shader-compiler entry point.
//!
//! See the module-level documentation in [`super`] for the full design.

use std::time::Instant;

use model_plan::ModelPlan;

use crate::cache::CompiledPipeline;
use crate::error::CompileError;
use crate::fingerprint::DeviceFingerprint;
use crate::RuntimeMode;

use super::budget::CompileBudget;
use super::msl_stub::{
    emit_msl_stub, plan_revision, synthesize_compile_work, synthetic_compile_time_ms,
};

/// Bounded shader compiler. Cheap to clone — all state is the budget.
#[derive(Debug, Clone)]
pub struct BoundedCompiler {
    budget: CompileBudget,
}

impl BoundedCompiler {
    /// Construct a compiler with the given budget.
    pub fn new(budget: CompileBudget) -> Self {
        Self { budget }
    }

    /// Return the active budget.
    pub fn budget(&self) -> CompileBudget {
        self.budget
    }

    /// Compile `plan` into a [`CompiledPipeline`] in reference mode.
    pub fn compile(
        &self,
        plan: &ModelPlan,
        fingerprint: &DeviceFingerprint,
    ) -> Result<CompiledPipeline, CompileError> {
        self.compile_with_mode(plan, fingerprint, RuntimeMode::Reference)
    }

    /// Compile `plan` under the configured runtime policy and budget.
    ///
    /// Errors:
    /// - [`CompileError::SourceCompilationForbidden`] in production mode.
    /// - [`CompileError::InvalidPlan`] if `plan` fails structural validation.
    /// - [`CompileError::BudgetExceeded`] if either budget is violated.
    pub fn compile_with_mode(
        &self,
        plan: &ModelPlan,
        fingerprint: &DeviceFingerprint,
        mode: RuntimeMode,
    ) -> Result<CompiledPipeline, CompileError> {
        if mode == RuntimeMode::Production {
            return Err(CompileError::SourceCompilationForbidden);
        }

        // 1. Validate the plan structurally.
        plan.validate()
            .map_err(|e| CompileError::InvalidPlan(e.to_string()))?;

        // 2. Emit the MSL shader source (stub string for now).
        let shader_source = emit_msl_stub(plan, fingerprint);
        let shader_bytes = shader_source.len();

        // 3. Simulate wall-clock compile. The real compiler will be a
        //    Metal device call; for now we measure the synthetic work
        //    via Instant so the budget check is meaningful.
        let start = Instant::now();
        let synthetic_work = synthesize_compile_work(plan, fingerprint);
        let _ = synthetic_work; // touched so the optimizer keeps it
        let elapsed = start.elapsed();
        let compile_ms = elapsed.as_millis() as u64 + synthetic_compile_time_ms(plan, fingerprint);

        // 4. Enforce the budget. We report BOTH dimensions in the error so
        //    the caller can debug which one tripped.
        if shader_bytes > self.budget.max_shader_bytes || compile_ms > self.budget.max_ms {
            return Err(CompileError::BudgetExceeded {
                compile_ms,
                max_ms: self.budget.max_ms,
                shader_bytes,
                max_shader_bytes: self.budget.max_shader_bytes,
            });
        }

        // 5. Compute the entry's source-revision hash (used as the cache
        //    dimension the plan uses to invalidate).
        let source_revision = plan_revision(plan);

        // 6. Build the result.
        let compiled_at_unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Ok(CompiledPipeline {
            plan_id: plan.id,
            source_revision,
            shader_source,
            compiled_at_unix_ms,
            // Software / no-Metal fallback always reports MSL version 0;
            // a real Metal compile would query the device here.
            ms_compute_version: 0,
            fingerprint_hash: fingerprint.fingerprint_hash(),
        })
    }
}