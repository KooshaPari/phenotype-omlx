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

use std::time::Instant;

use model_plan::ModelPlan;
use sha2::{Digest, Sha256};

use crate::cache::CompiledPipeline;
use crate::error::CompileError;
use crate::fingerprint::DeviceFingerprint;

/// Wall-clock and shader-byte budgets for [`BoundedCompiler::compile`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompileBudget {
    /// Maximum wall-clock compile time in milliseconds. `0` makes the
    /// budget impossible to satisfy for any non-trivial plan (used by
    /// tests to drive the over-budget error path).
    pub max_ms: u64,
    /// Maximum total shader-source bytes the compiler may emit.
    pub max_shader_bytes: usize,
}

impl CompileBudget {
    /// Generous default used when the caller has no specific budget in
    /// mind. Mirrors the policy used by the reference interpreter's
    /// compile-time smoke tests.
    pub const DEFAULT: Self = Self {
        max_ms: 5_000,
        max_shader_bytes: 1 << 20, // 1 MiB
    };
}

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

    /// Compile `plan` into a [`CompiledPipeline`] under the configured
    /// budget.
    ///
    /// Errors:
    /// - [`CompileError::InvalidPlan`] if `plan` fails structural validation.
    /// - [`CompileError::BudgetExceeded`] if either budget is violated.
    pub fn compile(
        &self,
        plan: &ModelPlan,
        fingerprint: &DeviceFingerprint,
    ) -> Result<CompiledPipeline, CompileError> {
        // 1. Validate the plan structurally.
        plan.validate().map_err(|e| CompileError::InvalidPlan(e.to_string()))?;

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
        let compile_ms = elapsed.as_millis() as u64
            + synthetic_compile_time_ms(plan, fingerprint);

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

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Emit a deterministic MSL stub string for `plan`. The stub is currently
/// informational — it includes the plan id, family, and a one-line kernel
/// per operator. A future task will replace this with real codegen driven
/// by the operator kinds.
fn emit_msl_stub(plan: &ModelPlan, fp: &DeviceFingerprint) -> String {
    let mut out = String::with_capacity(256 + plan.operators.len() * 80);
    out.push_str("// metal-runtime MSL stub (real codegen in a future task)\n");
    out.push_str(&format!("// plan_id      = {}\n", plan.id.0));
    out.push_str(&format!("// family       = {}\n", plan.model_family));
    out.push_str(&format!("// gpu_family   = {}\n", fp.gpu_family.tag()));
    out.push_str(&format!("// op_count     = {}\n", plan.operators.len()));
    out.push_str(&format!("// max_seq_len  = {}\n", plan.max_seq_len));
    out.push_str("#include <metal_stdlib>\n");
    out.push_str("using namespace metal;\n\n");
    for op in &plan.operators {
        out.push_str(&format!(
            "// op#{} {} : {} input(s), {} output(s)\n",
            op.id.0,
            op.kind.tag(),
            op.inputs.len(),
            op.outputs.len(),
        ));
    }
    out
}

/// Simulate the compile-side work — sha256 over the plan + fingerprint so
/// different plans produce different outputs and the optimizer cannot fold
/// it away.
fn synthesize_compile_work(plan: &ModelPlan, fp: &DeviceFingerprint) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(plan.id.0.to_le_bytes());
    h.update(plan.name.as_bytes());
    h.update(plan.model_family.as_bytes());
    h.update(plan.max_seq_len.to_le_bytes());
    h.update(plan.vocab_size.to_le_bytes());
    h.update(fp.device_name.as_bytes());
    h.update(fp.os.as_bytes());
    h.update(fp.arch.as_bytes());
    h.update(fp.simd_bit_width.to_le_bytes());
    h.update(fp.total_memory_bytes.to_le_bytes());
    h.update((fp.gpu_family as u8).to_le_bytes());
    for op in &plan.operators {
        h.update(op.id.0.to_le_bytes());
        h.update(op.kind.tag().as_bytes());
    }
    h.finalize().into()
}

/// Estimate compile wall-clock time in milliseconds. Scales with operator
/// count so a tiny plan is fast and a large plan is slow. Combined with
/// `Instant` elapsed, this gives a deterministic-ish budget signal.
fn synthetic_compile_time_ms(plan: &ModelPlan, fp: &DeviceFingerprint) -> u64 {
    // 1 ms per operator + 5 ms base + an extra 1 ms per byte of
    // simd_bit_width * 8 (purely deterministic and stable).
    let base = 5u64;
    let per_op = plan.operators.len() as u64;
    let fp_overhead = (fp.simd_bit_width as u64) / 8;
    base + per_op + fp_overhead
}

/// Derive a `source_revision` u64 from the plan contents. The current
/// policy is a content-derived hash of the operator set so any change
/// (added op, new dep, swapped kind) bumps the revision and invalidates
/// cached entries. Plan-level metadata (id, name, family) is deliberately
/// excluded so renaming a plan does not invalidate the cache.
pub(crate) fn plan_revision(plan: &ModelPlan) -> u64 {
    let mut h = Sha256::new();
    h.update((plan.operators.len() as u64).to_le_bytes());
    h.update(plan.max_seq_len.to_le_bytes());
    h.update(plan.vocab_size.to_le_bytes());
    for op in &plan.operators {
        h.update(op.id.0.to_le_bytes());
        h.update(op.kind.tag().as_bytes());
        h.update((op.inputs.len() as u64).to_le_bytes());
        h.update((op.outputs.len() as u64).to_le_bytes());
        h.update((op.deps.len() as u64).to_le_bytes());
        for d in &op.deps {
            h.update(d.0.to_le_bytes());
        }
    }
    let bytes = h.finalize();
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    u64::from_le_bytes(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_plan() -> ModelPlan {
        use model_plan::{
            DType, ModelId, OperatorId, OperatorKind, OperatorPlan, Precision,
            QuantizationPolicy, SchedulerPolicy, TensorRef,
        };
        let op = |id: u64, kind: OperatorKind, ins: usize, outs: usize| OperatorPlan {
            id: OperatorId(id),
            kind,
            attention: None,
            inputs: (0..ins)
                .map(|i| TensorRef {
                    name: format!("i{}", i),
                    shape: vec![1],
                    dtype: DType::F32,
                    state_id: None,
                })
                .collect(),
            outputs: (0..outs)
                .map(|i| TensorRef {
                    name: format!("o{}", i),
                    shape: vec![1],
                    dtype: DType::F32,
                    state_id: None,
                })
                .collect(),
            precision: Precision::Fp32,
            quant: QuantizationPolicy::Dense,
            deps: vec![],
        };
        let plan = ModelPlan::new_unchecked(
            ModelId(1),
            "tiny",
            "test",
            vec![op(1, OperatorKind::Copy, 1, 1)],
            vec![],
            SchedulerPolicy::Eager,
            4,
            8,
        );
        plan.validate().expect("tiny plan must validate");
        plan
    }

    #[test]
    fn budget_default_is_generous() {
        let b = CompileBudget::DEFAULT;
        assert!(b.max_ms >= 100);
        assert!(b.max_shader_bytes >= 4096);
    }

    #[test]
    fn plan_revision_changes_when_op_added() {
        let plan = small_plan();
        let r0 = plan_revision(&plan);
        use model_plan::{
            DType, ModelId, OperatorId, OperatorKind, OperatorPlan, Precision,
            QuantizationPolicy, SchedulerPolicy, TensorRef,
        };
        let bigger = ModelPlan::new_unchecked(
            ModelId(1),
            "tiny",
            "test",
            vec![
                OperatorPlan {
                    id: OperatorId(1),
                    kind: OperatorKind::Copy,
                    attention: None,
                    inputs: vec![TensorRef {
                        name: "i0".into(),
                        shape: vec![1],
                        dtype: DType::F32,
                        state_id: None,
                    }],
                    outputs: vec![TensorRef {
                        name: "o0".into(),
                        shape: vec![1],
                        dtype: DType::F32,
                        state_id: None,
                    }],
                    precision: Precision::Fp32,
                    quant: QuantizationPolicy::Dense,
                    deps: vec![],
                },
                OperatorPlan {
                    id: OperatorId(2),
                    kind: OperatorKind::Copy,
                    attention: None,
                    inputs: vec![TensorRef {
                        name: "i1".into(),
                        shape: vec![1],
                        dtype: DType::F32,
                        state_id: None,
                    }],
                    outputs: vec![TensorRef {
                        name: "o1".into(),
                        shape: vec![1],
                        dtype: DType::F32,
                        state_id: None,
                    }],
                    precision: Precision::Fp32,
                    quant: QuantizationPolicy::Dense,
                    deps: vec![],
                },
            ],
            vec![],
            SchedulerPolicy::Eager,
            4,
            8,
        );
        let r1 = plan_revision(&bigger);
        assert_ne!(r0, r1);
    }
}