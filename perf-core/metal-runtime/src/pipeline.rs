//! Executable pipeline.
//!
//! [`Pipeline`] is the user-facing runtime handle: it owns the compiled
//! shader template, a copy of the [`ModelPlan`] it was compiled from, and
//! the device fingerprint. It exposes [`Pipeline::step`] for a single
//! forward pass and reuses the [`model_plan::ReferenceInterpreter`]'s
//! slow but correct scalar oracle so the pipeline produces
//! mathematically expected outputs regardless of whether the Metal device
//! is present. A future task will swap the oracle for the real MSL
//! dispatch once the `metal` feature is wired in.

use std::collections::HashMap;
use std::time::Instant;

use model_plan::{ModelId, ModelPlan, OperatorId, ReferenceInterpreter};
use tracing::{info, instrument};

use crate::cache::{CompiledPipeline, PipelineCache};
use crate::compile::BoundedCompiler;
use crate::error::PipelineError;
use crate::fingerprint::DeviceFingerprint;

/// One step's outputs.
#[derive(Debug, Clone, PartialEq)]
pub struct StepOutput {
    /// Map of tensor name to computed f32 buffer. Mirrors the
    /// [`model_plan::StepOutputs::outputs`] shape so callers can treat
    /// the two interchangeably.
    pub values: HashMap<String, Vec<f32>>,
    /// Operator execution order recorded by the pipeline. Useful for
    /// chained plans and for asserting on topo-sort correctness.
    pub execution_order: Vec<OperatorId>,
}

/// One compiled, executable pipeline. Holds a clone of the plan so
/// `step` can run the reference interpreter without an external plan
/// reference.
#[derive(Debug)]
pub struct Pipeline {
    plan: ModelPlan,
    max_seq_len: usize,
    /// Compiled MSL shader template (stub string in this revision).
    shader_template: String,
    /// Unix epoch millis at which the underlying `CompiledPipeline` was
    /// produced.
    compiled_at: u64,
    /// The device fingerprint used to compile this pipeline.
    fingerprint: DeviceFingerprint,
}

impl Pipeline {
    /// Compile `plan` into a [`Pipeline`], consulting `cache` for a
    /// previously-compiled entry under `(plan_id, source_revision,
    /// fingerprint_hash)`. On a hit the cached entry is reused; on a
    /// miss the plan is passed through `compiler` and the result is
    /// inserted into the cache.
    #[instrument(skip(plan, fp, compiler, cache), fields(plan_id = plan.id.0))]
    pub fn compile(
        plan: &ModelPlan,
        fp: &DeviceFingerprint,
        compiler: &BoundedCompiler,
        cache: &mut PipelineCache,
    ) -> Result<Self, PipelineError> {
        // 1. Validate.
        plan.validate().map_err(|e| PipelineError::InvalidPlan(e.to_string()))?;

        // 2. Topo-sort to surface cycle errors before we bother compiling.
        let _topo = plan
            .topo_sort()
            .map_err(|e| PipelineError::TopoSortFailed(e.to_string()))?;

        let source_revision = crate::compile::plan_revision(plan);
        let key_fp = fp.fingerprint_hash();

        // 3. Cache lookup.
        if let Some(cached) = cache.get(plan.id, source_revision, key_fp) {
            info!(source_revision, key_fp, "pipeline cache hit");
            return Ok(Self::from_compiled(plan.clone(), fp, cached));
        }

        // 4. Compile.
        let compiled = compiler.compile(plan, fp)?;
        let result = Self::from_compiled(plan.clone(), fp, compiled.clone());
        cache.insert(plan.id, source_revision, key_fp, compiled);
        info!(source_revision, key_fp, "pipeline compiled and cached");
        Ok(result)
    }

    fn from_compiled(
        plan: ModelPlan,
        fp: &DeviceFingerprint,
        compiled: CompiledPipeline,
    ) -> Self {
        let max_seq_len = plan.max_seq_len;
        let shader_template = compiled.shader_source;
        let compiled_at = compiled.compiled_at_unix_ms;
        Self {
            plan,
            max_seq_len,
            shader_template,
            compiled_at,
            fingerprint: fp.clone(),
        }
    }

    /// Maximum sequence length the plan supports.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// The compiled MSL shader template (stub string in this revision).
    pub fn shader_template(&self) -> &str {
        &self.shader_template
    }

    /// The device fingerprint under which this pipeline was compiled.
    pub fn fingerprint(&self) -> &DeviceFingerprint {
        &self.fingerprint
    }

    /// Plan id.
    pub fn plan_id(&self) -> ModelId {
        self.plan.id
    }

    /// Borrow the underlying plan.
    pub fn plan(&self) -> &ModelPlan {
        &self.plan
    }

    /// Unix epoch millis at which the underlying `CompiledPipeline` was
    /// produced.
    pub fn compiled_at_unix_ms(&self) -> u64 {
        self.compiled_at
    }

    /// Execute a single forward pass. Inputs are looked up by tensor
    /// name; missing inputs return [`PipelineError::MissingInput`].
    ///
    /// Internally the pipeline drives the [`model_plan::ReferenceInterpreter`]
    /// so the output is mathematically correct against the plan. A future
    /// task will route through a real Metal dispatch when the `metal`
    /// feature is enabled.
    #[instrument(skip(self, inputs), fields(plan_id = self.plan.id.0))]
    pub fn step(
        &self,
        inputs: &HashMap<String, Vec<f32>>,
    ) -> Result<StepOutput, PipelineError> {
        let interpreter = ReferenceInterpreter::new(self.plan.clone());
        let started = Instant::now();
        let mut per_op_started = Instant::now();
        let out = interpreter.run(inputs).map_err(map_interpreter_error)?;
        for op_id in &out.execution_order {
            info!(
                operator = %op_id,
                elapsed_us = per_op_started.elapsed().as_micros() as u64,
                "pipeline op dispatched"
            );
            per_op_started = Instant::now();
        }
        info!(
            elapsed_us = started.elapsed().as_micros() as u64,
            op_count = out.execution_order.len(),
            "pipeline step complete"
        );
        Ok(StepOutput {
            values: out.outputs,
            execution_order: out.execution_order,
        })
    }
}

fn map_interpreter_error(e: model_plan::InterpreterError) -> PipelineError {
    use model_plan::InterpreterError as IE;
    match e {
        IE::MissingInput { operator, name } => PipelineError::MissingInput { operator, name },
        IE::ShapeMismatch {
            operator,
            name,
            expected,
            actual,
            ..
        } => PipelineError::ShapeMismatch {
            operator,
            name,
            expected,
            actual,
        },
        IE::UnsupportedOperator { operator, kind } => PipelineError::UnsupportedOperator {
            operator,
            reason: format!("{:?}", kind),
        },
        IE::ArityMismatch {
            operator,
            expected,
            actual,
        } => PipelineError::ShapeMismatch {
            operator,
            name: format!("arity_mismatch_{}_vs_{}", expected, actual),
            expected,
            actual,
        },
        IE::Cycle => PipelineError::TopoSortFailed("cycle in operator deps".to_string()),
        other => PipelineError::UnsupportedOperator {
            operator: OperatorId(0),
            reason: format!("{:?}", other),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CompileBudget;
    use model_plan::{DType, OperatorId, OperatorKind, OperatorPlan, Precision,
        QuantizationPolicy, SchedulerPolicy, TensorRef};

    fn make_plan() -> ModelPlan {
        let plan = ModelPlan::new_unchecked(
            ModelId(1),
            "tiny",
            "test",
            vec![OperatorPlan {
                id: OperatorId(1),
                kind: OperatorKind::Copy,
                attention: None,
                inputs: vec![TensorRef {
                    name: "x".into(),
                    shape: vec![1],
                    dtype: DType::F32,
                    state_id: None,
                }],
                outputs: vec![TensorRef {
                    name: "y".into(),
                    shape: vec![1],
                    dtype: DType::F32,
                    state_id: None,
                }],
                precision: Precision::Fp32,
                quant: QuantizationPolicy::Dense,
                deps: vec![],
            }],
            vec![],
            SchedulerPolicy::Eager,
            4,
            8,
        );
        plan.validate().expect("tiny plan must validate");
        plan
    }

    #[test]
    fn step_runs_reference_interpreter() {
        let plan = make_plan();
        let fp = DeviceFingerprint::compute_software();
        let compiler = BoundedCompiler::new(CompileBudget::DEFAULT);
        let mut cache = PipelineCache::new(crate::EvictionPolicy::Lru, 4);
        let pipeline = Pipeline::compile(&plan, &fp, &compiler, &mut cache).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), vec![42.0]);
        let out = pipeline.step(&inputs).unwrap();
        assert_eq!(out.values.get("y"), Some(&vec![42.0]));
        assert_eq!(out.execution_order, vec![OperatorId(1)]);
    }
}

// Compile-time assertion: the cache + pipeline can be shared across threads.
#[allow(dead_code)]
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn _f() {
        assert_send_sync::<Pipeline>();
        assert_send_sync::<PipelineCache>();
    }
};