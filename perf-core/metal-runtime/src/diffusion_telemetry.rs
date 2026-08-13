//! Host-side contract for reporting diffusion dispatch outcomes.
//!
//! This module intentionally has no Metal or model-runtime dependency.  A
//! caller can construct a report around any execution path (native, fallback,
//! or a failed dispatch) and validate it before emitting an envelope.

use thiserror::Error;

use crate::{DiffusionDispatchPlan, DiffusionStage};

/// Outcome of one diffusion stage.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionStageTelemetry {
    pub stage: DiffusionStage,
    pub elapsed_ms: f64,
    pub completed: bool,
    pub error: Option<String>,
    pub fallback: bool,
}

/// A stage result paired with the validated telemetry needed for promotion.
///
/// `output` is absent when the native command failed; the failure remains in
/// `telemetry.error` so callers cannot accidentally treat fallback as success.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionStageOutcome<T> {
    pub output: Option<T>,
    pub telemetry: DiffusionStageTelemetry,
}

impl<T> DiffusionStageOutcome<T> {
    pub fn from_result<E: std::fmt::Display>(
        stage: DiffusionStage,
        elapsed_ms: f64,
        result: Result<T, E>,
        fallback: bool,
    ) -> Result<Self, DiffusionTelemetryError> {
        let telemetry = DiffusionStageTelemetry::from_result(
            stage,
            elapsed_ms,
            result
                .as_ref()
                .map(|_| ())
                .map_err(|error| error.to_string()),
            fallback,
        )?;
        Ok(Self {
            output: result.ok(),
            telemetry,
        })
    }
}

impl DiffusionStageTelemetry {
    /// Construct a completed stage outcome.
    pub fn completed(
        stage: DiffusionStage,
        elapsed_ms: f64,
    ) -> Result<Self, DiffusionTelemetryError> {
        Self::new(stage, elapsed_ms, true, None, false)
    }

    /// Construct a stage outcome, validating the fields at the boundary.
    pub fn new(
        stage: DiffusionStage,
        elapsed_ms: f64,
        completed: bool,
        error: Option<String>,
        fallback: bool,
    ) -> Result<Self, DiffusionTelemetryError> {
        if !elapsed_ms.is_finite() || elapsed_ms < 0.0 {
            return Err(DiffusionTelemetryError::InvalidElapsed { stage, elapsed_ms });
        }
        if completed && error.is_some() {
            return Err(DiffusionTelemetryError::CompletedWithError { stage });
        }
        if !completed && error.is_none() {
            return Err(DiffusionTelemetryError::IncompleteWithoutError { stage });
        }
        Ok(Self {
            stage,
            elapsed_ms,
            completed,
            error,
            fallback,
        })
    }

    /// Convert a command result into a validated stage envelope.
    ///
    /// The conversion is deliberately independent of Metal so command
    /// encoders and host-only tests share one failure/fallback policy.
    pub fn from_result<E: std::fmt::Display>(
        stage: DiffusionStage,
        elapsed_ms: f64,
        result: Result<(), E>,
        fallback: bool,
    ) -> Result<Self, DiffusionTelemetryError> {
        match result {
            Ok(()) => Self::new(stage, elapsed_ms, true, None, fallback),
            Err(error) => Self::new(stage, elapsed_ms, false, Some(error.to_string()), fallback),
        }
    }
}

/// Validated report for the fixed three-stage diffusion dispatch plan.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionDispatchTelemetry {
    pub stages: [DiffusionStageTelemetry; 3],
    pub total_elapsed_ms: f64,
}

/// Report is the stable semantic name used by envelope producers.
pub type DiffusionDispatchReport = DiffusionDispatchTelemetry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionDispatchDecision {
    Promote,
    Fallback,
    Rollback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiffusionRollbackPolicy {
    pub max_failed_stages: usize,
    pub allow_fallback: bool,
}

impl DiffusionRollbackPolicy {
    pub const fn bounded(max_failed_stages: usize, allow_fallback: bool) -> Option<Self> {
        if max_failed_stages == 0 {
            None
        } else {
            Some(Self {
                max_failed_stages,
                allow_fallback,
            })
        }
    }

    pub fn decide(&self, report: &DiffusionDispatchTelemetry) -> DiffusionDispatchDecision {
        let failed = report
            .stages
            .iter()
            .filter(|stage| !stage.completed)
            .count();
        if failed == 0 && !report.used_fallback() {
            DiffusionDispatchDecision::Promote
        } else if self.allow_fallback && failed <= self.max_failed_stages {
            DiffusionDispatchDecision::Fallback
        } else {
            DiffusionDispatchDecision::Rollback
        }
    }
}

impl DiffusionDispatchTelemetry {
    /// Build telemetry for a dispatch plan without touching Metal.
    ///
    /// The plan is treated as the source of truth for stage order and state
    /// layout.  This keeps host-side reports honest: a caller cannot attach
    /// timings to a hand-built or stale plan and accidentally emit a valid
    /// looking envelope.
    pub fn for_plan(
        plan: &DiffusionDispatchPlan,
        stages: [DiffusionStageTelemetry; 3],
    ) -> Result<Self, DiffusionTelemetryError> {
        let expected = DiffusionDispatchPlan::for_tokens(plan.tokens).map_err(|_| {
            DiffusionTelemetryError::InvalidPlan {
                tokens: plan.tokens,
            }
        })?;
        if plan.layout != expected.layout || plan.stages != expected.stages {
            return Err(DiffusionTelemetryError::InvalidPlan {
                tokens: plan.tokens,
            });
        }
        Self::new(stages)
    }

    /// Build a report and derive its total duration from the stage durations.
    pub fn new(stages: [DiffusionStageTelemetry; 3]) -> Result<Self, DiffusionTelemetryError> {
        let expected = [
            DiffusionStage::ActiveCompact,
            DiffusionStage::Remask,
            DiffusionStage::Trajectory,
        ];
        for (index, (actual, expected)) in stages.iter().zip(expected).enumerate() {
            if actual.stage != expected {
                return Err(DiffusionTelemetryError::StageOrder {
                    index,
                    expected,
                    got: actual.stage,
                });
            }
        }
        let total_elapsed_ms: f64 = stages.iter().map(|stage| stage.elapsed_ms).sum();
        if !total_elapsed_ms.is_finite() {
            return Err(DiffusionTelemetryError::InvalidTotal { total_elapsed_ms });
        }
        Ok(Self {
            stages,
            total_elapsed_ms,
        })
    }

    pub fn all_completed(&self) -> bool {
        self.stages.iter().all(|stage| stage.completed)
    }

    pub fn used_fallback(&self) -> bool {
        self.stages.iter().any(|stage| stage.fallback)
    }
}

/// Rejection reasons for malformed telemetry envelopes.
#[derive(Debug, Error, PartialEq)]
pub enum DiffusionTelemetryError {
    #[error("{stage:?} elapsed_ms must be finite and non-negative, got {elapsed_ms}")]
    InvalidElapsed {
        stage: DiffusionStage,
        elapsed_ms: f64,
    },
    #[error("{stage:?} is completed but carries an error")]
    CompletedWithError { stage: DiffusionStage },
    #[error("{stage:?} is incomplete but carries no error")]
    IncompleteWithoutError { stage: DiffusionStage },
    #[error("stage {index} expected {expected:?}, got {got:?}")]
    StageOrder {
        index: usize,
        expected: DiffusionStage,
        got: DiffusionStage,
    },
    #[error("total elapsed_ms must be finite, got {total_elapsed_ms}")]
    InvalidTotal { total_elapsed_ms: f64 },
    #[error("dispatch plan is invalid for {tokens} tokens")]
    InvalidPlan { tokens: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DiffusionStateLayout;

    fn stage(stage: DiffusionStage, elapsed_ms: f64) -> DiffusionStageTelemetry {
        DiffusionStageTelemetry::completed(stage, elapsed_ms).unwrap()
    }

    #[test]
    fn report_derives_total_and_completion_flags() {
        let report = DiffusionDispatchTelemetry::new([
            stage(DiffusionStage::ActiveCompact, 0.5),
            stage(DiffusionStage::Remask, 1.25),
            stage(DiffusionStage::Trajectory, 0.25),
        ])
        .unwrap();
        assert_eq!(report.total_elapsed_ms, 2.0);
        assert!(report.all_completed());
        assert!(!report.used_fallback());
    }

    #[test]
    fn rejects_invalid_timing_state_and_order() {
        assert!(matches!(
            DiffusionStageTelemetry::completed(DiffusionStage::Remask, f64::NAN),
            Err(DiffusionTelemetryError::InvalidElapsed { .. })
        ));
        assert!(matches!(
            DiffusionStageTelemetry::new(DiffusionStage::Remask, 1.0, false, None, false,),
            Err(DiffusionTelemetryError::IncompleteWithoutError { .. })
        ));
        let wrong = [
            stage(DiffusionStage::Remask, 0.1),
            stage(DiffusionStage::ActiveCompact, 0.1),
            stage(DiffusionStage::Trajectory, 0.1),
        ];
        assert!(matches!(
            DiffusionDispatchTelemetry::new(wrong),
            Err(DiffusionTelemetryError::StageOrder { .. })
        ));
    }

    #[test]
    fn failed_fallback_is_reported_without_claiming_completion() {
        let failed = DiffusionStageTelemetry::new(
            DiffusionStage::Trajectory,
            3.0,
            false,
            Some("native dispatch unavailable".into()),
            true,
        )
        .unwrap();
        assert!(!failed.completed);
        assert!(failed.fallback);
        assert_eq!(failed.error.as_deref(), Some("native dispatch unavailable"));
    }

    #[test]
    fn result_conversion_preserves_failure_and_fallback() {
        let failed = DiffusionStageTelemetry::from_result(
            DiffusionStage::Remask,
            2.5,
            Err("command buffer failed"),
            true,
        )
        .unwrap();
        assert!(!failed.completed);
        assert!(failed.fallback);
        assert_eq!(failed.error.as_deref(), Some("command buffer failed"));

        let completed = DiffusionStageTelemetry::from_result(
            DiffusionStage::Remask,
            1.0,
            Ok::<(), &str>(()),
            false,
        )
        .unwrap();
        assert!(completed.completed);
        assert!(completed.error.is_none());

        let outcome = DiffusionStageOutcome::from_result(
            DiffusionStage::ActiveCompact,
            0.5,
            Err::<u32, _>("native failure"),
            false,
        )
        .unwrap();
        assert!(outcome.output.is_none());
        assert_eq!(outcome.telemetry.error.as_deref(), Some("native failure"));
    }

    #[test]
    fn rollback_policy_is_bounded_and_explicit() {
        assert!(DiffusionRollbackPolicy::bounded(0, true).is_none());
        let policy = DiffusionRollbackPolicy::bounded(1, true).unwrap();
        let report = DiffusionDispatchTelemetry::new([
            stage(DiffusionStage::ActiveCompact, 0.1),
            DiffusionStageTelemetry::from_result(
                DiffusionStage::Remask,
                0.2,
                Err::<(), _>("native unavailable"),
                true,
            )
            .unwrap(),
            stage(DiffusionStage::Trajectory, 0.1),
        ])
        .unwrap();
        assert_eq!(policy.decide(&report), DiffusionDispatchDecision::Fallback);
        let strict = DiffusionRollbackPolicy::bounded(1, false).unwrap();
        assert_eq!(strict.decide(&report), DiffusionDispatchDecision::Rollback);
    }

    #[test]
    fn plan_report_binds_layout_without_execution() {
        let plan = DiffusionDispatchPlan::for_tokens(8).unwrap();
        let report = DiffusionDispatchTelemetry::for_plan(
            &plan,
            [
                stage(DiffusionStage::ActiveCompact, 0.25),
                stage(DiffusionStage::Remask, 0.5),
                stage(DiffusionStage::Trajectory, 0.75),
            ],
        )
        .unwrap();
        assert_eq!(report.total_elapsed_ms, 1.5);
    }

    #[test]
    fn plan_report_rejects_stale_layout_or_stage_plan() {
        let mut plan = DiffusionDispatchPlan::for_tokens(8).unwrap();
        plan.layout = DiffusionStateLayout::for_tokens(16).unwrap();
        let result = DiffusionDispatchTelemetry::for_plan(
            &plan,
            [
                stage(DiffusionStage::ActiveCompact, 0.1),
                stage(DiffusionStage::Remask, 0.1),
                stage(DiffusionStage::Trajectory, 0.1),
            ],
        );
        assert!(matches!(
            result,
            Err(DiffusionTelemetryError::InvalidPlan { tokens: 8 })
        ));
    }
}
