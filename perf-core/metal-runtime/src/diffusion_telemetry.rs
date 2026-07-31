//! Host-side contract for reporting diffusion dispatch outcomes.
//!
//! This module intentionally has no Metal or model-runtime dependency.  A
//! caller can construct a report around any execution path (native, fallback,
//! or a failed dispatch) and validate it before emitting an envelope.

use thiserror::Error;

use crate::DiffusionStage;

/// Outcome of one diffusion stage.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionStageTelemetry {
    pub stage: DiffusionStage,
    pub elapsed_ms: f64,
    pub completed: bool,
    pub error: Option<String>,
    pub fallback: bool,
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
}

/// Validated report for the fixed three-stage diffusion dispatch plan.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionDispatchTelemetry {
    pub stages: [DiffusionStageTelemetry; 3],
    pub total_elapsed_ms: f64,
}

/// Report is the stable semantic name used by envelope producers.
pub type DiffusionDispatchReport = DiffusionDispatchTelemetry;

impl DiffusionDispatchTelemetry {
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
