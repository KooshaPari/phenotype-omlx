//! Deterministic dispatch plan for the masked-diffusion state stages.

use crate::DiffusionStageOutcome;
use crate::{
    DiffusionDispatchDecision, DiffusionDispatchTelemetry, DiffusionRollbackPolicy,
    DiffusionTelemetryError,
};
use crate::{DiffusionStateLayout, DiffusionStateLayoutError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionStage {
    ActiveCompact,
    Remask,
    Trajectory,
}

impl DiffusionStage {
    pub const fn tag(self) -> &'static str {
        match self {
            Self::ActiveCompact => "active_compact",
            Self::Remask => "remask",
            Self::Trajectory => "trajectory",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffusionDispatchPlan {
    pub tokens: usize,
    pub layout: DiffusionStateLayout,
    pub stages: [DiffusionStage; 3],
}

/// Host-only result of evaluating all three native stage outcomes.
///
/// The outcomes are retained so callers can still consume successful stage
/// outputs after the promotion decision.  This helper deliberately performs
/// no Metal work: it only binds telemetry to this plan and applies the bounded
/// rollback policy.
#[derive(Debug, PartialEq)]
pub struct DiffusionDispatchEvaluation<A, B, C> {
    pub outcomes: (
        DiffusionStageOutcome<A>,
        DiffusionStageOutcome<B>,
        DiffusionStageOutcome<C>,
    ),
    pub report: DiffusionDispatchTelemetry,
    pub decision: DiffusionDispatchDecision,
}

impl DiffusionDispatchPlan {
    pub fn for_tokens(tokens: usize) -> Result<Self, DiffusionStateLayoutError> {
        Ok(Self {
            tokens,
            layout: DiffusionStateLayout::for_tokens(tokens)?,
            stages: [
                DiffusionStage::ActiveCompact,
                DiffusionStage::Remask,
                DiffusionStage::Trajectory,
            ],
        })
    }

    pub const fn thread_grid(&self) -> usize {
        self.tokens
    }

    pub const fn stage_tags(&self) -> [&'static str; 3] {
        [
            self.stages[0].tag(),
            self.stages[1].tag(),
            self.stages[2].tag(),
        ]
    }

    /// Evaluate a complete report against this plan and a bounded policy.
    ///
    /// Re-validating the report here prevents callers from making promotion
    /// decisions from telemetry belonging to a different token layout or
    /// stage order.
    pub fn evaluate(
        &self,
        report: &DiffusionDispatchTelemetry,
        policy: &DiffusionRollbackPolicy,
    ) -> Result<DiffusionDispatchDecision, DiffusionTelemetryError> {
        let validated = DiffusionDispatchTelemetry::for_plan(self, report.stages.clone())?;
        Ok(policy.decide(&validated))
    }

    /// Consume active-compaction, remask, and trajectory outcomes together.
    ///
    /// Keeping this orchestration at the plan boundary prevents a caller from
    /// accidentally evaluating a partial or differently ordered stage tuple.
    pub fn evaluate_outcomes<A, B, C>(
        &self,
        outcomes: (
            DiffusionStageOutcome<A>,
            DiffusionStageOutcome<B>,
            DiffusionStageOutcome<C>,
        ),
        policy: &DiffusionRollbackPolicy,
    ) -> Result<DiffusionDispatchEvaluation<A, B, C>, DiffusionTelemetryError> {
        let report = DiffusionDispatchTelemetry::for_plan(
            self,
            [
                outcomes.0.telemetry.clone(),
                outcomes.1.telemetry.clone(),
                outcomes.2.telemetry.clone(),
            ],
        )?;
        let decision = self.evaluate(&report, policy)?;
        Ok(DiffusionDispatchEvaluation {
            outcomes,
            report,
            decision,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_binds_layout_and_stage_order() {
        let plan = DiffusionDispatchPlan::for_tokens(16).unwrap();
        assert_eq!(plan.thread_grid(), 16);
        assert_eq!(
            plan.stage_tags(),
            ["active_compact", "remask", "trajectory"]
        );
        assert_eq!(plan.layout.total_bytes, 224);
    }

    #[test]
    fn plan_rejects_zero_tokens() {
        assert_eq!(
            DiffusionDispatchPlan::for_tokens(0),
            Err(DiffusionStateLayoutError::ZeroTokens)
        );
    }

    #[test]
    fn plan_evaluates_only_plan_bound_reports() {
        let plan = DiffusionDispatchPlan::for_tokens(8).unwrap();
        let report = DiffusionDispatchTelemetry::for_plan(
            &plan,
            [
                crate::DiffusionStageTelemetry::completed(DiffusionStage::ActiveCompact, 0.1)
                    .unwrap(),
                crate::DiffusionStageTelemetry::completed(DiffusionStage::Remask, 0.1).unwrap(),
                crate::DiffusionStageTelemetry::completed(DiffusionStage::Trajectory, 0.1).unwrap(),
            ],
        )
        .unwrap();
        let policy = DiffusionRollbackPolicy::bounded(1, false).unwrap();
        assert_eq!(
            plan.evaluate(&report, &policy).unwrap(),
            DiffusionDispatchDecision::Promote
        );
    }

    #[test]
    fn evaluates_three_outcomes_and_retains_outputs() {
        let plan = DiffusionDispatchPlan::for_tokens(4).unwrap();
        let outcomes = (
            DiffusionStageOutcome::from_result(
                DiffusionStage::ActiveCompact,
                0.2,
                Ok::<_, &str>(vec![1_u32, 2]),
                false,
            )
            .unwrap(),
            DiffusionStageOutcome::from_result(
                DiffusionStage::Remask,
                0.3,
                Ok::<_, &str>(vec![1_u8, 0]),
                false,
            )
            .unwrap(),
            DiffusionStageOutcome::from_result(
                DiffusionStage::Trajectory,
                0.4,
                Ok::<_, &str>(vec![0.25_f32, 0.5]),
                false,
            )
            .unwrap(),
        );
        let policy = DiffusionRollbackPolicy::bounded(1, false).unwrap();
        let evaluation = plan.evaluate_outcomes(outcomes, &policy).unwrap();
        assert_eq!(evaluation.decision, DiffusionDispatchDecision::Promote);
        assert_eq!(evaluation.report.total_elapsed_ms, 0.9);
        assert_eq!(
            evaluation.outcomes.0.output.as_deref(),
            Some(&[1_u32, 2][..])
        );
        assert_eq!(
            evaluation.outcomes.1.output.as_deref(),
            Some(&[1_u8, 0][..])
        );
    }

    #[test]
    fn evaluates_failure_as_fallback_or_rollback_by_policy() {
        let plan = DiffusionDispatchPlan::for_tokens(4).unwrap();
        let outcomes = (
            DiffusionStageOutcome::from_result(
                DiffusionStage::ActiveCompact,
                0.2,
                Ok::<_, &str>(vec![1_u32]),
                false,
            )
            .unwrap(),
            DiffusionStageOutcome::from_result(
                DiffusionStage::Remask,
                0.3,
                Err::<Vec<u8>, _>("remask unavailable"),
                true,
            )
            .unwrap(),
            DiffusionStageOutcome::from_result(
                DiffusionStage::Trajectory,
                0.4,
                Ok::<_, &str>(vec![0.25_f32]),
                false,
            )
            .unwrap(),
        );
        let fallback = DiffusionRollbackPolicy::bounded(1, true).unwrap();
        assert_eq!(
            plan.evaluate_outcomes(outcomes, &fallback)
                .unwrap()
                .decision,
            DiffusionDispatchDecision::Fallback
        );

        let rollback = DiffusionRollbackPolicy::bounded(1, false).unwrap();
        let outcomes = (
            DiffusionStageOutcome::from_result(
                DiffusionStage::ActiveCompact,
                0.2,
                Ok::<_, &str>(vec![1_u32]),
                false,
            )
            .unwrap(),
            DiffusionStageOutcome::from_result(
                DiffusionStage::Remask,
                0.3,
                Err::<Vec<u8>, _>("remask unavailable"),
                false,
            )
            .unwrap(),
            DiffusionStageOutcome::from_result(
                DiffusionStage::Trajectory,
                0.4,
                Ok::<_, &str>(vec![0.25_f32]),
                false,
            )
            .unwrap(),
        );
        assert_eq!(
            plan.evaluate_outcomes(outcomes, &rollback)
                .unwrap()
                .decision,
            DiffusionDispatchDecision::Rollback
        );
    }
}
