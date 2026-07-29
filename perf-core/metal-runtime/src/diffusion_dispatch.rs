//! Deterministic dispatch plan for the masked-diffusion state stages.

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
}
