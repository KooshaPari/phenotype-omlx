//! Deterministic dispatch plan for the masked-diffusion state stages.

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
}
