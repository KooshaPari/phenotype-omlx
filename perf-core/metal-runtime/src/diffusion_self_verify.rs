//! Bounded host contract for block-diffusion self-verification.
//!
//! This module only partitions work and checks returned blocks. It never
//! allocates a command buffer or launches a model workload.

use std::fmt;

use crate::{compare_f32, DiffusionParityError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiffusionVerificationBlock {
    pub start: usize,
    pub end: usize,
}

impl DiffusionVerificationBlock {
    pub const fn len(self) -> usize {
        self.end - self.start
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffusionVerificationPlan {
    pub tokens: usize,
    pub block_tokens: usize,
    pub max_blocks: usize,
    pub blocks: Vec<DiffusionVerificationBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffusionSelfVerifyError {
    ZeroTokens,
    ZeroBlockSize,
    ZeroBlockBudget,
    BlockBudgetExceeded { required: usize, max: usize },
    Parity(DiffusionParityError),
}

impl fmt::Display for DiffusionSelfVerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroTokens => f.write_str("self-verification requires at least one token"),
            Self::ZeroBlockSize => f.write_str("self-verification block size must be non-zero"),
            Self::ZeroBlockBudget => f.write_str("self-verification block budget must be non-zero"),
            Self::BlockBudgetExceeded { required, max } => {
                write!(
                    f,
                    "self-verification requires {required} blocks, budget is {max}"
                )
            }
            Self::Parity(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for DiffusionSelfVerifyError {}

impl From<DiffusionParityError> for DiffusionSelfVerifyError {
    fn from(error: DiffusionParityError) -> Self {
        Self::Parity(error)
    }
}

impl DiffusionVerificationPlan {
    pub fn for_tokens(
        tokens: usize,
        block_tokens: usize,
        max_blocks: usize,
    ) -> Result<Self, DiffusionSelfVerifyError> {
        if tokens == 0 {
            return Err(DiffusionSelfVerifyError::ZeroTokens);
        }
        if block_tokens == 0 {
            return Err(DiffusionSelfVerifyError::ZeroBlockSize);
        }
        if max_blocks == 0 {
            return Err(DiffusionSelfVerifyError::ZeroBlockBudget);
        }
        let required = tokens
            .checked_add(block_tokens - 1)
            .map(|value| value / block_tokens)
            .ok_or(DiffusionSelfVerifyError::BlockBudgetExceeded {
                required: usize::MAX,
                max: max_blocks,
            })?;
        if required > max_blocks {
            return Err(DiffusionSelfVerifyError::BlockBudgetExceeded {
                required,
                max: max_blocks,
            });
        }
        let blocks = (0..tokens)
            .step_by(block_tokens)
            .map(|start| DiffusionVerificationBlock {
                start,
                end: start.saturating_add(block_tokens).min(tokens),
            })
            .collect();
        Ok(Self {
            tokens,
            block_tokens,
            max_blocks,
            blocks,
        })
    }

    pub fn verify_f32_block(
        &self,
        block: DiffusionVerificationBlock,
        expected: &[f32],
        actual: &[f32],
        tolerance: f32,
    ) -> Result<(), DiffusionSelfVerifyError> {
        if !self.blocks.contains(&block) {
            return Err(DiffusionSelfVerifyError::Parity(
                DiffusionParityError::Value {
                    what: "verification block",
                    index: 0,
                    expected: "planned block".into(),
                    got: format!("{}..{}", block.start, block.end),
                },
            ));
        }
        if expected.len() != block.len() || actual.len() != block.len() {
            return Err(DiffusionSelfVerifyError::Parity(
                DiffusionParityError::Length {
                    what: "verification block",
                    expected: block.len(),
                    got: expected.len().max(actual.len()),
                },
            ));
        }
        compare_f32("diffusion block", expected, actual, tolerance).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitions_with_a_bounded_tail() {
        let plan = DiffusionVerificationPlan::for_tokens(10, 4, 3).unwrap();
        assert_eq!(
            plan.blocks,
            vec![
                DiffusionVerificationBlock { start: 0, end: 4 },
                DiffusionVerificationBlock { start: 4, end: 8 },
                DiffusionVerificationBlock { start: 8, end: 10 },
            ]
        );
    }

    #[test]
    fn rejects_unbounded_or_over_budget_plans() {
        assert_eq!(
            DiffusionVerificationPlan::for_tokens(10, 4, 2),
            Err(DiffusionSelfVerifyError::BlockBudgetExceeded {
                required: 3,
                max: 2
            })
        );
        assert_eq!(
            DiffusionVerificationPlan::for_tokens(0, 4, 1),
            Err(DiffusionSelfVerifyError::ZeroTokens)
        );
    }

    #[test]
    fn verifies_only_planned_blocks() {
        let plan = DiffusionVerificationPlan::for_tokens(4, 4, 1).unwrap();
        let block = plan.blocks[0];
        plan.verify_f32_block(block, &[0.1, 0.2], &[0.1, 0.2], 1e-5)
            .unwrap_err();
        plan.verify_f32_block(block, &[0.1, 0.2, 0.3, 0.4], &[0.1, 0.2, 0.3, 0.4], 1e-5)
            .unwrap();
    }
}
