//! Runtime allocation contract for masked-diffusion state.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiffusionStateLayout {
    pub tokens: usize,
    pub mask_bytes: usize,
    pub confidence_bytes: usize,
    pub entropy_bytes: usize,
    pub momentum_bytes: usize,
    pub converged_bytes: usize,
    pub total_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionStateLayoutError {
    ZeroTokens,
    SizeOverflow,
}

impl fmt::Display for DiffusionStateLayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroTokens => f.write_str("diffusion state requires at least one token"),
            Self::SizeOverflow => f.write_str("diffusion state byte layout overflowed usize"),
        }
    }
}

impl std::error::Error for DiffusionStateLayoutError {}

impl DiffusionStateLayout {
    pub fn for_tokens(tokens: usize) -> Result<Self, DiffusionStateLayoutError> {
        if tokens == 0 {
            return Err(DiffusionStateLayoutError::ZeroTokens);
        }
        let f32_bytes = tokens
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(DiffusionStateLayoutError::SizeOverflow)?;
        let byte_masks = tokens;
        let total = f32_bytes
            .checked_mul(3)
            .and_then(|bytes| bytes.checked_add(byte_masks.checked_mul(2)?))
            .ok_or(DiffusionStateLayoutError::SizeOverflow)?;
        Ok(Self {
            tokens,
            mask_bytes: byte_masks,
            confidence_bytes: f32_bytes,
            entropy_bytes: f32_bytes,
            momentum_bytes: f32_bytes,
            converged_bytes: byte_masks,
            total_bytes: total,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_mixed_dtype_and_exact() {
        let layout = DiffusionStateLayout::for_tokens(8).unwrap();
        assert_eq!(layout.confidence_bytes, 32);
        assert_eq!(layout.mask_bytes, 8);
        assert_eq!(layout.converged_bytes, 8);
        assert_eq!(layout.total_bytes, 112);
    }

    #[test]
    fn layout_rejects_zero_and_overflow() {
        assert_eq!(
            DiffusionStateLayout::for_tokens(0),
            Err(DiffusionStateLayoutError::ZeroTokens)
        );
        assert_eq!(
            DiffusionStateLayout::for_tokens(usize::MAX),
            Err(DiffusionStateLayoutError::SizeOverflow)
        );
    }
}
