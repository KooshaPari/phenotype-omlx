//! Active-position compaction for masked diffusion decoding.
//!
//! Compaction keeps the denoiser's next pass proportional to unresolved
//! positions while preserving original sequence indices for scatter-back.

use crate::error::{KernelError, Result};

/// Return the stable, ascending indices of positions that remain masked.
pub fn active_positions(mask: &[bool]) -> Vec<u32> {
    mask.iter()
        .enumerate()
        .filter_map(|(index, active)| active.then_some(index as u32))
        .collect()
}

/// Compact values at masked positions and return their original indices.
pub fn compact_active<T: Copy>(values: &[T], mask: &[bool]) -> Result<(Vec<T>, Vec<u32>)> {
    if values.len() != mask.len() {
        return Err(KernelError::DimMismatch {
            what: "compact_active.values vs mask",
            expected: values.len(),
            got: mask.len(),
        });
    }
    let positions = active_positions(mask);
    let compacted = positions
        .iter()
        .map(|&index| values[index as usize])
        .collect();
    Ok((compacted, positions))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_are_stable_and_ascending() {
        assert_eq!(active_positions(&[false, true, false, true]), vec![1, 3]);
    }

    #[test]
    fn compaction_preserves_values_and_scatter_indices() {
        let (values, positions) = compact_active(&[10u32, 20, 30, 40], &[true, false, true, false])
            .unwrap();
        assert_eq!(values, vec![10, 30]);
        assert_eq!(positions, vec![0, 2]);
    }

    #[test]
    fn rejects_shape_mismatch() {
        let err = compact_active(&[1u32], &[true, false]).unwrap_err();
        assert!(matches!(err, KernelError::DimMismatch { .. }));
    }
}
