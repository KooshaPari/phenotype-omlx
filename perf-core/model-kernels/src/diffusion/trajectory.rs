//! Per-position confidence trajectory state for masked diffusion decoding.

use crate::error::{KernelError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryState {
    pub confidence: Vec<f32>,
    pub entropy: Vec<f32>,
    pub momentum: Vec<f32>,
    pub step: u32,
    pub converged: Vec<bool>,
}

pub fn update_trajectory(
    previous_confidence: &[f32],
    confidence: &[f32],
    entropy: &[f32],
    step: u32,
    confidence_threshold: f32,
    momentum_threshold: f32,
) -> Result<TrajectoryState> {
    if previous_confidence.len() != confidence.len() {
        return Err(KernelError::DimMismatch {
            what: "trajectory previous_confidence vs confidence",
            expected: previous_confidence.len(),
            got: confidence.len(),
        });
    }
    if confidence.len() != entropy.len() {
        return Err(KernelError::DimMismatch {
            what: "trajectory confidence vs entropy",
            expected: confidence.len(),
            got: entropy.len(),
        });
    }
    if !confidence_threshold.is_finite() || !(0.0..=1.0).contains(&confidence_threshold) {
        return Err(KernelError::OutOfRange {
            what: "trajectory confidence_threshold",
            min: 0.0,
            max: 1.0,
            got: confidence_threshold,
        });
    }
    if !momentum_threshold.is_finite() || momentum_threshold < 0.0 {
        return Err(KernelError::OutOfRange {
            what: "trajectory momentum_threshold",
            min: 0.0,
            max: f32::INFINITY,
            got: momentum_threshold,
        });
    }
    for (index, (&previous, &current)) in previous_confidence.iter().zip(confidence).enumerate() {
        if !previous.is_finite() || !current.is_finite() {
            return Err(KernelError::NonFiniteValue {
                what: "trajectory confidence",
                index,
            });
        }
    }
    if let Some(index) = entropy
        .iter()
        .position(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(KernelError::NonFiniteValue {
            what: "trajectory entropy",
            index,
        });
    }
    let momentum = previous_confidence
        .iter()
        .zip(confidence)
        .map(|(previous, current)| (current - previous).abs())
        .collect::<Vec<_>>();
    let converged = confidence
        .iter()
        .zip(&momentum)
        .map(|(current, delta)| *current >= confidence_threshold && *delta <= momentum_threshold)
        .collect();
    Ok(TrajectoryState {
        confidence: confidence.to_vec(),
        entropy: entropy.to_vec(),
        momentum,
        step,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_tracks_momentum_and_convergence() {
        let state =
            update_trajectory(&[0.7, 0.2], &[0.8, 0.5], &[0.1, 0.9], 3, 0.75, 0.15).unwrap();
        assert!((state.momentum[0] - 0.1).abs() < 1e-6);
        assert!((state.momentum[1] - 0.3).abs() < 1e-6);
        assert_eq!(state.converged, [true, false]);
        assert_eq!(state.step, 3);
    }

    #[test]
    fn rejects_shape_and_non_finite_inputs() {
        assert!(matches!(
            update_trajectory(&[0.1], &[0.2, 0.3], &[0.1, 0.2], 0, 0.5, 0.1),
            Err(KernelError::DimMismatch { .. })
        ));
        assert!(matches!(
            update_trajectory(&[0.1], &[f32::NAN], &[0.1], 0, 0.5, 0.1),
            Err(KernelError::NonFiniteValue { .. })
        ));
    }
}
