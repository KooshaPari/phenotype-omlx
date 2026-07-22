//! Deterministic flow-matching schedules and classifier-free guidance.

use crate::error::{KernelError, Result};

/// Construct a monotonically descending sigma schedule for Euler flow steps.
/// `shift` bends the schedule toward low-noise steps without changing its
/// endpoints; `sigma_min` prevents the final step from reaching zero.
pub fn flow_sigma_schedule(steps: usize, shift: f32, sigma_min: f32) -> Result<Vec<f32>> {
    if steps == 0 || !shift.is_finite() || shift <= 0.0 || !sigma_min.is_finite() {
        return Err(KernelError::OutOfRange {
            what: "flow schedule parameters",
            min: 0.0,
            max: f32::INFINITY,
            got: shift,
        });
    }
    if !(0.0..1.0).contains(&sigma_min) {
        return Err(KernelError::OutOfRange {
            what: "sigma_min",
            min: 0.0,
            max: 1.0,
            got: sigma_min,
        });
    }
    if steps == 1 {
        return Ok(vec![1.0]);
    }
    let denom = (steps - 1) as f32;
    Ok((0..steps)
        .map(|i| {
            let t = i as f32 / denom;
            let warped = (t * shift) / (1.0 + (shift - 1.0) * t);
            (1.0 - warped).max(sigma_min)
        })
        .collect())
}

/// Blend unconditional and conditional velocity predictions for CFG.
pub fn classifier_free_guidance(
    unconditional: &[f32],
    conditional: &[f32],
    guidance_scale: f32,
) -> Result<Vec<f32>> {
    if unconditional.len() != conditional.len() {
        return Err(KernelError::DimMismatch {
            what: "cfg unconditional vs conditional",
            expected: unconditional.len(),
            got: conditional.len(),
        });
    }
    if !guidance_scale.is_finite() || guidance_scale < 0.0 {
        return Err(KernelError::OutOfRange {
            what: "guidance_scale",
            min: 0.0,
            max: f32::INFINITY,
            got: guidance_scale,
        });
    }
    Ok(unconditional
        .iter()
        .zip(conditional)
        .map(|(u, c)| u + guidance_scale * (c - u))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_descends_and_respects_floor() {
        let schedule = flow_sigma_schedule(8, 2.0, 0.01).unwrap();
        assert_eq!(schedule.len(), 8);
        assert_eq!(schedule[0], 1.0);
        assert!(schedule.windows(2).all(|w| w[0] >= w[1]));
        assert!(schedule.last().unwrap() >= &0.01);
    }

    #[test]
    fn cfg_scale_zero_is_unconditional() {
        assert_eq!(
            classifier_free_guidance(&[1.0, 2.0], &[3.0, 4.0], 0.0).unwrap(),
            [1.0, 2.0]
        );
        assert_eq!(
            classifier_free_guidance(&[1.0, 2.0], &[3.0, 4.0], 2.0).unwrap(),
            [5.0, 6.0]
        );
    }
}
