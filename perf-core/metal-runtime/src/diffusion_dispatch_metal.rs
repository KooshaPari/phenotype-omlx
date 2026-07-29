//! Feature-gated Metal bindings for the diffusion scheduler stages.

use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum DiffusionDispatchError {
    #[error("diffusion token buffers must be non-empty")]
    ZeroTokens,
    #[error("diffusion buffer '{what}' has length {got}, expected {expected}")]
    BadShape {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("diffusion threshold '{what}' must be finite and in [{min}, {max}], got {got}")]
    InvalidThreshold {
        what: &'static str,
        min: f32,
        max: f32,
        got: f32,
    },
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal diffusion dispatch failed: {0}")]
    Metal(String),
    #[error("diffusion telemetry construction failed: {0}")]
    Telemetry(String),
}

/// Validate confidence/remask thresholds before they cross the FFI boundary.
///
/// Metal comparisons with NaN are well-defined but silently produce an all-zero
/// convergence/remask result, which is not equivalent to the host oracle. Keep
/// this check pure and shared by every device dispatch entry point.
pub fn validate_diffusion_threshold(
    what: &'static str,
    value: f32,
    min: f32,
    max: f32,
) -> Result<(), DiffusionDispatchError> {
    if !value.is_finite() || value < min || value > max {
        return Err(DiffusionDispatchError::InvalidThreshold {
            what,
            min,
            max,
            got: value,
        });
    }
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn validate_len(
    what: &'static str,
    got: usize,
    expected: usize,
) -> Result<(), DiffusionDispatchError> {
    if got != expected {
        return Err(DiffusionDispatchError::BadShape {
            what,
            expected,
            got,
        });
    }
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn dispatch_size(pipeline: &metal::ComputePipelineState, tokens: usize) -> metal::MTLSize {
    use metal::MTLSize;
    let width = pipeline.thread_execution_width().max(1);
    MTLSize::new(width.min(tokens as u64), 1, 1)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn diffusion_active_compact_metal(
    values: &[u32],
    active: &[u8],
    artifact: &crate::MetallibArtifact,
) -> Result<(Vec<u32>, Vec<u32>), DiffusionDispatchError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;
    let tokens = values.len();
    if tokens == 0 {
        return Err(DiffusionDispatchError::ZeroTokens);
    }
    validate_len("active", active.len(), tokens)?;
    crate::metal_cache::with_catalogued_pipeline(
        artifact,
        "active_compact",
        |device, queue, pipeline| {
            let shared = MTLResourceOptions::StorageModeShared;
            let input = device.new_buffer_with_data(
                values.as_ptr().cast::<c_void>(),
                (tokens * 4) as u64,
                shared,
            );
            let mask = device.new_buffer_with_data(
                active.as_ptr().cast::<c_void>(),
                tokens as u64,
                shared,
            );
            let compacted = device.new_buffer((tokens * 4) as u64, shared);
            let positions = device.new_buffer((tokens * 4) as u64, shared);
            let count = device.new_buffer(4, shared);
            unsafe {
                *(count.contents().cast::<u32>()) = 0;
            }
            let command = queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            for (index, buffer) in [&input, &mask, &compacted, &positions, &count]
                .iter()
                .enumerate()
            {
                encoder.set_buffer(index as u64, Some(buffer), 0);
            }
            let token_count = tokens as u32;
            encoder.set_bytes(5, 4, (&token_count as *const u32).cast());
            encoder.dispatch_threads(
                MTLSize::new(tokens as u64, 1, 1),
                dispatch_size(pipeline, tokens),
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(format!("command buffer status {:?}", command.status()));
            }
            let count = unsafe { *(count.contents().cast::<u32>()) as usize }.min(tokens);
            let mut out_values = vec![0u32; count];
            let mut out_positions = vec![0u32; count];
            unsafe {
                out_values.copy_from_slice(std::slice::from_raw_parts(
                    compacted.contents().cast(),
                    count,
                ));
                out_positions.copy_from_slice(std::slice::from_raw_parts(
                    positions.contents().cast(),
                    count,
                ));
            }
            Ok((out_values, out_positions))
        },
    )
    .map_err(DiffusionDispatchError::Metal)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn diffusion_remask_metal(
    candidate_mask: &[u8],
    confidence: &[f32],
    threshold: f32,
    artifact: &crate::MetallibArtifact,
) -> Result<Vec<u8>, DiffusionDispatchError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;
    let tokens = candidate_mask.len();
    if tokens == 0 {
        return Err(DiffusionDispatchError::ZeroTokens);
    }
    validate_diffusion_threshold("remask", threshold, 0.0, 1.0)?;
    validate_len("confidence", confidence.len(), tokens)?;
    crate::metal_cache::with_catalogued_pipeline(artifact, "remask", |device, queue, pipeline| {
        let shared = MTLResourceOptions::StorageModeShared;
        let mask = device.new_buffer_with_data(
            candidate_mask.as_ptr().cast::<c_void>(),
            tokens as u64,
            shared,
        );
        let scores = device.new_buffer_with_data(
            confidence.as_ptr().cast::<c_void>(),
            (tokens * 4) as u64,
            shared,
        );
        let output = device.new_buffer(tokens as u64, shared);
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&mask), 0);
        encoder.set_buffer(1, Some(&scores), 0);
        encoder.set_buffer(2, Some(&output), 0);
        let token_count = tokens as u32;
        encoder.set_bytes(3, 4, (&threshold as *const f32).cast());
        encoder.set_bytes(4, 4, (&token_count as *const u32).cast());
        encoder.dispatch_threads(
            MTLSize::new(tokens as u64, 1, 1),
            dispatch_size(pipeline, tokens),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!("command buffer status {:?}", command.status()));
        }
        let mut result = vec![0u8; tokens];
        unsafe {
            result.copy_from_slice(std::slice::from_raw_parts(output.contents().cast(), tokens));
        }
        Ok(result)
    })
    .map_err(DiffusionDispatchError::Metal)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn diffusion_trajectory_metal(
    previous_confidence: &[f32],
    confidence: &[f32],
    entropy: &[f32],
    confidence_threshold: f32,
    momentum_threshold: f32,
    artifact: &crate::MetallibArtifact,
) -> Result<(Vec<f32>, Vec<u8>), DiffusionDispatchError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;
    let tokens = confidence.len();
    if tokens == 0 {
        return Err(DiffusionDispatchError::ZeroTokens);
    }
    validate_diffusion_threshold("confidence", confidence_threshold, 0.0, 1.0)?;
    validate_diffusion_threshold("momentum", momentum_threshold, 0.0, f32::MAX)?;
    validate_len("previous_confidence", previous_confidence.len(), tokens)?;
    validate_len("entropy", entropy.len(), tokens)?;
    crate::metal_cache::with_catalogued_pipeline(
        artifact,
        "trajectory",
        |device, queue, pipeline| {
            let shared = MTLResourceOptions::StorageModeShared;
            let float_buffer = |data: &[f32]| {
                device.new_buffer_with_data(
                    data.as_ptr().cast::<c_void>(),
                    (data.len() * 4) as u64,
                    shared,
                )
            };
            let previous = float_buffer(previous_confidence);
            let current = float_buffer(confidence);
            let entropy_buffer = float_buffer(entropy);
            let momentum = device.new_buffer((tokens * 4) as u64, shared);
            let converged = device.new_buffer(tokens as u64, shared);
            let command = queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            for (index, buffer) in [&previous, &current, &entropy_buffer, &momentum, &converged]
                .iter()
                .enumerate()
            {
                encoder.set_buffer(index as u64, Some(buffer), 0);
            }
            let token_count = tokens as u32;
            encoder.set_bytes(5, 4, (&confidence_threshold as *const f32).cast());
            encoder.set_bytes(6, 4, (&momentum_threshold as *const f32).cast());
            encoder.set_bytes(7, 4, (&token_count as *const u32).cast());
            encoder.dispatch_threads(
                MTLSize::new(tokens as u64, 1, 1),
                dispatch_size(pipeline, tokens),
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(format!("command buffer status {:?}", command.status()));
            }
            let mut out_momentum = vec![0f32; tokens];
            let mut out_converged = vec![0u8; tokens];
            unsafe {
                out_momentum.copy_from_slice(std::slice::from_raw_parts(
                    momentum.contents().cast(),
                    tokens,
                ));
                out_converged.copy_from_slice(std::slice::from_raw_parts(
                    converged.contents().cast(),
                    tokens,
                ));
            }
            Ok((out_momentum, out_converged))
        },
    )
    .map_err(DiffusionDispatchError::Metal)
}

/// Execute active-position compaction and retain a promotion-grade outcome.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn diffusion_active_compact_metal_with_telemetry(
    values: &[u32],
    active: &[u8],
    artifact: &crate::MetallibArtifact,
) -> Result<crate::DiffusionStageOutcome<(Vec<u32>, Vec<u32>)>, DiffusionDispatchError> {
    let started = std::time::Instant::now();
    let result = diffusion_active_compact_metal(values, active, artifact);
    crate::DiffusionStageOutcome::from_result(
        crate::DiffusionStage::ActiveCompact,
        started.elapsed().as_secs_f64() * 1_000.0,
        result,
        false,
    )
    .map_err(|error| DiffusionDispatchError::Telemetry(error.to_string()))
}

/// Execute remasking and retain a promotion-grade outcome.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn diffusion_remask_metal_with_telemetry(
    candidate_mask: &[u8],
    confidence: &[f32],
    threshold: f32,
    artifact: &crate::MetallibArtifact,
) -> Result<crate::DiffusionStageOutcome<Vec<u8>>, DiffusionDispatchError> {
    let started = std::time::Instant::now();
    let result = diffusion_remask_metal(candidate_mask, confidence, threshold, artifact);
    crate::DiffusionStageOutcome::from_result(
        crate::DiffusionStage::Remask,
        started.elapsed().as_secs_f64() * 1_000.0,
        result,
        false,
    )
    .map_err(|error| DiffusionDispatchError::Telemetry(error.to_string()))
}

/// Execute trajectory update and retain a promotion-grade outcome.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn diffusion_trajectory_metal_with_telemetry(
    previous_confidence: &[f32],
    confidence: &[f32],
    entropy: &[f32],
    confidence_threshold: f32,
    momentum_threshold: f32,
    artifact: &crate::MetallibArtifact,
) -> Result<crate::DiffusionStageOutcome<(Vec<f32>, Vec<u8>)>, DiffusionDispatchError> {
    let started = std::time::Instant::now();
    let result = diffusion_trajectory_metal(
        previous_confidence,
        confidence,
        entropy,
        confidence_threshold,
        momentum_threshold,
        artifact,
    );
    crate::DiffusionStageOutcome::from_result(
        crate::DiffusionStage::Trajectory,
        started.elapsed().as_secs_f64() * 1_000.0,
        result,
        false,
    )
    .map_err(|error| DiffusionDispatchError::Telemetry(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::{validate_diffusion_threshold, DiffusionDispatchError};

    #[test]
    fn thresholds_accept_canonical_bounds() {
        assert!(validate_diffusion_threshold("confidence", 0.0, 0.0, 1.0).is_ok());
        assert!(validate_diffusion_threshold("confidence", 1.0, 0.0, 1.0).is_ok());
        assert!(validate_diffusion_threshold("momentum", 0.0, 0.0, f32::MAX).is_ok());
    }

    #[test]
    fn thresholds_reject_non_finite_and_out_of_range_values() {
        for value in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, -0.01, 1.01] {
            assert!(matches!(
                validate_diffusion_threshold("confidence", value, 0.0, 1.0),
                Err(DiffusionDispatchError::InvalidThreshold { .. })
            ));
        }
    }
}
