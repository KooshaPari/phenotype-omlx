//! Packed 2-bit ternary GEMM for Bonsai/BitNet-style weights.

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TernaryGemmError {
    #[error("matrix dimensions must be non-zero")]
    ZeroDimension,
    #[error("activation length must equal m*k and packed weight length must equal n*ceil(k/4)")]
    BadShape,
    #[error("scale length must equal n")]
    BadScaleShape,
    #[error("ternary dimensions exceed the u32 Metal kernel contract")]
    DimensionLimit,
    #[error("ternary buffer size overflow")]
    SizeOverflow,
    #[error("ternary scale at index {index} must be finite")]
    NonFiniteScale { index: usize },
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal ternary GEMM failed: {0}")]
    Metal(String),
}

#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
fn validate_ternary_shape(
    activations: &[f32],
    packed_weights: &[u8],
    scales: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<usize, TernaryGemmError> {
    if m == 0 || k == 0 || n == 0 {
        return Err(TernaryGemmError::ZeroDimension);
    }
    if m > u32::MAX as usize || k > u32::MAX as usize || n > u32::MAX as usize {
        return Err(TernaryGemmError::DimensionLimit);
    }
    let packed_stride = k.checked_add(3).ok_or(TernaryGemmError::SizeOverflow)? / 4;
    let activation_len = m.checked_mul(k).ok_or(TernaryGemmError::SizeOverflow)?;
    let packed_len = n
        .checked_mul(packed_stride)
        .ok_or(TernaryGemmError::SizeOverflow)?;
    if activations.len() != activation_len || packed_weights.len() != packed_len {
        return Err(TernaryGemmError::BadShape);
    }
    if scales.len() != n {
        return Err(TernaryGemmError::BadScaleShape);
    }
    if let Some(index) = scales.iter().position(|scale| !scale.is_finite()) {
        return Err(TernaryGemmError::NonFiniteScale { index });
    }
    m.checked_mul(n).ok_or(TernaryGemmError::SizeOverflow)?;
    Ok(packed_stride)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn ternary_gemm_metal(
    activations: &[f32],
    packed_weights: &[u8],
    scales: &[f32],
    m: usize,
    k: usize,
    n: usize,
    artifact: &crate::MetallibArtifact,
) -> Result<Vec<f32>, TernaryGemmError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;

    let _packed_stride = validate_ternary_shape(activations, packed_weights, scales, m, k, n)?;
    let out_len = m.checked_mul(n).ok_or(TernaryGemmError::SizeOverflow)?;
    let out_bytes = out_len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(TernaryGemmError::SizeOverflow)? as u64;
    crate::metal_cache::with_catalogued_pipeline(
        artifact,
        "ternary_pack",
        |device, queue, pipeline| {
            let shared = MTLResourceOptions::StorageModeShared;
            let float_buffer = |data: &[f32]| {
                device.new_buffer_with_data(
                    data.as_ptr().cast::<c_void>(),
                    std::mem::size_of_val(data) as u64,
                    shared,
                )
            };
            let input = float_buffer(activations);
            let weights = device.new_buffer_with_data(
                packed_weights.as_ptr().cast::<c_void>(),
                packed_weights.len() as u64,
                shared,
            );
            let scales_buffer = float_buffer(scales);
            let out_buffer = device.new_buffer(out_bytes, shared);
            let command = queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            for (index, value) in [
                (0, &input),
                (1, &weights),
                (2, &scales_buffer),
                (3, &out_buffer),
            ] {
                encoder.set_buffer(index, Some(value), 0);
            }
            let dims = [m as u32, k as u32, n as u32];
            for (index, value) in dims.iter().enumerate() {
                encoder.set_bytes((4 + index) as u64, 4, (value as *const u32).cast());
            }
            let width = pipeline.thread_execution_width().max(1);
            encoder.dispatch_threads(
                MTLSize::new(m as u64, n as u64, 1),
                MTLSize::new(width.min(m as u64), 1, 1),
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(format!("command buffer status {:?}", command.status()));
            }
            let mut output = vec![0.0; out_len];
            unsafe {
                output.copy_from_slice(std::slice::from_raw_parts(
                    out_buffer.contents().cast(),
                    out_len,
                ));
            }
            Ok(output)
        },
    )
    .map_err(TernaryGemmError::Metal)
}

#[cfg(test)]
mod tests {
    use super::{validate_ternary_shape, TernaryGemmError};

    #[test]
    fn shape_contract_rejects_non_finite_scales() {
        let error = validate_ternary_shape(&[1.0; 2], &[0], &[f32::NAN], 1, 2, 1)
            .expect_err("NaN scale must fail closed");
        assert_eq!(error, TernaryGemmError::NonFiniteScale { index: 0 });
    }

    #[test]
    fn shape_contract_requires_exact_buffers() {
        let error = validate_ternary_shape(&[1.0], &[0], &[1.0], 1, 2, 1)
            .expect_err("activation shape must be exact");
        assert_eq!(error, TernaryGemmError::BadShape);
    }
}
