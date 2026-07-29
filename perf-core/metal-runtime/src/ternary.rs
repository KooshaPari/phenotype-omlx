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
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal ternary GEMM failed: {0}")]
    Metal(String),
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

    if m == 0 || k == 0 || n == 0 {
        return Err(TernaryGemmError::ZeroDimension);
    }
    let packed_stride = k.div_ceil(4);
    if activations.len() != m * k || packed_weights.len() != n * packed_stride {
        return Err(TernaryGemmError::BadShape);
    }
    if scales.len() != n {
        return Err(TernaryGemmError::BadScaleShape);
    }
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
            let out_len = m * n;
            let out_buffer =
                device.new_buffer((out_len * std::mem::size_of::<f32>()) as u64, shared);
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
