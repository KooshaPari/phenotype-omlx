//! LFM2-style single-channel short-convolution step on Metal.

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ShortConvError {
    #[error("kernel must be non-empty")]
    EmptyKernel,
    #[error("state length must equal kernel length minus one")]
    BadState,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal short convolution failed: {0}")]
    Metal(String),
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn short_conv1d_step_metal(
    x: f32,
    kernel: &[f32],
    state: &mut [f32],
    artifact: &crate::MetallibArtifact,
) -> Result<f32, ShortConvError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;

    if kernel.is_empty() {
        return Err(ShortConvError::EmptyKernel);
    }
    if state.len() != kernel.len() - 1 {
        return Err(ShortConvError::BadState);
    }
    crate::metal_cache::with_pipeline(
        artifact,
        "short_conv1d_step_f32",
        |device, queue, pipeline| {
            let shared = MTLResourceOptions::StorageModeShared;
            let input = device.new_buffer_with_data(
                (&x as *const f32).cast::<c_void>(),
                std::mem::size_of::<f32>() as u64,
                shared,
            );
            let weights = device.new_buffer_with_data(
                kernel.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(kernel) as u64,
                shared,
            );
            let history = device.new_buffer_with_data(
                state.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(state) as u64,
                shared,
            );
            let output = device.new_buffer(std::mem::size_of::<f32>() as u64, shared);
            let command = queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&input), 0);
            encoder.set_buffer(1, Some(&weights), 0);
            encoder.set_buffer(2, Some(&history), 0);
            encoder.set_buffer(3, Some(&output), 0);
            let taps = kernel.len() as u32;
            encoder.set_bytes(4, 4, (&taps as *const u32).cast());
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(format!("command buffer status {:?}", command.status()));
            }
            let value = unsafe { *(output.contents().cast::<f32>()) };
            if !state.is_empty() {
                state.copy_within(1.., 0);
                state[state.len() - 1] = x;
            }
            Ok(value)
        },
    )
    .map_err(ShortConvError::Metal)
}
