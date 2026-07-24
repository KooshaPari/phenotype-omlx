//! RWKV-7 channel-mix recurrent step on Metal.

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RwkvError {
    #[error("RWKV input and state must each contain four channels")]
    BadShape,
    #[error("decay must be finite")]
    BadDecay,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal RWKV step failed: {0}")]
    Metal(String),
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn rwkv7_time_mix_metal(
    x: &[f32; 4],
    state: &mut [f32; 4],
    mix_k: f32,
    mix_v: f32,
    mix_r: f32,
    mix_g: f32,
    decay: f32,
    artifact: &crate::MetallibArtifact,
) -> Result<f32, RwkvError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;
    if !decay.is_finite() {
        return Err(RwkvError::BadDecay);
    }
    crate::metal_cache::with_pipeline(artifact, "rwkv7_time_mix_f32", |device, queue, pipeline| {
        let shared = MTLResourceOptions::StorageModeShared;
        let buf = |data: &[f32]| {
            device.new_buffer_with_data(
                data.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(data) as u64,
                shared,
            )
        };
        let input = buf(x);
        let state_buf = buf(state);
        let output = device.new_buffer(std::mem::size_of::<f32>() as u64, shared);
        let params = [mix_k, mix_v, mix_r, mix_g, decay];
        let params_buf = buf(&params);
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&input), 0);
        encoder.set_buffer(1, Some(&state_buf), 0);
        encoder.set_buffer(2, Some(&params_buf), 0);
        encoder.set_buffer(3, Some(&output), 0);
        encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!("command buffer status {:?}", command.status()));
        }
        unsafe {
            state.copy_from_slice(std::slice::from_raw_parts(state_buf.contents().cast(), 4));
            Ok(*(output.contents().cast::<f32>()))
        }
    })
    .map_err(RwkvError::Metal)
}
