//! Mamba selective-scan single-step Metal kernel.

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MambaError {
    #[error("state and a_log must be non-empty and equal length")]
    BadShape,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal Mamba scan failed: {0}")]
    Metal(String),
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn mamba_selective_step_metal(
    u: f32,
    dt: f32,
    b: f32,
    c: f32,
    d: f32,
    a_log: &[f32],
    state: &mut [f32],
    artifact: &crate::MetallibArtifact,
) -> Result<f32, MambaError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;
    if a_log.is_empty() || a_log.len() != state.len() {
        return Err(MambaError::BadShape);
    }
    crate::metal_cache::with_pipeline(
        artifact,
        "mamba_selective_step_f32",
        |device, queue, pipeline| {
            let shared = MTLResourceOptions::StorageModeShared;
            let buf = |data: &[f32]| {
                device.new_buffer_with_data(
                    data.as_ptr().cast::<c_void>(),
                    std::mem::size_of_val(data) as u64,
                    shared,
                )
            };
            let al = buf(a_log);
            let st = buf(state);
            let out = device.new_buffer(4, shared);
            let params = [u, dt, b, c, d];
            let p = buf(&params);
            let command = queue.new_command_buffer();
            let enc = command.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(&al), 0);
            enc.set_buffer(1, Some(&st), 0);
            enc.set_buffer(2, Some(&p), 0);
            enc.set_buffer(3, Some(&out), 0);
            let n = state.len() as u32;
            enc.set_bytes(4, 4, (&n as *const u32).cast());
            enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            enc.end_encoding();
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(format!("command buffer status {:?}", command.status()));
            }
            unsafe {
                state.copy_from_slice(std::slice::from_raw_parts(
                    st.contents().cast(),
                    state.len(),
                ));
                Ok(*(out.contents().cast::<f32>()))
            }
        },
    )
    .map_err(MambaError::Metal)
}
