//! RetNet recurrent retention step on Metal.
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RetNetError {
    #[error("head dimension must be non-zero and buffers must match it")]
    BadShape,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal RetNet step failed: {0}")]
    Metal(String),
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn retnet_retention_step_metal(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    state: &mut [f32],
    decay: f32,
    head_dim: usize,
    artifact: &crate::MetallibArtifact,
) -> Result<Vec<f32>, RetNetError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;
    if head_dim == 0
        || q.len() != head_dim
        || k.len() != head_dim
        || v.len() != head_dim
        || state.len() != head_dim * head_dim
    {
        return Err(RetNetError::BadShape);
    }
    crate::metal_cache::with_pipeline(
        artifact,
        "retnet_retention_step_f32",
        |device, queue, pipeline| {
            let sh = MTLResourceOptions::StorageModeShared;
            let b = |d: &[f32]| {
                device.new_buffer_with_data(
                    d.as_ptr().cast::<c_void>(),
                    std::mem::size_of_val(d) as u64,
                    sh,
                )
            };
            let qb = b(q);
            let kb = b(k);
            let vb = b(v);
            let sb = b(state);
            let db = b(&[decay]);
            let out = device.new_buffer((head_dim * 4) as u64, sh);
            let next = device.new_buffer((head_dim * head_dim * 4) as u64, sh);
            let n = head_dim as u32;
            let c = queue.new_command_buffer();
            let e = c.new_compute_command_encoder();
            e.set_compute_pipeline_state(pipeline);
            e.set_buffer(0, Some(&qb), 0);
            e.set_buffer(1, Some(&kb), 0);
            e.set_buffer(2, Some(&vb), 0);
            e.set_buffer(3, Some(&sb), 0);
            e.set_buffer(4, Some(&db), 0);
            e.set_buffer(5, Some(&out), 0);
            e.set_buffer(6, Some(&next), 0);
            e.set_bytes(7, 4, (&n as *const u32).cast());
            e.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            e.end_encoding();
            c.commit();
            c.wait_until_completed();
            if c.status() != MTLCommandBufferStatus::Completed {
                return Err(format!("command buffer status {:?}", c.status()));
            }
            unsafe {
                state.copy_from_slice(std::slice::from_raw_parts(
                    next.contents().cast(),
                    state.len(),
                ));
                Ok(std::slice::from_raw_parts(out.contents().cast(), head_dim).to_vec())
            }
        },
    )
    .map_err(RetNetError::Metal)
}
