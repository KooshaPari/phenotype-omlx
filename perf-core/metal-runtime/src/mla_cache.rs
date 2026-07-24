//! DeepSeek-style MLA cache attention on Metal.
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MlaCacheError {
    #[error("MLA cache buffers have inconsistent dimensions")]
    BadShape,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal MLA cache failed: {0}")]
    Metal(String),
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn mla_cache_attend_metal(
    q_latent: &[f32],
    q_rope: &[f32],
    compressed_kv: &[f32],
    k_rope: &[f32],
    d_latent: usize,
    d_rope: usize,
    artifact: &crate::MetallibArtifact,
) -> Result<Vec<f32>, MlaCacheError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;
    if d_latent == 0
        || d_rope == 0
        || q_latent.len() != d_latent
        || q_rope.len() != d_rope
        || compressed_kv.len() % d_latent != 0
        || k_rope.len() % d_rope != 0
        || compressed_kv.len() / d_latent != k_rope.len() / d_rope
    {
        return Err(MlaCacheError::BadShape);
    }
    crate::metal_cache::with_pipeline(
        artifact,
        "mla_cache_attend_f32",
        |device, queue, pipeline| {
            let sh = MTLResourceOptions::StorageModeShared;
            let fb = |d: &[f32]| {
                device.new_buffer_with_data(
                    d.as_ptr().cast::<c_void>(),
                    std::mem::size_of_val(d) as u64,
                    sh,
                )
            };
            let qb = fb(q_latent);
            let rb = fb(q_rope);
            let kb = fb(compressed_kv);
            let krb = fb(k_rope);
            let out = device.new_buffer((d_latent * 4) as u64, sh);
            let entries = (compressed_kv.len() / d_latent) as u32;
            let dl = d_latent as u32;
            let dr = d_rope as u32;
            let c = queue.new_command_buffer();
            let e = c.new_compute_command_encoder();
            e.set_compute_pipeline_state(pipeline);
            e.set_buffer(0, Some(&qb), 0);
            e.set_buffer(1, Some(&rb), 0);
            e.set_buffer(2, Some(&kb), 0);
            e.set_buffer(3, Some(&krb), 0);
            e.set_buffer(4, Some(&out), 0);
            e.set_bytes(5, 4, (&entries as *const u32).cast());
            e.set_bytes(6, 4, (&dl as *const u32).cast());
            e.set_bytes(7, 4, (&dr as *const u32).cast());
            e.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            e.end_encoding();
            c.commit();
            c.wait_until_completed();
            if c.status() != MTLCommandBufferStatus::Completed {
                return Err(format!("command buffer status {:?}", c.status()));
            }
            unsafe { Ok(std::slice::from_raw_parts(out.contents().cast(), d_latent).to_vec()) }
        },
    )
    .map_err(MlaCacheError::Metal)
}
