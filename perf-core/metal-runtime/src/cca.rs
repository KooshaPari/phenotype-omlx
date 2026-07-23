//! ZAYA-style compressed-context attention on Metal.
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CcaError {
    #[error("CCA buffers have inconsistent dimensions")]
    BadShape,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal CCA failed: {0}")]
    Metal(String),
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn cca_block_attend_metal(
    q: &[f32],
    summaries: &[f32],
    scales: &[f32],
    sizes: &[u32],
    head_dim: usize,
    artifact: &crate::MetallibArtifact,
) -> Result<Vec<f32>, CcaError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;
    if head_dim == 0
        || summaries.len() != scales.len() * head_dim
        || scales.len() != sizes.len()
        || q.len() != head_dim
    {
        return Err(CcaError::BadShape);
    }
    crate::metal_cache::with_pipeline(
        artifact,
        "cca_block_attend_f32",
        |device, queue, pipeline| {
            let sh = MTLResourceOptions::StorageModeShared;
            let fb = |d: &[f32]| {
                device.new_buffer_with_data(
                    d.as_ptr().cast::<c_void>(),
                    std::mem::size_of_val(d) as u64,
                    sh,
                )
            };
            let qb = fb(q);
            let sb = fb(summaries);
            let cb = fb(scales);
            let ub = device.new_buffer_with_data(
                sizes.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(sizes) as u64,
                sh,
            );
            let out = device.new_buffer((head_dim * 4) as u64, sh);
            let blocks = scales.len() as u32;
            let dim = head_dim as u32;
            let c = queue.new_command_buffer();
            let e = c.new_compute_command_encoder();
            e.set_compute_pipeline_state(pipeline);
            e.set_buffer(0, Some(&qb), 0);
            e.set_buffer(1, Some(&sb), 0);
            e.set_buffer(2, Some(&cb), 0);
            e.set_buffer(3, Some(&ub), 0);
            e.set_buffer(4, Some(&out), 0);
            e.set_bytes(5, 4, (&blocks as *const u32).cast());
            e.set_bytes(6, 4, (&dim as *const u32).cast());
            e.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            e.end_encoding();
            c.commit();
            c.wait_until_completed();
            if c.status() != MTLCommandBufferStatus::Completed {
                return Err(format!("command buffer status {:?}", c.status()));
            }
            unsafe { Ok(std::slice::from_raw_parts(out.contents().cast(), head_dim).to_vec()) }
        },
    )
    .map_err(CcaError::Metal)
}
