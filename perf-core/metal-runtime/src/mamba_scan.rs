//! Artifact-backed chunked Mamba selective scan.
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MambaScanError {
    #[error("sequence and parameter lengths must match and state must match a_log")]
    BadShape,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal Mamba scan failed: {0}")]
    Metal(String),
}

pub fn validate_scan_shapes(
    u: &[f32],
    dt: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    a_log: &[f32],
    state: &[f32],
) -> Result<(), MambaScanError> {
    if u.is_empty()
        || dt.len() != u.len()
        || b.len() != u.len()
        || c.len() != u.len()
        || d.len() != u.len()
        || a_log.is_empty()
        || state.len() != a_log.len()
    {
        return Err(MambaScanError::BadShape);
    }
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn mamba_selective_scan_metal(
    u: &[f32],
    dt: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    a_log: &[f32],
    state: &mut [f32],
    artifact: &crate::MetallibArtifact,
) -> Result<Vec<f32>, MambaScanError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;
    validate_scan_shapes(u, dt, b, c, d, a_log, state)?;
    crate::metal_cache::with_pipeline(
        artifact,
        "mamba_selective_scan_f32",
        |device, queue, pipeline| {
            let sh = MTLResourceOptions::StorageModeShared;
            let buf = |x: &[f32]| {
                device.new_buffer_with_data(
                    x.as_ptr().cast::<c_void>(),
                    std::mem::size_of_val(x) as u64,
                    sh,
                )
            };
            let ub = buf(u);
            let dtb = buf(dt);
            let bb = buf(b);
            let cb = buf(c);
            let db = buf(d);
            let ab = buf(a_log);
            let sb = buf(state);
            let out = device.new_buffer((u.len() * 4) as u64, sh);
            let steps = u.len() as u32;
            let dim = a_log.len() as u32;
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            for (i, x) in [&ub, &dtb, &bb, &cb, &db, &ab, &sb, &out]
                .into_iter()
                .enumerate()
            {
                enc.set_buffer(i as u64, Some(x), 0);
            }
            enc.set_bytes(8, 4, (&steps as *const u32).cast());
            enc.set_bytes(9, 4, (&dim as *const u32).cast());
            enc.dispatch_threads(MTLSize::new(256, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            if cmd.status() != MTLCommandBufferStatus::Completed {
                return Err(format!("command buffer status {:?}", cmd.status()));
            }
            unsafe {
                state.copy_from_slice(std::slice::from_raw_parts(
                    sb.contents().cast(),
                    state.len(),
                ));
                Ok(std::slice::from_raw_parts(out.contents().cast(), u.len()).to_vec())
            }
        },
    )
    .map_err(MambaScanError::Metal)
}
