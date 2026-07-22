//! Fused classifier-free-guided Euler flow step.

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FlowStepError {
    #[error("flow vectors must be non-empty")]
    Empty,
    #[error("flow vectors must have equal lengths")]
    BadShape,
    #[error("guidance scale and dt must be finite")]
    NonFinite,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal flow step failed: {0}")]
    Metal(String),
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn flow_cfg_step_metal(
    x: &[f32],
    velocity_uncond: &[f32],
    velocity_cond: &[f32],
    guidance_scale: f32,
    dt: f32,
    artifact: &crate::MetallibArtifact,
) -> Result<Vec<f32>, FlowStepError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;
    if x.is_empty() {
        return Err(FlowStepError::Empty);
    }
    if x.len() != velocity_uncond.len() || x.len() != velocity_cond.len() {
        return Err(FlowStepError::BadShape);
    }
    if !guidance_scale.is_finite() || !dt.is_finite() {
        return Err(FlowStepError::NonFinite);
    }
    crate::metal_cache::with_pipeline(artifact, "flow_cfg_step_f32", |device, queue, pipeline| {
        let shared = MTLResourceOptions::StorageModeShared;
        let buffer = |data: &[f32]| {
            device.new_buffer_with_data(
                data.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(data) as u64,
                shared,
            )
        };
        let xb = buffer(x);
        let ub = buffer(velocity_uncond);
        let cb = buffer(velocity_cond);
        let ob = device.new_buffer((x.len() * std::mem::size_of::<f32>()) as u64, shared);
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for (index, value) in [(0, &xb), (1, &ub), (2, &cb), (3, &ob)] {
            encoder.set_buffer(index, Some(value), 0);
        }
        let n = x.len() as u32;
        encoder.set_bytes(4, 4, (&n as *const u32).cast());
        encoder.set_bytes(5, 4, (&guidance_scale as *const f32).cast());
        encoder.set_bytes(6, 4, (&dt as *const f32).cast());
        let width = pipeline.thread_execution_width().max(1);
        encoder.dispatch_threads(
            MTLSize::new(x.len() as u64, 1, 1),
            MTLSize::new(width.min(x.len() as u64), 1, 1),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!("command buffer status {:?}", command.status()));
        }
        let mut out = vec![0.0; x.len()];
        unsafe {
            out.copy_from_slice(std::slice::from_raw_parts(ob.contents().cast(), x.len()));
        }
        Ok(out)
    })
    .map_err(FlowStepError::Metal)
}
