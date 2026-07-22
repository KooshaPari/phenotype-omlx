//! Fused adaptive RMS normalization for diffusion transformer blocks.

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AdaLnError {
    #[error("tokens and dimension must be non-zero")]
    ZeroDimension,
    #[error("input and conditioning lengths must equal tokens * dim")]
    BadShape,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal AdaLN failed: {0}")]
    Metal(String),
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn adaln_rms_metal(
    x: &[f32],
    scale: &[f32],
    shift: &[f32],
    tokens: usize,
    dim: usize,
    epsilon: f32,
    artifact: &crate::MetallibArtifact,
) -> Result<Vec<f32>, AdaLnError> {
    use metal::{Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;

    if tokens == 0 || dim == 0 {
        return Err(AdaLnError::ZeroDimension);
    }
    let len = tokens.checked_mul(dim).ok_or(AdaLnError::BadShape)?;
    if x.len() != len || scale.len() != len || shift.len() != len {
        return Err(AdaLnError::BadShape);
    }
    let device = Device::system_default()
        .ok_or_else(|| AdaLnError::Metal("no system Metal device".into()))?;
    let library = device
        .new_library_with_data(artifact.bytes())
        .map_err(AdaLnError::Metal)?;
    let function = library
        .get_function("adaln_rms_f32", None)
        .map_err(AdaLnError::Metal)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(AdaLnError::Metal)?;
    let shared = MTLResourceOptions::StorageModeShared;
    let buffer = |data: &[f32]| {
        device.new_buffer_with_data(
            data.as_ptr().cast::<c_void>(),
            std::mem::size_of_val(data) as u64,
            shared,
        )
    };
    let x_buffer = buffer(x);
    let scale_buffer = buffer(scale);
    let shift_buffer = buffer(shift);
    let out_buffer = device.new_buffer((len * std::mem::size_of::<f32>()) as u64, shared);
    let queue = device.new_command_queue();
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    for (index, value) in [
        (0, &x_buffer),
        (1, &scale_buffer),
        (2, &shift_buffer),
        (3, &out_buffer),
    ] {
        encoder.set_buffer(index, Some(value), 0);
    }
    let token_u32 = tokens as u32;
    let dim_u32 = dim as u32;
    encoder.set_bytes(4, 4, (&token_u32 as *const u32).cast());
    encoder.set_bytes(5, 4, (&dim_u32 as *const u32).cast());
    encoder.set_bytes(6, 4, (&epsilon as *const f32).cast());
    let width = pipeline.thread_execution_width().max(1);
    encoder.dispatch_threads(
        MTLSize::new(tokens as u64, dim as u64, 1),
        MTLSize::new(1, width.min(dim as u64), 1),
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(AdaLnError::Metal(format!(
            "command buffer status {:?}",
            command.status()
        )));
    }
    let mut output = vec![0.0; len];
    unsafe {
        output.copy_from_slice(std::slice::from_raw_parts(
            out_buffer.contents().cast(),
            len,
        ));
    }
    Ok(output)
}
