//! 3D rotary positional embedding for video and diffusion token layouts.

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Rope3dError {
    #[error("head_dim must be even and non-zero")]
    InvalidHeadDim,
    #[error("rotary pairs per axis must be non-zero")]
    InvalidRotaryPairs,
    #[error("input shape does not match tokens * heads * head_dim")]
    BadInputShape,
    #[error("position count must equal tokens")]
    BadPositionShape,
    #[error("inverse-frequency vectors must match rotary pairs per axis")]
    BadFrequencyShape,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal 3D RoPE failed: {0}")]
    Metal(String),
}

/// Apply axis-separated 3D RoPE to Q and K, returning `(q_out, k_out)`.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn rope_3d_metal(
    q: &[f32],
    k: &[f32],
    // Each entry is `(time, height, width, padding)`; the padding preserves
    // Metal `uint3`'s 16-byte stride.
    positions: &[[u32; 4]],
    inv_time: &[f32],
    inv_height: &[f32],
    inv_width: &[f32],
    heads: usize,
    head_dim: usize,
    artifact: &crate::MetallibArtifact,
) -> Result<(Vec<f32>, Vec<f32>), Rope3dError> {
    use metal::{Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;

    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(Rope3dError::InvalidHeadDim);
    }
    let pairs = head_dim / 6;
    if pairs == 0
        || inv_time.len() != pairs
        || inv_height.len() != pairs
        || inv_width.len() != pairs
    {
        return Err(Rope3dError::BadFrequencyShape);
    }
    let tokens = positions.len();
    let expected = tokens
        .checked_mul(heads)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or(Rope3dError::BadInputShape)?;
    if q.len() != expected || k.len() != expected {
        return Err(Rope3dError::BadInputShape);
    }
    let device = Device::system_default()
        .ok_or_else(|| Rope3dError::Metal("no system Metal device".into()))?;
    let library = device
        .new_library_with_data(artifact.bytes())
        .map_err(Rope3dError::Metal)?;
    let function = library
        .get_function("rope_3d_f32", None)
        .map_err(Rope3dError::Metal)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(Rope3dError::Metal)?;
    let shared = MTLResourceOptions::StorageModeShared;
    let buffer = |data: &[u8]| {
        device.new_buffer_with_data(data.as_ptr().cast::<c_void>(), data.len() as u64, shared)
    };
    let bytes = |ptr: *const f32, len: usize| unsafe {
        std::slice::from_raw_parts(ptr.cast::<u8>(), len * std::mem::size_of::<f32>())
    };
    let q_buffer = buffer(bytes(q.as_ptr(), q.len()));
    let k_buffer = buffer(bytes(k.as_ptr(), k.len()));
    let positions_buffer = buffer(unsafe {
        std::slice::from_raw_parts(
            positions.as_ptr().cast::<u8>(),
            std::mem::size_of_val(positions),
        )
    });
    let time_buffer = buffer(bytes(inv_time.as_ptr(), inv_time.len()));
    let height_buffer = buffer(bytes(inv_height.as_ptr(), inv_height.len()));
    let width_buffer = buffer(bytes(inv_width.as_ptr(), inv_width.len()));
    let mut q_out = vec![0.0; expected];
    let mut k_out = vec![0.0; expected];
    let q_out_buffer = device.new_buffer((expected * std::mem::size_of::<f32>()) as u64, shared);
    let k_out_buffer = device.new_buffer((expected * std::mem::size_of::<f32>()) as u64, shared);
    let queue = device.new_command_queue();
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    for (index, value) in [
        (0, &q_buffer),
        (1, &k_buffer),
        (2, &q_out_buffer),
        (3, &k_out_buffer),
        (4, &positions_buffer),
        (5, &time_buffer),
        (6, &height_buffer),
        (7, &width_buffer),
    ] {
        encoder.set_buffer(index, Some(value), 0);
    }
    let scalars = [tokens as u32, heads as u32, head_dim as u32, pairs as u32];
    for (index, value) in scalars.iter().enumerate() {
        encoder.set_bytes((8 + index) as u64, 4, (value as *const u32).cast());
    }
    let width = pipeline.thread_execution_width().max(1);
    encoder.dispatch_threads(
        MTLSize::new(tokens as u64, heads as u64, head_dim as u64),
        MTLSize::new(1, 1, width.min(head_dim as u64)),
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(Rope3dError::Metal(format!(
            "command buffer status {:?}",
            command.status()
        )));
    }
    unsafe {
        q_out.copy_from_slice(std::slice::from_raw_parts(
            q_out_buffer.contents().cast(),
            expected,
        ));
        k_out.copy_from_slice(std::slice::from_raw_parts(
            k_out_buffer.contents().cast(),
            expected,
        ));
    }
    Ok((q_out, k_out))
}
