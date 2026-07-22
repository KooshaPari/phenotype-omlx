//! Windowed causal attention for long video-token sequences.

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TemporalAttentionError {
    #[error("attention dimensions and window must be non-zero")]
    ZeroDimension,
    #[error("tensor length does not match tokens * heads * head_dim")]
    BadShape,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal temporal attention failed: {0}")]
    Metal(String),
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn temporal_window_attention_metal(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    window: usize,
    scale: f32,
    artifact: &crate::MetallibArtifact,
) -> Result<Vec<f32>, TemporalAttentionError> {
    use metal::{Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;

    if tokens == 0 || heads == 0 || head_dim == 0 || window == 0 {
        return Err(TemporalAttentionError::ZeroDimension);
    }
    let len = tokens
        .checked_mul(heads)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or(TemporalAttentionError::BadShape)?;
    if q.len() != len || k.len() != len || v.len() != len {
        return Err(TemporalAttentionError::BadShape);
    }
    let device = Device::system_default()
        .ok_or_else(|| TemporalAttentionError::Metal("no system Metal device".into()))?;
    let library = device
        .new_library_with_data(artifact.bytes())
        .map_err(TemporalAttentionError::Metal)?;
    let function = library
        .get_function("temporal_window_attention_f32", None)
        .map_err(TemporalAttentionError::Metal)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(TemporalAttentionError::Metal)?;
    let shared = MTLResourceOptions::StorageModeShared;
    let buffer = |data: &[f32]| {
        device.new_buffer_with_data(
            data.as_ptr().cast::<c_void>(),
            std::mem::size_of_val(data) as u64,
            shared,
        )
    };
    let q_buffer = buffer(q);
    let k_buffer = buffer(k);
    let v_buffer = buffer(v);
    let out_buffer = device.new_buffer((len * std::mem::size_of::<f32>()) as u64, shared);
    let queue = device.new_command_queue();
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    for (index, value) in [
        (0, &q_buffer),
        (1, &k_buffer),
        (2, &v_buffer),
        (3, &out_buffer),
    ] {
        encoder.set_buffer(index, Some(value), 0);
    }
    let dims = [tokens as u32, heads as u32, head_dim as u32, window as u32];
    for (index, value) in dims.iter().enumerate() {
        encoder.set_bytes((4 + index) as u64, 4, (value as *const u32).cast());
    }
    encoder.set_bytes(8, 4, (&scale as *const f32).cast());
    let width = pipeline.thread_execution_width().max(1);
    encoder.dispatch_threads(
        MTLSize::new(tokens as u64, heads as u64, head_dim as u64),
        MTLSize::new(1, 1, width.min(head_dim as u64)),
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(TemporalAttentionError::Metal(format!(
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
