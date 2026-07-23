//! Joint attention primitive for Flux/SD3-style image+text streams.

use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum JointAttentionError {
    #[error("attention dimensions must be non-zero")]
    ZeroDimension,
    #[error("tensor length does not match the declared shape")]
    BadShape,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal joint attention failed: {0}")]
    Metal(String),
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn joint_attention_metal(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_tokens: usize,
    kv_tokens: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
    artifact: &crate::MetallibArtifact,
) -> Result<Vec<f32>, JointAttentionError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;

    if q_tokens == 0 || kv_tokens == 0 || heads == 0 || head_dim == 0 {
        return Err(JointAttentionError::ZeroDimension);
    }
    let q_len = q_tokens
        .checked_mul(heads)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or(JointAttentionError::BadShape)?;
    let kv_len = kv_tokens
        .checked_mul(heads)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or(JointAttentionError::BadShape)?;
    if q.len() != q_len || k.len() != kv_len || v.len() != kv_len {
        return Err(JointAttentionError::BadShape);
    }
    crate::metal_cache::with_pipeline(
        artifact,
        "joint_attention_f32",
        |device, queue, pipeline| {
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
            let out_len = q_len;
            let out_buffer =
                device.new_buffer((out_len * std::mem::size_of::<f32>()) as u64, shared);
            let command = queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            for (index, value) in [
                (0, &q_buffer),
                (1, &k_buffer),
                (2, &v_buffer),
                (3, &out_buffer),
            ] {
                encoder.set_buffer(index, Some(value), 0);
            }
            let dims = [
                q_tokens as u32,
                kv_tokens as u32,
                heads as u32,
                head_dim as u32,
            ];
            for (index, value) in dims.iter().enumerate() {
                encoder.set_bytes((4 + index) as u64, 4, (value as *const u32).cast());
            }
            encoder.set_bytes(8, 4, (&scale as *const f32).cast());
            let width = pipeline.thread_execution_width().max(1);
            encoder.dispatch_threads(
                MTLSize::new(q_tokens as u64, heads as u64, head_dim as u64),
                MTLSize::new(1, 1, width.min(head_dim as u64)),
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(format!("command buffer status {:?}", command.status()));
            }
            let mut output = vec![0.0; out_len];
            unsafe {
                output.copy_from_slice(std::slice::from_raw_parts(
                    out_buffer.contents().cast(),
                    out_len,
                ));
            }
            Ok(output)
        },
    )
    .map_err(JointAttentionError::Metal)
}
