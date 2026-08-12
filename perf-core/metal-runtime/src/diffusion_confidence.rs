//! Fused diffusion argmax and softmax-max confidence kernel.
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DiffusionConfidenceError {
    #[error("tokens and vocabulary must be non-zero")]
    ZeroDimension,
    #[error("logits length must equal tokens * vocabulary")]
    BadShape,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal diffusion confidence failed: {0}")]
    Metal(String),
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn diffusion_argmax_confidence_metal(
    logits: &[f32],
    tokens: usize,
    vocab: usize,
    artifact: &crate::MetallibArtifact,
) -> Result<(Vec<u32>, Vec<f32>), DiffusionConfidenceError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;
    if tokens == 0 || vocab == 0 {
        return Err(DiffusionConfidenceError::ZeroDimension);
    }
    let len = tokens
        .checked_mul(vocab)
        .ok_or(DiffusionConfidenceError::BadShape)?;
    if logits.len() != len {
        return Err(DiffusionConfidenceError::BadShape);
    }
    crate::metal_cache::with_catalogued_pipeline(artifact, "denoise", |device, queue, pipeline| {
        let shared = MTLResourceOptions::StorageModeShared;
        let input = device.new_buffer_with_data(
            logits.as_ptr().cast::<c_void>(),
            std::mem::size_of_val(logits) as u64,
            shared,
        );
        let ids = device.new_buffer((tokens * std::mem::size_of::<u32>()) as u64, shared);
        let confidence = device.new_buffer((tokens * std::mem::size_of::<f32>()) as u64, shared);
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&input), 0);
        encoder.set_buffer(1, Some(&ids), 0);
        encoder.set_buffer(2, Some(&confidence), 0);
        let tokens_u32 = tokens as u32;
        let vocab_u32 = vocab as u32;
        encoder.set_bytes(3, 4, (&tokens_u32 as *const u32).cast());
        encoder.set_bytes(4, 4, (&vocab_u32 as *const u32).cast());
        let width = pipeline.thread_execution_width().max(1);
        encoder.dispatch_threads(
            MTLSize::new(tokens as u64, 1, 1),
            MTLSize::new(width.min(tokens as u64), 1, 1),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(format!("command buffer status {:?}", command.status()));
        }
        let mut out_ids = vec![0u32; tokens];
        let mut out_confidence = vec![0.0f32; tokens];
        unsafe {
            out_ids.copy_from_slice(std::slice::from_raw_parts(ids.contents().cast(), tokens));
            out_confidence.copy_from_slice(std::slice::from_raw_parts(
                confidence.contents().cast(),
                tokens,
            ));
        }
        Ok((out_ids, out_confidence))
    })
    .map_err(DiffusionConfidenceError::Metal)
}
