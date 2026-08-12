//! Fused diffusion argmax and softmax-max confidence kernel.
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DiffusionConfidenceError {
    #[error("tokens and vocabulary must be non-zero")]
    ZeroDimension,
    #[error("logits length must equal tokens * vocabulary")]
    BadShape,
    #[error("logit at index {index} is not finite")]
    NonFiniteLogit { index: usize },
    #[error("dimension {dimension}={value} exceeds Metal uint range")]
    DimensionOutOfRange {
        dimension: &'static str,
        value: usize,
    },
    #[error("diffusion confidence output byte size overflows host address space")]
    OutputByteSizeOverflow,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal diffusion confidence failed: {0}")]
    Metal(String),
}

#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
struct DiffusionConfidenceLayout {
    logits_len: usize,
    ids_bytes: u64,
    confidence_bytes: u64,
    tokens_u32: u32,
    vocab_u32: u32,
}

#[cfg_attr(not(any(test, all(feature = "metal", target_os = "macos"))), allow(dead_code))]
fn validate_diffusion_confidence_layout(
    tokens: usize,
    vocab: usize,
    output_tokens: usize,
) -> Result<DiffusionConfidenceLayout, DiffusionConfidenceError> {
    if tokens == 0 || vocab == 0 {
        return Err(DiffusionConfidenceError::ZeroDimension);
    }
    let tokens_u32 =
        u32::try_from(tokens).map_err(|_| DiffusionConfidenceError::DimensionOutOfRange {
            dimension: "tokens",
            value: tokens,
        })?;
    let vocab_u32 =
        u32::try_from(vocab).map_err(|_| DiffusionConfidenceError::DimensionOutOfRange {
            dimension: "vocab",
            value: vocab,
        })?;
    let logits_len = tokens
        .checked_mul(vocab)
        .ok_or(DiffusionConfidenceError::BadShape)?;
    let ids_bytes = checked_output_bytes(output_tokens, std::mem::size_of::<u32>())?;
    let confidence_bytes = checked_output_bytes(output_tokens, std::mem::size_of::<f32>())?;

    Ok(DiffusionConfidenceLayout {
        logits_len,
        ids_bytes: ids_bytes as u64,
        confidence_bytes: confidence_bytes as u64,
        tokens_u32,
        vocab_u32,
    })
}

#[cfg_attr(not(any(test, all(feature = "metal", target_os = "macos"))), allow(dead_code))]
fn checked_output_bytes(
    output_tokens: usize,
    element_size: usize,
) -> Result<u64, DiffusionConfidenceError> {
    output_tokens
        .checked_mul(element_size)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(DiffusionConfidenceError::OutputByteSizeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_nonfinite_logits_with_their_index_before_device_work() {
        for (logits, index) in [
            (&[0.0, f32::NAN][..], 1),
            (&[f32::INFINITY, 0.0][..], 0),
            (&[0.0, f32::NEG_INFINITY][..], 1),
        ] {
            let err = validate_diffusion_confidence_inputs(logits, 1, 2).unwrap_err();

            assert_eq!(err, DiffusionConfidenceError::NonFiniteLogit { index });
        }
    }

    #[test]
    fn accepts_finite_logits_with_a_matching_shape() {
        assert_eq!(
            validate_diffusion_confidence_inputs(&[0.0, -1.0], 1, 2),
            Ok(())
        );
    }

    #[test]
    fn rejects_dimensions_and_output_sizes_before_metal_work() {
        let too_large_for_metal = u32::MAX as usize + 1;
        assert!(matches!(
            validate_diffusion_confidence_layout(too_large_for_metal, 1, too_large_for_metal),
            Err(DiffusionConfidenceError::DimensionOutOfRange {
                dimension: "tokens",
                value,
            })
            if value == too_large_for_metal
        ));

        let too_large_for_output = usize::MAX / std::mem::size_of::<u32>() + 1;
        assert!(matches!(
            checked_output_bytes(too_large_for_output, std::mem::size_of::<u32>()),
            Err(DiffusionConfidenceError::OutputByteSizeOverflow)
        ));
    }
}

/// Validate diffusion confidence inputs without acquiring a Metal pipeline or device resource.
#[cfg_attr(not(any(test, all(feature = "metal", target_os = "macos"))), allow(dead_code))]
fn validate_diffusion_confidence_inputs(
    logits: &[f32],
    tokens: usize,
    vocab: usize,
) -> Result<(), DiffusionConfidenceError> {
    let layout = validate_diffusion_confidence_layout(tokens, vocab, tokens)?;
    if logits.len() != layout.logits_len {
        return Err(DiffusionConfidenceError::BadShape);
    }
    if let Some(index) = logits.iter().position(|value| !value.is_finite()) {
        return Err(DiffusionConfidenceError::NonFiniteLogit { index });
    }
    Ok(())
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
    let layout = validate_diffusion_confidence_layout(tokens, vocab, tokens)?;
    if logits.len() != layout.logits_len {
        return Err(DiffusionConfidenceError::BadShape);
    }
    if let Some(index) = logits.iter().position(|value| !value.is_finite()) {
        return Err(DiffusionConfidenceError::NonFiniteLogit { index });
    }
    crate::metal_cache::with_catalogued_pipeline(artifact, "denoise", |device, queue, pipeline| {
        let shared = MTLResourceOptions::StorageModeShared;
        let input = device.new_buffer_with_data(
            logits.as_ptr().cast::<c_void>(),
            std::mem::size_of_val(logits) as u64,
            shared,
        );
        let ids = device.new_buffer(layout.ids_bytes, shared);
        let confidence = device.new_buffer(layout.confidence_bytes, shared);
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&input), 0);
        encoder.set_buffer(1, Some(&ids), 0);
        encoder.set_buffer(2, Some(&confidence), 0);
        encoder.set_bytes(3, 4, (&layout.tokens_u32 as *const u32).cast());
        encoder.set_bytes(4, 4, (&layout.vocab_u32 as *const u32).cast());
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
