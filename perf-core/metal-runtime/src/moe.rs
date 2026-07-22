//! Fused Qwen-style mixture-of-experts top-k routing.

use model_kernels::moe::router_topk;
use thiserror::Error;

/// Logical shape of row-major router logits `[tokens, experts]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeShape {
    pub tokens: usize,
    pub experts: usize,
}

/// Selected expert IDs and selected-softmax weights, both `[tokens, top_k]`.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeRouterOutput {
    pub expert_ids: Vec<u32>,
    pub weights: Vec<f32>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MoeRouterError {
    #[error("{dimension} must be greater than zero")]
    ZeroDimension { dimension: &'static str },
    #[error("experts must be <= 256, got {experts}")]
    TooManyExperts { experts: usize },
    #[error("top_k must be one of 1, 2, 4, or 8, got {top_k}")]
    UnsupportedTopK { top_k: usize },
    #[error("top_k {top_k} exceeds experts {experts}")]
    TopKExceedsExperts { top_k: usize, experts: usize },
    #[error("router logits length must be {expected}, got {got}")]
    BadLogitLength { expected: usize, got: usize },
    #[error("router logit at index {index} is not finite")]
    NonFiniteLogit { index: usize },
    #[error("router shape element count overflowed usize")]
    ShapeOverflow,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("Metal router failed: {0}")]
    Metal(String),
}

/// Validated fused top-k router configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeRouter {
    shape: MoeShape,
    top_k: usize,
}

/// Execute assignment-list grouped GEMM on Metal.
///
/// The output is assignment-major `[assignments, n]`; capacity and dropped
/// token policy stay in the dispatch layer. `expert_weights` is expert-major
/// `[experts, k, n]`, matching the standalone `moe_grouped_gemm_f32` shader.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn grouped_gemm_metal(
    activations: &[f32],
    expert_weights: &[f32],
    assignment_tokens: &[u32],
    assignment_experts: &[u32],
    k: usize,
    n: usize,
    artifact: &crate::MetallibArtifact,
) -> Result<Vec<f32>, MoeRouterError> {
    use metal::{Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
    use std::ffi::c_void;

    if k == 0 || n == 0 {
        return Err(MoeRouterError::Metal("k and n must be non-zero".into()));
    }
    if assignment_tokens.len() != assignment_experts.len() {
        return Err(MoeRouterError::Metal("assignment arrays have different lengths".into()));
    }
    if activations.len() % k != 0 || expert_weights.len() % (k * n) != 0 {
        return Err(MoeRouterError::Metal("input shape does not match k/n".into()));
    }
    let assignments = assignment_tokens.len();
    let device = Device::system_default()
        .ok_or_else(|| MoeRouterError::Metal("no system Metal device".into()))?;
    let library = device
        .new_library_with_data(artifact.bytes())
        .map_err(MoeRouterError::Metal)?;
    let function = library
        .get_function("moe_grouped_gemm_f32", None)
        .map_err(MoeRouterError::Metal)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(MoeRouterError::Metal)?;
    let shared = MTLResourceOptions::StorageModeShared;
    let input = device.new_buffer_with_data(
        activations.as_ptr().cast::<c_void>(),
        std::mem::size_of_val(activations) as u64,
        shared,
    );
    let weights = device.new_buffer_with_data(
        expert_weights.as_ptr().cast::<c_void>(),
        std::mem::size_of_val(expert_weights) as u64,
        shared,
    );
    let token_ids = device.new_buffer_with_data(
        assignment_tokens.as_ptr().cast::<c_void>(),
        std::mem::size_of_val(assignment_tokens) as u64,
        shared,
    );
    let expert_ids = device.new_buffer_with_data(
        assignment_experts.as_ptr().cast::<c_void>(),
        std::mem::size_of_val(assignment_experts) as u64,
        shared,
    );
    let mut output = vec![0.0f32; assignments * n];
    let output_buffer = device.new_buffer(
        (output.len() * std::mem::size_of::<f32>()) as u64,
        shared,
    );
    let assignments_u32 = assignments as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;
    let queue = device.new_command_queue();
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(&weights), 0);
    encoder.set_buffer(2, Some(&token_ids), 0);
    encoder.set_buffer(3, Some(&expert_ids), 0);
    encoder.set_buffer(4, Some(&output_buffer), 0);
    for (index, value) in [(5, &assignments_u32), (6, &k_u32), (7, &n_u32)] {
        encoder.set_bytes(index, std::mem::size_of::<u32>() as u64, (value as *const u32).cast());
    }
    encoder.dispatch_threads(
        MTLSize::new(assignments as u64, n as u64, 1),
        MTLSize::new(
            pipeline.thread_execution_width().min(assignments as u64),
            1,
            1,
        ),
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(MoeRouterError::Metal(format!(
            "command buffer completed with status {:?}",
            command.status()
        )));
    }
    let output_len = output.len();
    unsafe {
        output.copy_from_slice(std::slice::from_raw_parts(
            output_buffer.contents().cast::<f32>(),
            output_len,
        ));
    }
    Ok(output)
}

impl MoeRouter {
    pub fn new(shape: MoeShape, top_k: usize) -> Result<Self, MoeRouterError> {
        if shape.tokens == 0 {
            return Err(MoeRouterError::ZeroDimension {
                dimension: "tokens",
            });
        }
        if shape.experts == 0 {
            return Err(MoeRouterError::ZeroDimension {
                dimension: "experts",
            });
        }
        // Covers current small MoE++ (16), Qwen sparse MoE (128), and
        // DeepSeek-style grouped routing (256) without changing the Metal
        // ABI: expert count is still a runtime scalar.
        if shape.experts > 256 {
            return Err(MoeRouterError::TooManyExperts {
                experts: shape.experts,
            });
        }
        if !matches!(top_k, 1 | 2 | 4 | 8) {
            return Err(MoeRouterError::UnsupportedTopK { top_k });
        }
        if top_k > shape.experts {
            return Err(MoeRouterError::TopKExceedsExperts {
                top_k,
                experts: shape.experts,
            });
        }
        shape
            .tokens
            .checked_mul(shape.experts)
            .ok_or(MoeRouterError::ShapeOverflow)?;
        Ok(Self { shape, top_k })
    }

    fn validate_logits(&self, logits: &[f32]) -> Result<(), MoeRouterError> {
        let expected = self
            .shape
            .tokens
            .checked_mul(self.shape.experts)
            .ok_or(MoeRouterError::ShapeOverflow)?;
        if logits.len() != expected {
            return Err(MoeRouterError::BadLogitLength {
                expected,
                got: logits.len(),
            });
        }
        if let Some(index) = logits.iter().position(|value| !value.is_finite()) {
            return Err(MoeRouterError::NonFiniteLogit { index });
        }
        Ok(())
    }

    /// Scalar correctness path using the model-kernels router oracle.
    pub fn route_reference(&self, logits: &[f32]) -> Result<MoeRouterOutput, MoeRouterError> {
        self.validate_logits(logits)?;
        let output_len = self.shape.tokens * self.top_k;
        let mut expert_ids = Vec::with_capacity(output_len);
        let mut weights = Vec::with_capacity(output_len);
        for row in logits.chunks_exact(self.shape.experts) {
            let selected = router_topk(row, self.shape.experts, self.top_k, 0)
                .expect("validated MoE router dimensions");
            expert_ids.extend(selected.iter().map(|(expert, _)| *expert as u32));
            weights.extend(selected.into_iter().map(|(_, weight)| weight));
        }
        Ok(MoeRouterOutput {
            expert_ids,
            weights,
        })
    }

    /// Encode and execute the fused router on the system Metal device.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn route_metal(
        &self,
        logits: &[f32],
        artifact: &crate::MetallibArtifact,
    ) -> Result<MoeRouterOutput, MoeRouterError> {
        use metal::{Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};
        use std::ffi::c_void;

        self.validate_logits(logits)?;
        let token_count = u64::try_from(self.shape.tokens)
            .map_err(|_| MoeRouterError::Metal("token count exceeds Metal NSUInteger".into()))?;
        let device = Device::system_default()
            .ok_or_else(|| MoeRouterError::Metal("no system Metal device".into()))?;
        let library = device
            .new_library_with_data(artifact.bytes())
            .map_err(MoeRouterError::Metal)?;
        let function = library
            .get_function("moe_topk_f32", None)
            .map_err(MoeRouterError::Metal)?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(MoeRouterError::Metal)?;
        let shared = MTLResourceOptions::StorageModeShared;
        let input = device.new_buffer_with_data(
            logits.as_ptr().cast::<c_void>(),
            std::mem::size_of_val(logits) as u64,
            shared,
        );
        let output_len = self.shape.tokens * self.top_k;
        let ids = device.new_buffer((output_len * std::mem::size_of::<u32>()) as u64, shared);
        let weights = device.new_buffer((output_len * std::mem::size_of::<f32>()) as u64, shared);
        let experts = self.shape.experts as u32;
        let top_k = self.top_k as u32;
        let queue = device.new_command_queue();
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input), 0);
        encoder.set_buffer(1, Some(&ids), 0);
        encoder.set_buffer(2, Some(&weights), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            (&experts as *const u32).cast(),
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            (&top_k as *const u32).cast(),
        );
        encoder.dispatch_threads(
            MTLSize::new(token_count, 1, 1),
            MTLSize::new(pipeline.thread_execution_width().min(token_count), 1, 1),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(MoeRouterError::Metal(format!(
                "command buffer completed with status {:?}",
                command.status()
            )));
        }

        // StorageModeShared buffers are CPU-visible after command completion.
        let expert_ids = unsafe {
            std::slice::from_raw_parts(ids.contents().cast::<u32>(), output_len).to_vec()
        };
        let weights = unsafe {
            std::slice::from_raw_parts(weights.contents().cast::<f32>(), output_len).to_vec()
        };
        Ok(MoeRouterOutput {
            expert_ids,
            weights,
        })
    }
}
