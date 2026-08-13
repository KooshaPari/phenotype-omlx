//! Backend traits — engine-agnostic interfaces for target and draft models.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single forward-pass output from the target model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetOutput {
    /// Logits for the next token: `[vocab_size]`.
    pub logits: Vec<f32>,
    /// Hidden states (optional, for Medusa heads): `[hidden_dim]`.
    #[serde(default)]
    pub hidden: Option<Vec<f32>>,
    /// Whether the EOS token was produced.
    pub finished: bool,
}

/// Metadata describing a loaded backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    pub engine: String,
    pub model_id: String,
    pub device: String,
    pub dtype: String,
    pub kv_cache_type: Option<String>,
}

/// Trait implemented by every target-model backend (MLX, Metal, CUDA,
/// vLLM, SGLang, llama.cpp, custom).
#[allow(
    clippy::double_must_use,
    reason = "async_trait generates a Future that is already #[must_use]"
)]
#[async_trait]
pub trait TargetBackend: Send + Sync {
    /// Run a forward pass over `token_ids` and return next-token logits.
    async fn forward(&self, token_ids: &[u32]) -> Result<TargetOutput, String>;

    /// Batched tree verification: run one forward pass over a set of
    /// candidate continuations and return acceptance masks.
    ///
    /// `candidates[i]` is a candidate token sequence; the returned vector
    /// has one bool per candidate indicating acceptance.
    async fn verify_tree(
        &self,
        prefix: &[u32],
        candidates: &[Vec<u32>],
    ) -> Result<Vec<bool>, String> {
        // Default: sequential verification (backends with tree attention override).
        let mut accepted = Vec::with_capacity(candidates.len());
        for cand in candidates {
            let mut seq = prefix.to_vec();
            seq.extend_from_slice(cand);
            let out = self.forward(&seq).await?;
            accepted.push(out.finished);
        }
        Ok(accepted)
    }

    fn info(&self) -> BackendInfo;
}

/// Trait implemented by draft-model backends.
#[allow(
    clippy::double_must_use,
    reason = "async_trait generates a Future that is already #[must_use]"
)]
#[async_trait]
pub trait DraftBackend: Send + Sync {
    /// Propose up to `max_tokens` continuation tokens after `prefix`.
    async fn draft(&self, prefix: &[u32], max_tokens: usize) -> Result<Vec<u32>, String>;

    fn info(&self) -> BackendInfo;
}

/// A no-op draft backend used for SameModel (prompt-lookup) mode — the
/// engine derives drafts from the KV cache, so no model is needed.
pub struct NullDraftBackend;

#[async_trait]
impl DraftBackend for NullDraftBackend {
    async fn draft(&self, _prefix: &[u32], _max_tokens: usize) -> Result<Vec<u32>, String> {
        Ok(Vec::new())
    }
    fn info(&self) -> BackendInfo {
        BackendInfo {
            engine: "null".into(),
            model_id: "prompt-lookup".into(),
            device: "n/a".into(),
            dtype: "n/a".into(),
            kv_cache_type: None,
        }
    }
}

/// Registry of available backends, keyed by engine name.
/// Used by the FFI layer to instantiate the right backend at runtime.
pub fn known_engines() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();
    m.insert("mlx", "Apple MLX (Metal) — default on Apple Silicon");
    m.insert("metal", "Raw Metal compute shaders");
    m.insert("cuda", "NVIDIA CUDA (Linux/Windows)");
    m.insert("mps", "PyTorch MPS (Apple Silicon)");
    m.insert("vllm", "vLLM server (remote)");
    m.insert("sglang", "SGLang server (remote, primary GPU path)");
    m.insert("tensorrt", "TensorRT-LLM server (remote)");
    m.insert("llama_cpp", "llama.cpp server (remote)");
    m.insert("custom", "User-provided custom engine");
    m
}
