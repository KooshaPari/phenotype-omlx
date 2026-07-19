//! Model backend contract used by evaluation runners.

use crate::Result;

/// A generated completion with backend-reported latency.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendCompletion {
    pub text: String,
    pub latency_ms: f64,
}

/// Minimal synchronous interface required by the evaluation harness.
pub trait Backend {
    /// Generate a completion for a prompt.
    fn complete(&self, prompt: &str, max_tokens: usize) -> Result<BackendCompletion>;

    /// Score a continuation conditioned on a prompt.
    fn log_likelihood(&self, prompt: &str, continuation: &str) -> Result<f64>;
}
