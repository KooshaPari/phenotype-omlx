//! Model backend contract used by evaluation runners.

/// A generated completion with backend-reported accounting and latency.
#[derive(Debug, Clone, PartialEq)]
pub struct Completion {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub latency_ms: f64,
}

/// Backwards-compatible type name retained for existing callers.
pub type BackendCompletion = Completion;

/// A backend-reported continuation likelihood with accounting and latency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Likelihood {
    pub log_probability: f64,
    pub token_count: usize,
    pub latency_ms: f64,
}

/// Typed backend failures preserved as sources on task-scoped evaluation errors.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum BackendError {
    #[error("backend unavailable: {message}")]
    Unavailable { message: String },
    #[error("invalid backend response: {message}")]
    InvalidResponse { message: String },
}

/// Synchronous interface required by the evaluation harness.
pub trait Backend {
    /// Generate a completion for a prompt.
    fn complete(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> std::result::Result<Completion, BackendError>;

    /// Score a continuation conditioned on a prompt.
    fn log_likelihood(
        &self,
        prompt: &str,
        continuation: &str,
    ) -> std::result::Result<Likelihood, BackendError>;
}
