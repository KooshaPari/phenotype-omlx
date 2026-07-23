//! Model backend contract used by evaluation runners.

/// Per-token log-probability detail returned by [`Backend::complete_with_logprobs`].
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    /// The token string.
    pub token: String,
    /// Log-probability assigned to this token.
    pub logprob: f64,
    /// Raw bytes of the token, if available from the backend.
    pub bytes: Option<Vec<u8>>,
}

/// Log-probability result for a single candidate choice.
#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceLogprobs {
    /// The choice text that was scored.
    pub choice: String,
    /// Aggregate log-probability for this choice.
    pub logprob: f64,
    /// Per-token log-probabilities composing this choice.
    pub tokens: Vec<TokenLogprob>,
}

/// Aggregated log-probability result across all candidate choices.
#[derive(Debug, Clone, PartialEq)]
pub struct LogprobResult {
    /// Per-choice log-probability details, one per entry in the `choices`
    /// slice passed to [`Backend::complete_with_logprobs`].
    pub choices: Vec<ChoiceLogprobs>,
}

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

    /// Return per-token log-probabilities for each candidate choice.
    ///
    /// The default implementation returns
    /// [`BackendError::Unavailable`]; backends that can supply token-level
    /// log-probabilities should override this method.
    fn complete_with_logprobs(
        &self,
        _prompt: &str,
        _choices: &[String],
    ) -> std::result::Result<LogprobResult, BackendError> {
        Err(BackendError::Unavailable {
            message: "complete_with_logprobs not implemented by this backend".into(),
        })
    }
}
