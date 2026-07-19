//! Typed errors emitted by loaders, backends, and runners.

use crate::Suite;

/// All errors produced by the eval harness.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("io error reading {path}: {source}")]
    Io { path: String, #[source] source: std::io::Error },
    #[error("json error in {path} at line {line}: {source}")]
    Json { path: String, line: usize, #[source] source: serde_json::Error },
    #[error("yaml error in {path} at line {line}: {source}")]
    Yaml { path: String, line: usize, #[source] source: serde_yaml::Error },
    #[error("csv format error in {path} at line {line}: {message}")]
    Csv { path: String, line: usize, message: String },
    #[error("missing required field '{field}' in {path}")]
    MissingField { path: String, field: &'static str },
    #[error("malformed record in {path} at line {line}: {message}")]
    Malformed { path: String, line: usize, message: String },
    #[error("inconsistent suite in {path}: expected {expected:?}, got {actual:?}")]
    SuiteMismatch { path: String, expected: Suite, actual: Suite },
    #[error("backend error: {message}")]
    Backend { message: String },
    #[error("invalid task {task_id}: {message}")]
    InvalidTask { task_id: String, message: String },
}

impl EvalError {
    pub fn io(path: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io { path: path.into(), source }
    }

    pub fn json(path: impl Into<String>, source: serde_json::Error) -> Self {
        Self::Json { path: path.into(), line: source.line(), source }
    }

    pub fn json_at_line(path: impl Into<String>, line: usize, source: serde_json::Error) -> Self {
        Self::Json { path: path.into(), line, source }
    }

    pub fn yaml(path: impl Into<String>, source: serde_yaml::Error) -> Self {
        let line = source.location().map(|location| location.line()).unwrap_or(1);
        Self::Yaml { path: path.into(), line, source }
    }

    pub fn csv(path: impl Into<String>, line: usize, message: impl Into<String>) -> Self {
        Self::Csv { path: path.into(), line, message: message.into() }
    }

    pub fn missing_field(path: impl Into<String>, field: &'static str) -> Self {
        Self::MissingField { path: path.into(), field }
    }

    pub fn malformed(path: impl Into<String>, line: usize, message: impl Into<String>) -> Self {
        Self::Malformed { path: path.into(), line, message: message.into() }
    }

    pub fn backend(message: impl Into<String>) -> Self {
        Self::Backend { message: message.into() }
    }

    pub fn invalid_task(task_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self::InvalidTask { task_id: task_id.into(), message: message.into() }
    }
}

pub type Result<T> = std::result::Result<T, EvalError>;
