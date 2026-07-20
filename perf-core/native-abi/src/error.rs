//! Error type returned to safe Rust callers.

use thiserror::Error;

use crate::status::Status;

/// Wraps a [`Status`] code with an optional detail message produced by the
/// dispatcher. Surfaced to safe callers; the C ABI itself uses raw `i32`
/// status codes.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[error("native ABI error: {kind:?} ({kind}) — {message}")]
pub struct NativeAbiError {
    pub kind: Status,
    pub message: String,
}

impl NativeAbiError {
    pub fn new(kind: Status, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}