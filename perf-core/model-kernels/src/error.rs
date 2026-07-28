//! Error types for `model-kernels`.
//!
//! All public fallible APIs return [`Result<T, KernelError>`]. Errors are
//! constructed with [`thiserror`] so call sites can match on structured
//! variants rather than parsing strings.

use thiserror::Error;

/// Errors that can be produced by any kernel in this crate.
#[derive(Debug, Error, PartialEq)]
pub enum KernelError {
    /// A dimension argument was zero where it must be strictly positive
    /// (e.g. `head_dim == 0`, `group_size == 0`, `chunk_size == 0`).
    #[error("dimension must be > 0: {what} = {got}")]
    ZeroDimension { what: &'static str, got: usize },

    /// Two dimensions that must agree did not.
    #[error("dimension mismatch: {what}: expected {expected}, got {got}")]
    DimMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },

    /// `q_heads` is not evenly divisible by `kv_heads` (GQA group size
    /// must be an integer).
    #[error("q_heads ({q_heads}) must be a positive multiple of kv_heads ({kv_heads})")]
    BadGqaGrouping { q_heads: usize, kv_heads: usize },

    /// A buffer passed to a kernel did not match the expected logical
    /// shape derived from the dimension arguments.
    #[error("buffer length {got} does not match expected {expected} for {what}")]
    BadBufferLength {
        what: &'static str,
        expected: usize,
        got: usize,
    },

    /// A bit-width outside the supported range (sub-byte quantization
    /// only accepts `1..=8`).
    #[error("bits {bits} outside supported range 1..=8")]
    BitsOutOfRange { bits: u8 },

    /// A capacity-factor argument was non-positive (would produce a
    /// zero-or-negative per-expert bucket).
    #[error("capacity_factor must be > 0, got {got}")]
    BadCapacityFactor { got: f32 },

    /// A routing or kernel input contained NaN or infinity.  Rejecting these
    /// before sorting/softmax keeps the scalar router's contract aligned with
    /// the Metal facade and prevents NaN weights from reaching expert GEMMs.
    #[error("non-finite value in {what} at index {index}")]
    NonFiniteValue { what: &'static str, index: usize },

    /// A row / column / index referenced inside a MoE dispatch plan was
    /// out of range.
    #[error("expert index {got} outside [0, {num_experts})")]
    ExpertOutOfRange { num_experts: usize, got: usize },

    /// A query or key sequence length was zero where the kernel
    /// requires at least one element.
    #[error("sequence length must be > 0: {what} = 0")]
    EmptySequence { what: &'static str },

    /// A scalar argument was outside its allowed interval
    /// `[min, max]`.
    #[error("{what} = {got} outside [{min}, {max}]")]
    OutOfRange {
        what: &'static str,
        min: f32,
        max: f32,
        got: f32,
    },
}

/// Result alias for `model-kernels`.
pub type Result<T> = std::result::Result<T, KernelError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_dimension_renders() {
        let e = KernelError::ZeroDimension {
            what: "head_dim",
            got: 0,
        };
        assert!(e.to_string().contains("head_dim"));
        assert!(e.to_string().contains("0"));
    }

    #[test]
    fn bad_bits_renders() {
        let e = KernelError::BitsOutOfRange { bits: 0 };
        assert!(e.to_string().contains("0"));
    }

    #[test]
    fn bad_buffer_length_renders() {
        let e = KernelError::BadBufferLength {
            what: "q",
            expected: 16,
            got: 8,
        };
        assert!(e.to_string().contains("q"));
        assert!(e.to_string().contains("16"));
        assert!(e.to_string().contains("8"));
    }
}
