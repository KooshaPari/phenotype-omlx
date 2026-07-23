//! Numeric precision policy used by an operator.
//!
//! `Precision` is the *policy* — what the kernel should logically treat
//! values as. The runtime dtype used for storage is a separate enum
//! ([`crate::dtype::DType`]) and is owned by individual tensor refs.
//!
//! `serde(rename_all = "snake_case")` keeps the JSON schema stable and
//! human-readable (e.g. `fp32`, `bf16`).

use serde::{Deserialize, Serialize};

/// Logical numeric precision for an operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    /// IEEE-754 single precision (32-bit float).
    Fp32,
    /// IEEE-754 half precision (16-bit float, 1 sign + 5 exponent + 10 mantissa).
    Fp16,
    /// Brain floating point (16-bit, 1 sign + 8 exponent + 7 mantissa).
    Bf16,
    /// Signed 8-bit integer.
    Int8,
    /// Unsigned 8-bit integer. Renamed explicitly because
    /// `rename_all = "snake_case"` would otherwise split `UInt8` as
    /// `u_int8`.
    #[serde(rename = "uint8")]
    UInt8,
}

impl Precision {
    /// Width of the precision in bytes.
    ///
    /// Returns 1 for the 8-bit variants and 2 / 4 for the 16 / 32-bit
    /// floating-point variants. This is the *nominal* storage width used
    /// for memory accounting in the plan; kernels that pack sub-byte
    /// values belong to [`crate::quantization::QuantizationPolicy`].
    pub fn bytes(&self) -> usize {
        match self {
            Precision::Fp32 => 4,
            Precision::Fp16 | Precision::Bf16 => 2,
            Precision::Int8 | Precision::UInt8 => 1,
        }
    }

    /// Lowercase string identifier; matches the JSON representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Precision::Fp32 => "fp32",
            Precision::Fp16 => "fp16",
            Precision::Bf16 => "bf16",
            Precision::Int8 => "int8",
            Precision::UInt8 => "uint8",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_for_each_variant() {
        assert_eq!(Precision::Fp32.bytes(), 4);
        assert_eq!(Precision::Fp16.bytes(), 2);
        assert_eq!(Precision::Bf16.bytes(), 2);
        assert_eq!(Precision::Int8.bytes(), 1);
        assert_eq!(Precision::UInt8.bytes(), 1);
    }

    #[test]
    fn as_str_matches_json() {
        for p in [
            Precision::Fp32,
            Precision::Fp16,
            Precision::Bf16,
            Precision::Int8,
            Precision::UInt8,
        ] {
            assert_eq!(
                serde_json::to_string(&p).unwrap(),
                format!("\"{}\"", p.as_str())
            );
        }
    }
}
