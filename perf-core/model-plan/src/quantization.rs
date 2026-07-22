//! Quantization policy applied to an operator's weights and activations.
//!
//! This is a *policy* (a description of the layout), not a packed buffer.
//! Concrete pack/unpack kernels live in the `turbo-quant*` family; the
//! plan simply declares what the layout is so the kernel selector and the
//! reference interpreter can stay aligned.

use serde::{Deserialize, Serialize};

/// Quantization policy for an operator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "scheme")]
pub enum QuantizationPolicy {
    /// No quantization: full-precision dense storage.
    Dense,

    /// Ternary (`{-1, 0, +1}`) weights packed into `bits`-wide words with
    /// `group_size` elements per scale factor. `bits` is fixed at `2` for
    /// the canonical Bonsai layout.
    Ternary {
        /// Number of elements packed into one scale group. Must be > 0.
        group_size: usize,
        /// Bits per weight. Canonical Bonsai value is `2`.
        bits: u8,
    },

    /// Sub-byte quantization with `bits` bits per value (e.g. 4-bit). The
    /// concrete pack layout is left to the implementing kernel.
    SubByte {
        /// Bits per quantized element. Must be in `1..=8`.
        bits: u8,
    },

    /// Symmetric per-channel scaling around zero (no zero-point).
    Symmetric,

    /// Affine (asymmetric) per-channel scaling with a zero-point offset.
    Affine,
}

impl QuantizationPolicy {
    /// True for policies that reduce memory below the dense baseline.
    pub fn is_compressed(&self) -> bool {
        matches!(
            self,
            QuantizationPolicy::Ternary { .. } | QuantizationPolicy::SubByte { .. }
        )
    }

    /// Validate the policy. Returns `Ok(())` for valid policies and
    /// `Err(reason)` for structural problems. Pure: never touches I/O.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            QuantizationPolicy::Dense => Ok(()),
            QuantizationPolicy::Symmetric | QuantizationPolicy::Affine => Ok(()),
            QuantizationPolicy::Ternary { group_size, bits } => {
                if *group_size == 0 {
                    return Err("ternary group_size must be > 0".to_string());
                }
                if *bits != 2 {
                    return Err(format!(
                        "ternary bits must be 2 (Bonsai contract), got {}",
                        bits
                    ));
                }
                Ok(())
            }
            QuantizationPolicy::SubByte { bits } => {
                if *bits == 0 || *bits > 8 {
                    return Err(format!("sub-byte bits must be in 1..=8, got {}", bits));
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_validates() {
        assert!(QuantizationPolicy::Dense.validate().is_ok());
    }

    #[test]
    fn ternary_rejects_zero_group_size() {
        let p = QuantizationPolicy::Ternary {
            group_size: 0,
            bits: 2,
        };
        let err = p.validate().unwrap_err();
        assert!(err.contains("group_size"));
    }

    #[test]
    fn ternary_rejects_non_two_bits() {
        let p = QuantizationPolicy::Ternary {
            group_size: 32,
            bits: 3,
        };
        let err = p.validate().unwrap_err();
        assert!(err.contains("bits"));
    }

    #[test]
    fn ternary_accepts_canonical_bonsai_layout() {
        let p = QuantizationPolicy::Ternary {
            group_size: 32,
            bits: 2,
        };
        assert!(p.validate().is_ok());
    }

    #[test]
    fn sub_byte_rejects_zero_bits() {
        let p = QuantizationPolicy::SubByte { bits: 0 };
        assert!(p.validate().is_err());
    }

    #[test]
    fn sub_byte_rejects_too_many_bits() {
        let p = QuantizationPolicy::SubByte { bits: 9 };
        assert!(p.validate().is_err());
    }

    #[test]
    fn symmetric_and_affine_are_valid() {
        assert!(QuantizationPolicy::Symmetric.validate().is_ok());
        assert!(QuantizationPolicy::Affine.validate().is_ok());
    }

    #[test]
    fn is_compressed_only_for_ternary_and_subbyte() {
        assert!(!QuantizationPolicy::Dense.is_compressed());
        assert!(!QuantizationPolicy::Symmetric.is_compressed());
        assert!(!QuantizationPolicy::Affine.is_compressed());
        assert!(QuantizationPolicy::Ternary {
            group_size: 32,
            bits: 2
        }
        .is_compressed());
        assert!(QuantizationPolicy::SubByte { bits: 4 }.is_compressed());
    }

    #[test]
    fn serde_round_trip() {
        let variants = vec![
            QuantizationPolicy::Dense,
            QuantizationPolicy::Ternary {
                group_size: 32,
                bits: 2,
            },
            QuantizationPolicy::SubByte { bits: 4 },
            QuantizationPolicy::Symmetric,
            QuantizationPolicy::Affine,
        ];
        for v in variants {
            let s = serde_json::to_string(&v).unwrap();
            let back: QuantizationPolicy = serde_json::from_str(&s).unwrap();
            assert_eq!(back, v);
        }
    }
}