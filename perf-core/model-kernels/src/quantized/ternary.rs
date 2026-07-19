//! Sign-magnitude ternary pack/unpack (`SignedTernary`, `ternary_pack`,
//! `ternary_unpack`).
//!
//! See the parent module docs for the symmetric per-group quantization
//! scheme and the location of the shared `min_max` helper.

use crate::error::{KernelError, Result};

/// Sign-magnitude ternary code. Three values per symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SignedTernary {
    /// Exact zero.
    Zero = 0b00,
    /// Positive unit (`+1` after dequant).
    Pos = 0b01,
    /// Negative unit (`-1` after dequant).
    Neg = 0b10,
}

impl SignedTernary {
    fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            0b01 => SignedTernary::Pos,
            0b10 => SignedTernary::Neg,
            _ => SignedTernary::Zero,
        }
    }
    fn bits(self) -> u8 {
        self as u8
    }
}

/// Pack a sequence of sign-magnitude ternary values into 2-bit blocks.
///
/// `values.len()` must be a multiple of `group_size` (a trailing partial
/// group is allowed). The returned `scales` and `zeros` have one entry
/// per group; for ternary the scale is fixed at `1.0` and the zero at
/// `0.0`.
pub fn ternary_pack(
    values: &[SignedTernary],
    group_size: usize,
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    if group_size == 0 {
        return Err(KernelError::ZeroDimension {
            what: "group_size",
            got: 0,
        });
    }
    let n = values.len();
    let num_full = n / group_size;
    let rem = n % group_size;
    let num_groups = num_full + if rem > 0 { 1 } else { 0 };
    let mut packed = vec![0u8; n.div_ceil(4)];
    let scales = vec![1.0f32; num_groups];
    let zeros = vec![0.0f32; num_groups];
    for (i, &v) in values.iter().enumerate() {
        let byte_idx = i / 4;
        let bit_off = (i % 4) * 2;
        packed[byte_idx] |= v.bits() << bit_off;
    }
    Ok((packed, scales, zeros))
}

/// Unpack a 2-bit ternary bitstream. `n` is the number of logical
/// symbols (i.e. the original `values.len()`). The output buffer is
/// resized/overwritten in place by the caller.
pub fn ternary_unpack(
    packed: &[u8],
    _scales: &[f32],
    _zeros: &[f32],
    n: usize,
    group_size: usize,
    out: &mut [SignedTernary],
) -> Result<()> {
    if group_size == 0 {
        return Err(KernelError::ZeroDimension {
            what: "group_size",
            got: 0,
        });
    }
    if out.len() < n {
        return Err(KernelError::BadBufferLength {
            what: "out",
            expected: n,
            got: out.len(),
        });
    }
    let expected_bytes = n.div_ceil(4);
    if packed.len() < expected_bytes {
        return Err(KernelError::BadBufferLength {
            what: "packed",
            expected: expected_bytes,
            got: packed.len(),
        });
    }
    for i in 0..n {
        let byte = packed[i / 4];
        let bit_off = (i % 4) * 2;
        let bits = (byte >> bit_off) & 0b11;
        out[i] = SignedTernary::from_bits(bits);
    }
    Ok(())
}
