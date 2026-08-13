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

/// Repack the canonical host `[k, n]` stream into the Metal GEMM layout
/// `[n, ceil(k / 4)]`.
///
/// Host [`ternary_pack`] groups adjacent columns in each byte, while the
/// Metal shader groups adjacent K lanes for each output column. Keeping this
/// conversion explicit prevents callers from silently uploading a valid
/// buffer with the wrong interpretation.
pub fn ternary_repack_for_metal(packed: &[u8], k: usize, n: usize) -> Result<Vec<u8>> {
    if k == 0 {
        return Err(KernelError::ZeroDimension { what: "k", got: 0 });
    }
    if n == 0 {
        return Err(KernelError::ZeroDimension { what: "n", got: 0 });
    }
    let symbols = k
        .checked_mul(n)
        .ok_or(KernelError::DimensionOverflow { what: "k*n" })?;
    let host_bytes = symbols
        .checked_add(3)
        .ok_or(KernelError::DimensionOverflow {
            what: "host packed byte count",
        })?
        / 4;
    if packed.len() != host_bytes {
        return Err(KernelError::BadBufferLength {
            what: "packed",
            expected: host_bytes,
            got: packed.len(),
        });
    }
    let stride = k.div_ceil(4);
    let output_bytes = n
        .checked_mul(stride)
        .ok_or(KernelError::DimensionOverflow {
            what: "Metal packed byte count",
        })?;
    let mut output = vec![0u8; output_bytes];
    for row in 0..k {
        for col in 0..n {
            let host_index = row * n + col;
            let code = (packed[host_index / 4] >> ((host_index % 4) * 2)) & 0b11;
            let metal_index = col * stride + row / 4;
            output[metal_index] |= code << ((row % 4) * 2);
        }
    }
    Ok(output)
}

/// Unpack a 2-bit ternary bitstream. `n` is the number of logical
/// symbols (i.e. the original `values.len()`). The output buffer is
/// resized/overwritten in place by the caller. `scales` and `zeros`
/// must carry one metadata entry per logical group, even though this
/// symbolic decoder does not apply the affine transform; the fused
/// matmul path consumes those entries separately.
pub fn ternary_unpack(
    packed: &[u8],
    scales: &[f32],
    zeros: &[f32],
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
    let expected_groups = n.div_ceil(group_size);
    if scales.len() != expected_groups {
        return Err(KernelError::BadBufferLength {
            what: "scales",
            expected: expected_groups,
            got: scales.len(),
        });
    }
    if zeros.len() != expected_groups {
        return Err(KernelError::BadBufferLength {
            what: "zeros",
            expected: expected_groups,
            got: zeros.len(),
        });
    }
    for (index, &scale) in scales.iter().enumerate() {
        if !scale.is_finite() {
            return Err(KernelError::NonFiniteValue {
                what: "scales",
                index,
            });
        }
        if scale != 1.0 {
            return Err(KernelError::OutOfRange {
                what: "scales",
                min: 1.0,
                max: 1.0,
                got: scale,
            });
        }
    }
    for (index, &zero) in zeros.iter().enumerate() {
        if !zero.is_finite() {
            return Err(KernelError::NonFiniteValue {
                what: "zeros",
                index,
            });
        }
        if zero != 0.0 {
            return Err(KernelError::OutOfRange {
                what: "zeros",
                min: 0.0,
                max: 0.0,
                got: zero,
            });
        }
    }
    for i in 0..n {
        let byte = packed[i / 4];
        let bit_off = (i % 4) * 2;
        let bits = (byte >> bit_off) & 0b11;
        out[i] = SignedTernary::from_bits(bits);
    }
    Ok(())
}
