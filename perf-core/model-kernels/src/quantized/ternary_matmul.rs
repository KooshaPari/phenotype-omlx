//! Bonsai-style fused ternary matmul.
//!
//! Computes `out[m, n] = a[m, :] @ b[:, :]` where `b` is stored as a
//! 2-bit-per-element ternary bitstream with per-group `(scale, zero)`.
//! Ternary is the Bonsai quantization scheme: every element is
//! one of `{Zero, +1, -1}` and the per-group scale is fixed at `1.0`
//! with `zero = 0.0`. The packed layout matches the one emitted by
//! [`crate::quantized::ternary_pack`].
//!
//! # Layout
//!
//! - `a` is `[m, k]` row-major.
//! - `b_packed` is the concatenation of `k` rows of length `n`, each
//!   row flattened and 2-bit-packed per the [`ternary_pack`] rule:
//!   four symbols per byte, low-to-high bit order.
//! - `b_scales`, `b_zeros` have one entry per group of
//!   `group_size` symbols (where a "symbol" is one position in the
//!   flattened `[k * n]` sequence). For pure Bonsai, both are
//!   constant (`1.0` and `0.0` respectively) and may be ignored.
//! - `out` is `[m, n]` row-major.
//!
//! # Why "fused"
//!
//! The naive ternary-matmul path unpacks every weight row to f32,
//! then runs a dense f32 matmul. This kernel walks the packed
//! bitstream *directly* during the inner-product accumulation, so
//! it never materializes the full `[k, n]` dequantized weight
//! matrix. It is the scalar oracle that future tile-friendly
//! variants (Metal/CUDA) will replace.
//!
//! # Constraints
//!
//! - `n` may be any positive size; host packing uses a flat stream and the
//!   Metal bridge repacks it into output-column-major bytes.
//! - `k` must be a positive multiple of `group_size` (or the trailing
//!   partial group, if any, must still be in-bounds; this is what
//!   `ternary_pack` itself allows).
//! - `m, k, n > 0`. `group_size > 0`.
//! - All buffers must agree with their declared shape.
//!
//! No `unsafe`, no allocation outside the caller-owned buffers.

use crate::error::{KernelError, Result};

/// Fused ternary matmul. See module-level docs for layout and
/// constraints.
///
/// `out[m, n] = sum_k a[m, k] * dequant(b_packed[k, n], group_size)[n]`
#[allow(clippy::too_many_arguments)]
pub fn ternary_matmul(
    a: &[f32],
    b_packed: &[u8],
    b_scales: &[f32],
    b_zeros: &[f32],
    group_size: usize,
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) -> Result<()> {
    // ---- dimension / divisor checks ---------------------------------
    if m == 0 {
        return Err(KernelError::ZeroDimension { what: "m", got: 0 });
    }
    if k == 0 {
        return Err(KernelError::ZeroDimension { what: "k", got: 0 });
    }
    if n == 0 {
        return Err(KernelError::ZeroDimension { what: "n", got: 0 });
    }
    if group_size == 0 {
        return Err(KernelError::ZeroDimension {
            what: "group_size",
            got: 0,
        });
    }
    let activation_len = m
        .checked_mul(k)
        .ok_or(KernelError::DimensionOverflow { what: "m*k" })?;
    let output_len = m
        .checked_mul(n)
        .ok_or(KernelError::DimensionOverflow { what: "m*n" })?;
    let flat = k
        .checked_mul(n)
        .ok_or(KernelError::DimensionOverflow { what: "k*n" })?;
    let need_bytes = flat.checked_add(3).ok_or(KernelError::DimensionOverflow {
        what: "packed byte count",
    })? / 4;
    let num_groups = flat
        .checked_add(group_size - 1)
        .ok_or(KernelError::DimensionOverflow {
            what: "group count",
        })?
        / group_size;
    // ---- buffer length checks --------------------------------------
    if a.len() != activation_len {
        return Err(KernelError::BadBufferLength {
            what: "a",
            expected: activation_len,
            got: a.len(),
        });
    }
    if out.len() != output_len {
        return Err(KernelError::BadBufferLength {
            what: "out",
            expected: output_len,
            got: out.len(),
        });
    }
    // The flat symbol stream is `[k, n]` -> `k * n` symbols, packed
    // 4-per-byte. The trailing partial byte is allowed (see
    // `ternary_pack`), but the caller-owned buffer must be exactly sized.
    if b_packed.len() != need_bytes {
        return Err(KernelError::BadBufferLength {
            what: "b_packed",
            expected: need_bytes,
            got: b_packed.len(),
        });
    }
    if b_scales.len() != num_groups {
        return Err(KernelError::BadBufferLength {
            what: "b_scales",
            expected: num_groups,
            got: b_scales.len(),
        });
    }
    if b_zeros.len() != num_groups {
        return Err(KernelError::BadBufferLength {
            what: "b_zeros",
            expected: num_groups,
            got: b_zeros.len(),
        });
    }
    for (index, value) in b_scales.iter().enumerate() {
        if !value.is_finite() {
            return Err(KernelError::NonFiniteValue {
                what: "b_scales",
                index,
            });
        }
    }
    for (index, value) in b_zeros.iter().enumerate() {
        if !value.is_finite() {
            return Err(KernelError::NonFiniteValue {
                what: "b_zeros",
                index,
            });
        }
    }

    // ---- zero the output -------------------------------------------
    for slot in out.iter_mut() {
        *slot = 0.0;
    }

    // ---- fused inner product ---------------------------------------
    //
    // For each (k_row, n_col) we want to fetch the 2-bit symbol at
    // flat index `flat = k_row * n + n_col`, scale it by
    // `(scale[g], zero[g])`, and accumulate `a[m, k_row] * w` into
    // `out[m, n_col]` for every m.
    //
    // Inner loop ordering (k_outer, n_outer, m_inner) gives good
    // cache behaviour on the activation rows; n_outer is in byte
    // stride, m_inner reuses a single packed byte.
    for k_row in 0..k {
        let a_col = k_row; // a is row-major [m, k]
        let row_base = k_row * n;
        for n_col in 0..n {
            let flat = row_base + n_col;
            let byte_idx = flat / 4;
            let bit_off = (flat % 4) * 2;
            let bits = (b_packed[byte_idx] >> bit_off) & 0b11;
            // Per-group scale / zero lookup. Group index is on the
            // *flat* symbol sequence.
            let g = flat / group_size;
            let scale = b_scales[g];
            let zero = b_zeros[g];
            let w = dequant_ternary(bits, scale, zero);
            if w == 0.0 {
                // Skip the (potentially long) accumulation when the
                // weight symbol is exactly zero.
                continue;
            }
            for row in 0..m {
                out[row * n + n_col] += a[row * k + a_col] * w;
            }
        }
    }
    Ok(())
}

/// Decode a single 2-bit ternary code into `(-1, 0, +1)` and apply
/// the per-group `(scale, zero)` affine transform. With Bonsai
/// defaults (`scale == 1.0`, `zero == 0.0`) this is just the sign.
#[inline]
fn dequant_ternary(bits: u8, scale: f32, zero: f32) -> f32 {
    // Bits: 0b01 -> +1, 0b10 -> -1, anything else -> 0.
    let unit = match bits & 0b11 {
        0b01 => 1.0f32,
        0b10 => -1.0f32,
        _ => 0.0f32,
    };
    zero + unit * scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::approx_eq;
    use crate::quantized::{ternary_pack, ternary_unpack, SignedTernary};

    /// Helper: compute `out = a @ b^T` where `a` is `[m, k]` and `b`
    /// is `[k, n]` f32 row-major, against the unpacked ternary
    /// reference.
    fn reference_matmul(
        a: &[f32],
        b_unpacked: &[SignedTernary],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    let w = match b_unpacked[kk * n + j] {
                        SignedTernary::Pos => 1.0,
                        SignedTernary::Neg => -1.0,
                        SignedTernary::Zero => 0.0,
                    };
                    acc += a[row * k + kk] * w;
                }
                out[row * n + j] = acc;
            }
        }
        out
    }

    #[test]
    fn fused_matches_unpack_then_matmul_small_case() {
        // m=4, k=8, n=4. Pack a [k=8, n=4] ternary weight and run
        // the fused kernel against the unpacked reference.
        let m = 4;
        let k = 8;
        let n = 4;
        let group_size = k * n; // single Bonsai group

        let values: Vec<SignedTernary> = (0..k * n)
            .map(|i| match i % 3 {
                0 => SignedTernary::Pos,
                1 => SignedTernary::Neg,
                _ => SignedTernary::Zero,
            })
            .collect();
        let (packed, scales, zeros) = ternary_pack(&values, group_size).unwrap();

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.125 - 1.0).collect();
        let mut out = vec![0.0f32; m * n];
        ternary_matmul(&a, &packed, &scales, &zeros, group_size, m, k, n, &mut out).unwrap();

        let mut unpacked = vec![SignedTernary::Zero; values.len()];
        ternary_unpack(
            &packed,
            &scales,
            &zeros,
            values.len(),
            group_size,
            &mut unpacked,
        )
        .unwrap();
        let expected = reference_matmul(&a, &unpacked, m, k, n);
        for (i, (&g, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(approx_eq(g, e), "mismatch at {i}: got {g}, expected {e}");
        }
    }

    #[test]
    fn rejects_mismatched_buffer_lengths() {
        let m = 2;
        let k = 4;
        let n = 4;
        let group_size = 4;
        let a = vec![0.0f32; m * k - 1]; // wrong length
        let b_packed = vec![0u8; (k * n).div_ceil(4)];
        let scales = vec![1.0f32; 1];
        let zeros = vec![0.0f32; 1];
        let mut out = vec![0.0f32; m * n];
        let err = ternary_matmul(
            &a, &b_packed, &scales, &zeros, group_size, m, k, n, &mut out,
        )
        .unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }

    #[test]
    fn rejects_extra_packed_and_metadata_buffers() {
        let a = vec![0.0f32; 4];
        let packed = vec![0u8; 2];
        let scales = vec![1.0f32, 1.0];
        let zeros = vec![0.0f32];
        let mut out = vec![0.0f32; 4];

        let packed_error =
            ternary_matmul(&a, &packed, &zeros, &zeros, 4, 1, 4, 4, &mut out).unwrap_err();
        assert!(matches!(packed_error, KernelError::BadBufferLength { .. }));

        let metadata_error =
            ternary_matmul(&a, &[0u8; 4], &scales, &zeros, 4, 1, 4, 4, &mut out).unwrap_err();
        assert!(matches!(
            metadata_error,
            KernelError::BadBufferLength { .. }
        ));
    }

    #[test]
    fn rejects_nonfinite_scale_or_zero_metadata() {
        let a = vec![0.0f32; 1];
        let packed = vec![0u8; 1];
        let mut out = vec![0.0f32; 4];

        let scale_error =
            ternary_matmul(&a, &packed, &[f32::NAN], &[0.0], 4, 1, 1, 4, &mut out).unwrap_err();
        assert!(matches!(
            scale_error,
            KernelError::NonFiniteValue {
                what: "b_scales",
                index: 0
            }
        ));

        let zero_error =
            ternary_matmul(&a, &packed, &[1.0], &[f32::INFINITY], 4, 1, 1, 4, &mut out)
                .unwrap_err();
        assert!(matches!(
            zero_error,
            KernelError::NonFiniteValue {
                what: "b_zeros",
                index: 0
            }
        ));
    }

    #[test]
    fn rejects_dimension_overflow_before_buffer_indexing() {
        let mut out = Vec::new();
        let err = ternary_matmul(&[], &[], &[], &[], 4, usize::MAX, 2, 4, &mut out).unwrap_err();
        assert!(matches!(
            err,
            KernelError::DimensionOverflow { what: "m*k" }
        ));

        let err = ternary_matmul(&[], &[], &[], &[], 4, 1, usize::MAX, 4, &mut out).unwrap_err();
        assert!(matches!(
            err,
            KernelError::DimensionOverflow { what: "k*n" }
        ));
    }

    #[test]
    fn rejects_zero_group_size() {
        let a = vec![0.0f32; 1];
        let b_packed = vec![0u8; 1];
        let scales = [1.0f32];
        let zeros = [0.0f32];
        let mut out = vec![0.0f32; 1];
        let err = ternary_matmul(&a, &b_packed, &scales, &zeros, 0, 1, 1, 1, &mut out).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn supports_n_tail_without_row_alignment() {
        let m = 1;
        let k = 4;
        let n = 3;
        let values = vec![SignedTernary::Pos; k * n];
        let (packed, scales, zeros) = ternary_pack(&values, k * n).unwrap();
        let mut out = vec![0.0f32; m * n];
        ternary_matmul(&[1.0, 2.0, 3.0, 4.0], &packed, &scales, &zeros, k * n, m, k, n, &mut out)
            .unwrap();
        assert_eq!(out, vec![10.0; n]);
    }

    #[test]
    fn all_pos_weight_reduces_to_plain_sum() {
        // When every weight is `Pos`, the kernel must produce
        // `out[m, n] = sum_k a[m, k]` (the same value for every n).
        let m = 3;
        let k = 6;
        let n = 8; // multiple of 4
        let group_size = k * n;
        let values = vec![SignedTernary::Pos; k * n];
        let (packed, scales, zeros) = ternary_pack(&values, group_size).unwrap();
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let mut out = vec![0.0f32; m * n];
        ternary_matmul(&a, &packed, &scales, &zeros, group_size, m, k, n, &mut out).unwrap();
        for row in 0..m {
            let expected = (0..k).map(|c| a[row * k + c]).sum::<f32>();
            for j in 0..n {
                assert!(
                    approx_eq(out[row * n + j], expected),
                    "row {row} col {j}: got {}, expected {}",
                    out[row * n + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn rejects_zero_dimensions() {
        let a = vec![0.0f32; 1];
        let b_packed = vec![0u8; 1];
        let scales = [1.0f32];
        let zeros = [0.0f32];
        // m == 0
        {
            let mut out = vec![0.0f32; 0];
            let err =
                ternary_matmul(&[], &b_packed, &scales, &zeros, 1, 0, 1, 1, &mut out).unwrap_err();
            assert!(matches!(err, KernelError::ZeroDimension { what: "m", .. }));
        }
        // k == 0
        {
            let mut out = vec![0.0f32; 1];
            let err =
                ternary_matmul(&a, &b_packed, &scales, &zeros, 1, 1, 0, 1, &mut out).unwrap_err();
            assert!(matches!(err, KernelError::ZeroDimension { what: "k", .. }));
        }
        // n == 0
        {
            let mut out = vec![0.0f32; 0];
            let err =
                ternary_matmul(&a, &b_packed, &scales, &zeros, 1, 1, 1, 0, &mut out).unwrap_err();
            assert!(matches!(err, KernelError::ZeroDimension { what: "n", .. }));
        }
    }
}
