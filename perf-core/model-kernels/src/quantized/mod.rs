//! Ternary (Bonsai-style) and sub-byte (2/3/4/5/6/7/8-bit) quantization
//! kernels.
//!
//! Both pack formats are *symmetric* per group: each group of
//! `group_size` values is quantized relative to its own `(min, max)`
//! pair with `bits` bits per element. Ternary is a special case that
//! only carries the sign-magnitude ternary code in 2 bits and stores
//! trivial `scale = 1.0` / `zero = 0.0` per group.
//!
//! All functions are pure: no allocation outside the returned buffers,
//! no global state, deterministic.

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

/// Symmetric sub-byte pack. `bits` must be in `1..=8`. Per-group
/// `(scale, zero)` are computed as
/// `scale = (max - min) / ((1 << bits) - 1)`, `zero = min`.
pub fn subbyte_pack(
    values: &[f32],
    bits: u8,
    group_size: usize,
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>)> {
    if bits == 0 || bits > 8 {
        return Err(KernelError::BitsOutOfRange { bits });
    }
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
    let mut scales = Vec::with_capacity(num_groups);
    let mut zeros = Vec::with_capacity(num_groups);
    let bits_us = bits as usize;
    let levels = (1u32 << bits_us) - 1;
    let mut packed = vec![0u8; n.div_ceil(8 / bits_us.max(1)).max(1)];
    let mut bit_pos = 0usize;
    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(n);
        let group = &values[start..end];
        let (min, max) = min_max(group);
        let scale = if max > min {
            (max - min) / levels as f32
        } else {
            1.0
        };
        let zero = min;
        scales.push(scale);
        zeros.push(zero);
        for (_idx, &v) in group.iter().enumerate() {
            let q = if scale > 0.0 {
                let normalized = ((v - zero) / scale).round();
                normalized.clamp(0.0, levels as f32) as u32
            } else {
                0
            };
            // Position `bit_pos` in `packed`. The symbol may
            // straddle a byte boundary, in which case the high bits
            // go into the next byte.
            let byte_idx = bit_pos / 8;
            let bit_off = bit_pos % 8;
            // Grow `packed` if needed (always reserve one extra
            // byte when straddling is possible).
            while byte_idx >= packed.len() {
                packed.push(0);
            }
            if bits_us == 8 {
                packed[byte_idx] = q as u8;
            } else if bit_off + bits_us <= 8 {
                packed[byte_idx] |= (q as u8) << bit_off;
            } else {
                let low = 8 - bit_off;
                let high = bits_us - low;
                let low_part = ((q as u8) & ((1u32 << low) - 1) as u8) << bit_off;
                let high_part = ((q >> low) as u8) & ((1u32 << high) - 1) as u8;
                packed[byte_idx] |= low_part;
                while byte_idx + 1 >= packed.len() {
                    packed.push(0);
                }
                packed[byte_idx + 1] |= high_part;
            }
            bit_pos += bits_us;
        }
    }
    Ok((packed, scales, zeros))
}

/// Symmetric sub-byte unpack. Reads `n` symbols from `packed` using
/// `group_size` for scale/zero lookup.
#[allow(clippy::too_many_arguments)]
pub fn subbyte_unpack(
    packed: &[u8],
    scales: &[f32],
    zeros: &[f32],
    n: usize,
    group_size: usize,
    bits: u8,
    out: &mut [f32],
) -> Result<()> {
    if bits == 0 || bits > 8 {
        return Err(KernelError::BitsOutOfRange { bits });
    }
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
    let bits_us = bits as usize;
    let levels = (1u32 << bits_us) - 1;
    let mut bit_pos = 0usize;
    for i in 0..n {
        let g = i / group_size;
        if g >= scales.len() || g >= zeros.len() {
            return Err(KernelError::BadBufferLength {
                what: "scales/zeros",
                expected: g + 1,
                got: scales.len().min(zeros.len()),
            });
        }
        let scale = scales[g];
        let zero = zeros[g];
        let byte_idx = bit_pos / 8;
        let bit_off = bit_pos % 8;
        if byte_idx >= packed.len() {
            return Err(KernelError::BadBufferLength {
                what: "packed",
                expected: byte_idx + 1,
                got: packed.len(),
            });
        }
        let mask = if bits_us == 8 {
            0xFFu8
        } else {
            ((1u32 << bits_us) - 1) as u8
        };
        let q = if bits_us == 8 {
            packed[byte_idx] as u32
        } else if bit_off + bits_us <= 8 {
            ((packed[byte_idx] >> bit_off) & mask) as u32
        } else {
            // Straddles byte boundary: read `low` bits from the
            // current byte and `high` bits from the next byte.
            let low = 8 - bit_off;
            let high = bits_us - low;
            let low_part = (packed[byte_idx] >> bit_off) as u32;
            let high_part = if byte_idx + 1 < packed.len() {
                (packed[byte_idx + 1] as u32) & ((1u32 << high) - 1)
            } else {
                0
            };
            (low_part | (high_part << low)) as u32
        };
        let v = if levels == 0 {
            zero
        } else {
            zero + (q as f32) * scale
        };
        out[i] = v;
        bit_pos += bits_us;
    }
    Ok(())
}

fn min_max(xs: &[f32]) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &x in xs {
        if x < lo {
            lo = x;
        }
        if x > hi {
            hi = x;
        }
    }
    if !lo.is_finite() {
        lo = 0.0;
    }
    if !hi.is_finite() {
        hi = 0.0;
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_pack_layout_matches_bit_table() {
        let values = vec![
            SignedTernary::Zero,
            SignedTernary::Pos,
            SignedTernary::Neg,
            SignedTernary::Pos,
            SignedTernary::Zero,
            SignedTernary::Neg,
            SignedTernary::Neg,
            SignedTernary::Pos,
        ];
        let (packed, scales, zeros) = ternary_pack(&values, 8).unwrap();
        assert_eq!(packed.len(), 2);
        assert_eq!(scales.len(), 1);
        assert_eq!(zeros.len(), 1);
        assert_eq!(packed[0], 0b01_10_01_00);
        assert_eq!(packed[1], 0b01_10_10_00);
    }

    #[test]
    fn ternary_round_trip_inverts_pack() {
        let values = vec![
            SignedTernary::Pos,
            SignedTernary::Zero,
            SignedTernary::Neg,
            SignedTernary::Pos,
            SignedTernary::Neg,
            SignedTernary::Pos,
            SignedTernary::Zero,
            SignedTernary::Pos,
        ];
        let (packed, scales, zeros) = ternary_pack(&values, 8).unwrap();
        let mut out = vec![SignedTernary::Zero; values.len()];
        ternary_unpack(&packed, &scales, &zeros, values.len(), 8, &mut out).unwrap();
        assert_eq!(out, values);
    }

    #[test]
    fn ternary_pack_zero_group_size_is_error() {
        let err = ternary_pack(&[], 0).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn ternary_partial_trailing_group_packs_cleanly() {
        let values = vec![SignedTernary::Pos, SignedTernary::Neg, SignedTernary::Zero];
        let (packed, scales, zeros) = ternary_pack(&values, 4).unwrap();
        // 3 values, group_size=4: a single trailing group is
        // emitted with the first three slots populated; the
        // remaining slot is Zero. Therefore packed is one byte
        // and scales/zeros have length 1.
        assert_eq!(packed.len(), 1);
        assert_eq!(scales.len(), 1);
        assert_eq!(zeros.len(), 1);
        let mut out = vec![SignedTernary::Zero; values.len()];
        ternary_unpack(&packed, &scales, &zeros, values.len(), 4, &mut out).unwrap();
        assert_eq!(out, values);
    }

    #[test]
    fn subbyte_round_trip_bits_2_3_4() {
        for &bits in &[2u8, 3, 4] {
            let n = 8;
            let group_size = 8;
            let values: Vec<f32> = (0..n).map(|i| i as f32 / (n as f32)).collect();
            let (packed, scales, zeros) = subbyte_pack(&values, bits, group_size).unwrap();
            let mut out = vec![0.0f32; n];
            subbyte_unpack(&packed, &scales, &zeros, n, group_size, bits, &mut out).unwrap();
            let slack = 1.0 / (1u32 << bits) as f32;
            for (i, (&v, &r)) in values.iter().zip(out.iter()).enumerate() {
                let tol = slack + 1e-5;
                assert!(
                    (v - r).abs() <= tol + 1e-4 * v.abs(),
                    "bits={bits} idx={i}: got {r}, expected {v} (slack {slack})"
                );
            }
        }
    }

    #[test]
    fn subbyte_pack_rejects_bits_outside_1_to_8() {
        let values = vec![0.0f32; 4];
        assert!(matches!(
            subbyte_pack(&values, 0, 4).unwrap_err(),
            KernelError::BitsOutOfRange { .. }
        ));
        assert!(matches!(
            subbyte_pack(&values, 9, 4).unwrap_err(),
            KernelError::BitsOutOfRange { .. }
        ));
    }

    #[test]
    fn subbyte_pack_rejects_zero_group_size() {
        let err = subbyte_pack(&[0.0f32; 4], 4, 0).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn subbyte_handles_partial_trailing_group() {
        let values = vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let group_size = 4;
        let bits = 4;
        let (packed, scales, zeros) = subbyte_pack(&values, bits, group_size).unwrap();
        let mut out = vec![0.0f32; values.len()];
        subbyte_unpack(&packed, &scales, &zeros, values.len(), group_size, bits, &mut out)
            .unwrap();
        let slack = 1.0 / (1u32 << bits) as f32;
        for (i, (&v, &r)) in values.iter().zip(out.iter()).enumerate() {
            let tol = slack + 1e-5;
            assert!(
                (v - r).abs() <= tol + 1e-4 * v.abs(),
                "idx {i}: got {r}, expected {v}"
            );
        }
    }
}