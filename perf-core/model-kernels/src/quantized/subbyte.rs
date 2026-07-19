//! Symmetric sub-byte pack/unpack for 2..=8-bit values.
//!
//! See the parent module docs for the symmetric per-group quantization
//! scheme. The `min_max` helper is private to this module.

use crate::error::{KernelError, Result};

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
