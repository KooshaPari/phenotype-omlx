use crate::{minmax::min_max, QuantizedTensor};

pub fn encode_uniform(data: &[f32], bits: u8, group_size: usize) -> QuantizedTensor {
    assert!(
        (2..=4).contains(&bits),
        "turbo-quant: bits must be in 2..=4, got {bits}"
    );
    assert!(
        group_size > 0,
        "turbo-quant: group_size must be greater than zero"
    );
    assert!(
        !data.is_empty(),
        "turbo-quant: encode_uniform requires non-empty input"
    );
    assert!(
        data.iter().all(|v| v.is_finite()),
        "turbo-quant: encode_uniform requires finite input data"
    );

    let qmax = ((1u32 << bits) - 1) as f32;
    let n_packed = (data.len() * bits as usize).div_ceil(8);
    let mut packed = vec![0u8; n_packed];
    let mut scales = Vec::new();
    let mut zeros = Vec::new();

    let mut bit_cursor = 0usize;
    for chunk in data.chunks(group_size) {
        // SIMD-dispatched min/max: NEON on aarch64, scalar fallback elsewhere.
        let (min, max) = min_max(chunk);
        let scale = (max - min) / qmax;
        let zero = min;
        scales.push(scale.max(1e-12));
        zeros.push(zero);

        for &v in chunk {
            let q = ((v - zero) / scale).round().clamp(0.0, qmax) as u32;
            let bp = bits as usize;
            // Write `bp` bits into packed[], LSB-first within each byte.
            let mut val = q;
            let mut remaining = bp;
            while remaining > 0 {
                let byte_idx = bit_cursor / 8;
                let bit_off = bit_cursor % 8;
                let room = 8 - bit_off;
                let take = remaining.min(room);
                let mask = (1u32 << take) - 1;
                packed[byte_idx] |= ((val & mask) as u8) << bit_off;
                val >>= take;
                bit_cursor += take;
                remaining -= take;
            }
        }
    }

    QuantizedTensor {
        shape: vec![data.len()],
        bits,
        group_size,
        packed,
        scales,
        zeros,
    }
}
