use crate::QuantizedTensor;

pub fn decode_uniform(tensor: &QuantizedTensor, out: &mut [f32]) {
    assert!(
        (2..=4).contains(&tensor.bits),
        "turbo-quant: stored bits must be in 2..=4, got {}",
        tensor.bits
    );
    assert!(
        tensor.group_size > 0,
        "turbo-quant: stored group_size must be > 0, got {}",
        tensor.group_size
    );
    let expected = tensor.shape.iter().product::<usize>();
    assert!(
        out.len() == expected,
        "turbo-quant: decode_uniform output length mismatch — expected \
             {expected}, got {}",
        out.len()
    );

    let bits = tensor.bits;
    let qmax = ((1u32 << bits) - 1) as f32;
    let bp = bits as usize;

    for (i, v) in out.iter_mut().enumerate() {
        let g = i / tensor.group_size;
        let s = tensor.scales.get(g).copied().unwrap_or(1e-12);
        let z = tensor.zeros.get(g).copied().unwrap_or(0.0);

        // Read `bp` bits from packed[], LSB-first within each byte.
        let mut bc = i * bp;
        let mut raw = 0u32;
        let mut remaining = bp;
        let mut shift = 0;
        while remaining > 0 {
            let byte_idx = bc / 8;
            let bit_off = bc % 8;
            let room = 8 - bit_off;
            let take = remaining.min(room);
            let mask = (1u32 << take) - 1;
            let byte = tensor.packed.get(byte_idx).copied().unwrap_or(0) as u32;
            raw |= ((byte >> bit_off) & mask) << shift;
            shift += take;
            bc += take;
            remaining -= take;
        }

        let q = (raw as f32).min(qmax);
        *v = q * s + z;
    }
}
