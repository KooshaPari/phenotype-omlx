pub use turbo_quant_c_build::*;

mod turbo_quant_c_build {
    extern "C" {
        fn tq_c_encode(
            data: *const f32,
            n: usize,
            bits: u8,
            group_size: usize,
            out_shape: *mut *mut usize,
            out_shape_len: *mut usize,
            out_packed: *mut *mut u8,
            out_packed_len: *mut usize,
            out_scales: *mut *mut f32,
            out_scales_len: *mut usize,
            out_zeros: *mut *mut f32,
            out_zeros_len: *mut usize,
        ) -> bool;
        fn tq_c_decode(
            packed: *const u8,
            packed_len: usize,
            scales: *const f32,
            zeros: *const f32,
            n: usize,
            group_size: usize,
            bits: u8,
            out: *mut f32,
        );
        fn tq_c_free(ptr: *mut std::ffi::c_void);
    }

    pub struct CTensor {
        pub shape: Vec<usize>,
        pub packed: Vec<u8>,
        pub scales: Vec<f32>,
        pub zeros: Vec<f32>,
    }

    impl Clone for CTensor {
        fn clone(&self) -> Self {
            Self {
                shape: self.shape.clone(),
                packed: self.packed.clone(),
                scales: self.scales.clone(),
                zeros: self.zeros.clone(),
            }
        }
    }

    pub fn encode(data: &[f32], bits: u8, group_size: usize) -> Option<CTensor> {
        let mut shape: *mut usize = std::ptr::null_mut();
        let mut shape_len: usize = 0;
        let mut packed: *mut u8 = std::ptr::null_mut();
        let mut packed_len: usize = 0;
        let mut scales: *mut f32 = std::ptr::null_mut();
        let mut scales_len: usize = 0;
        let mut zeros: *mut f32 = std::ptr::null_mut();
        let mut zeros_len: usize = 0;

        let ok = unsafe {
            tq_c_encode(
                data.as_ptr(),
                data.len(),
                bits,
                group_size,
                &mut shape,
                &mut shape_len,
                &mut packed,
                &mut packed_len,
                &mut scales,
                &mut scales_len,
                &mut zeros,
                &mut zeros_len,
            )
        };
        if !ok {
            return None;
        }

        let t = CTensor {
            shape: unsafe { std::slice::from_raw_parts(shape, shape_len).to_vec() },
            packed: unsafe { std::slice::from_raw_parts(packed, packed_len).to_vec() },
            scales: unsafe { std::slice::from_raw_parts(scales, scales_len).to_vec() },
            zeros: unsafe { std::slice::from_raw_parts(zeros, zeros_len).to_vec() },
        };

        unsafe {
            tq_c_free(shape as *mut _);
            tq_c_free(packed as *mut _);
            tq_c_free(scales as *mut _);
            tq_c_free(zeros as *mut _);
        }
        Some(t)
    }

    /// Decode into a caller-owned buffer.
    ///
    /// The C ABI is contractually required to leave `out` untouched whenever
    /// its inputs are invalid (null pointers, zero `n`, `bits` outside
    /// 2..=4, `group_size == 0`, mismatched `packed_len`, or overflow). Letting
    /// the caller supply the buffer lets tests pre-fill a sentinel and verify
    /// the no-write guarantee directly, instead of papering over invalidity
    /// with a magic value returned from `decode`.
    ///
    /// # Panics
    ///
    /// Panics if `out.len() < n` — the C ABI writes `n` elements when it
    /// succeeds, so an undersized buffer is a logic error rather than a
    /// recoverable condition. Excess capacity is allowed and ignored.
    pub fn decode_into(t: &CTensor, n: usize, group_size: usize, bits: u8, out: &mut [f32]) {
        assert!(
            out.len() >= n,
            "decode_into: output buffer length ({}) must be >= n ({})",
            out.len(),
            n
        );
        unsafe {
            tq_c_decode(
                t.packed.as_ptr(),
                t.packed.len(),
                t.scales.as_ptr(),
                t.zeros.as_ptr(),
                n,
                group_size,
                bits,
                out.as_mut_ptr(),
            );
        }
    }

    /// Convenience wrapper that allocates a fresh zero-initialised output.
    /// Suitable for valid callers; tests that need to observe rejection
    /// semantics should use [`decode_into`] with a sentinel-pre-filled buffer.
    pub fn decode(t: &CTensor, n: usize, group_size: usize, bits: u8) -> Vec<f32> {
        let mut out = vec![0f32; n];
        decode_into(t, n, group_size, bits, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{decode, decode_into, encode};

    #[test]
    fn rejects_invalid_encode_arguments() {
        assert!(encode(&[], 2, 4).is_none());
        assert!(encode(&[1.0], 1, 4).is_none());
        assert!(encode(&[1.0], 5, 4).is_none());
        assert!(encode(&[1.0], 2, 0).is_none());
    }

    #[test]
    fn round_trips_all_supported_bit_widths() {
        let input = [-3.0, -1.0, 0.5, 2.0, 7.0, 9.0, 12.0];

        for bits in [2u8, 3, 4] {
            let tensor = encode(&input, bits, 3).expect("supported encoding");
            assert_eq!(tensor.shape, vec![input.len()]);
            assert_eq!(tensor.scales.len(), 3);
            assert_eq!(tensor.zeros.len(), 3);

            let output = decode(&tensor, input.len(), 3, bits);
            for (actual, expected) in output.iter().zip(input) {
                let tolerance = tensor.scales.iter().copied().fold(0.0_f32, f32::max);
                assert!((actual - expected).abs() <= tolerance + 1e-5);
            }
        }
    }

    /// Regression coverage for partial trailing groups where the packed
    /// bitstream ends mid-byte. n=7 with group_size=3 leaves a single-element
    /// final group; for bits=3 and bits=4 the last value writes into a byte
    /// that the previous one-byte-at-a-time implementation silently truncated.
    /// Decoded values must match within the per-group quantisation tolerance
    /// and the packed length must equal the contract `(n * bits + 7) / 8`.
    #[test]
    fn round_trips_partial_trailing_group_across_byte_boundary() {
        let input = [-3.0, -1.0, 0.5, 2.0, 7.0, 9.0, 12.0];

        for bits in [2u8, 3, 4] {
            let tensor = encode(&input, bits, 3).expect("supported encoding");
            assert_eq!(tensor.packed.len(), (input.len() * bits as usize + 7) / 8);

            let mut out = vec![f32::NAN; input.len()];
            decode_into(&tensor, input.len(), 3, bits, &mut out);
            for (actual, expected) in out.iter().zip(input) {
                let tolerance = tensor.scales.iter().copied().fold(0.0_f32, f32::max);
                assert!(!actual.is_nan(), "decode left NaN at bits={bits}");
                assert!((actual - expected).abs() <= tolerance + 1e-5);
            }
        }
    }

    #[test]
    fn decode_ignores_invalid_bounds_without_writing_output() {
        let tensor = encode(&[0.0, 1.0, 2.0, 3.0], 3, 4).expect("supported encoding");
        let sentinel_value = 91.0_f32;

        // `decode_into` lets each branch observe the C ABI's no-write
        // guarantee by pre-filling its own buffer with a sentinel and
        // verifying that an invalid call leaves the buffer untouched.

        // Sanity: a valid call DOES overwrite the sentinel. Recorded first
        // so the remaining cases can move/clone `tensor` freely.
        let mut buf = vec![sentinel_value; 4];
        decode_into(&tensor, 4, 4, 3, &mut buf);
        assert_ne!(buf, vec![sentinel_value; 4]);

        // group_size == 0
        let mut buf = vec![sentinel_value; 4];
        decode_into(&tensor, 4, 0, 3, &mut buf);
        assert_eq!(buf, vec![sentinel_value; 4]);

        // bits == 1 (outside 2..=4)
        let mut buf = vec![sentinel_value; 4];
        decode_into(&tensor, 4, 4, 1, &mut buf);
        assert_eq!(buf, vec![sentinel_value; 4]);

        // bits == 5 (outside 2..=4)
        let mut buf = vec![sentinel_value; 4];
        decode_into(&tensor, 4, 4, 5, &mut buf);
        assert_eq!(buf, vec![sentinel_value; 4]);

        // truncated packed buffer (packed_len mismatch)
        let mut truncated = tensor;
        truncated.packed.pop();
        let mut buf = vec![sentinel_value; 4];
        decode_into(&truncated, 4, 4, 3, &mut buf);
        assert_eq!(buf, vec![sentinel_value; 4]);
    }
}
