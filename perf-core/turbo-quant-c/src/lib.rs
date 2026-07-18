pub use turbo_quant_c_build::*;
mod turbo_quant_c_build {
    extern "C" {
        fn tq_c_encode(data: *const f32, n: usize, bits: u8, group_size: usize,
            out_shape: *mut *mut usize, out_shape_len: *mut usize,
            out_packed: *mut *mut u8, out_packed_len: *mut usize,
            out_scales: *mut *mut f32, out_scales_len: *mut usize,
            out_zeros: *mut *mut f32, out_zeros_len: *mut usize) -> bool;
        fn tq_c_decode(packed: *const u8, packed_len: usize,
            scales: *const f32, zeros: *const f32,
            n: usize, group_size: usize, bits: u8, out: *mut f32);
        fn tq_c_free(ptr: *mut std::ffi::c_void);
    }
    pub struct CTensor { pub shape: Vec<usize>, pub packed: Vec<u8>,
        pub scales: Vec<f32>, pub zeros: Vec<f32> }
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
            tq_c_encode(data.as_ptr(), data.len(), bits, group_size,
                &mut shape, &mut shape_len,
                &mut packed, &mut packed_len,
                &mut scales, &mut scales_len,
                &mut zeros, &mut zeros_len)
        };
        if !ok { return None; }
        let t = CTensor {
            shape: unsafe { std::slice::from_raw_parts(shape, shape_len).to_vec() },
            packed: unsafe { std::slice::from_raw_parts(packed, packed_len).to_vec() },
            scales: unsafe { std::slice::from_raw_parts(scales, scales_len).to_vec() },
            zeros: unsafe { std::slice::from_raw_parts(zeros, zeros_len).to_vec() },
        };
        unsafe { tq_c_free(shape as *mut _); tq_c_free(packed as *mut _);
            tq_c_free(scales as *mut _); tq_c_free(zeros as *mut _); }
        Some(t)
    }
    pub fn decode(t: &CTensor, n: usize, group_size: usize, bits: u8) -> Vec<f32> {
        let mut out = vec![0f32; n];
        unsafe { tq_c_decode(t.packed.as_ptr(), t.packed.len(),
            t.scales.as_ptr(), t.zeros.as_ptr(),
            n, group_size, bits, out.as_mut_ptr()); }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{decode, encode};

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

        for bits in [2, 3, 4] {
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

    #[test]
    fn decode_ignores_invalid_bounds_without_writing_output() {
        let tensor = encode(&[0.0, 1.0, 2.0, 3.0], 3, 4).expect("supported encoding");
        let sentinel = vec![91.0; 4];

        assert_eq!(decode(&tensor, 4, 0, 3), sentinel);
        assert_eq!(decode(&tensor, 4, 4, 1), sentinel);
        assert_eq!(decode(&tensor, 4, 4, 3), sentinel);

        let mut truncated = tensor;
        truncated.packed.pop();
        assert_eq!(decode(&truncated, 4, 4, 3), sentinel);
    }
}
