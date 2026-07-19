// turbo-quant-zig — Rust wrapper for the Zig implementation.
//
// Requires the Zig compiler in PATH (`brew install zig`). build.rs compiles
// zig-src/turbo_quant.zig via `zig build-lib` and links it unconditionally.

use native::{zig_decode, zig_encode};

#[derive(Debug, Clone)]
pub struct ZigQuantizedTensor {
    pub shape: Vec<usize>,
    pub packed: Vec<u8>,
    pub scales: Vec<f32>,
    pub zeros: Vec<f32>,
}

impl ZigQuantizedTensor {
    /// Encode `data` to TurboQuant via the Zig kernel.
    pub fn encode(data: &[f32], bits: u8, group_size: usize) -> Result<Self, String> {
        zig_encode(data, bits, group_size)
    }

    /// Decode a TurboQuant tensor.
    pub fn decode(&self, n: usize, group_size: usize, bits: u8) -> Vec<f32> {
        zig_decode(&self.packed, &self.scales, &self.zeros, n, group_size, bits)
    }
}

mod native {
    use super::ZigQuantizedTensor;
    use std::os::raw::{c_uchar, c_void};

    extern "C" {
        fn tq_zig_encode(
            data_ptr: *const f32,
            n: usize,
            bits: c_uchar,
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

        fn tq_zig_decode(
            packed_ptr: *const u8,
            packed_len: usize,
            scales_ptr: *const f32,
            zeros_ptr: *const f32,
            n: usize,
            group_size: usize,
            bits: c_uchar,
            out_ptr: *mut f32,
        );

        fn tq_zig_free(ptr: *mut c_void, byte_len: usize);
    }

    pub(super) fn zig_encode(
        data: &[f32],
        bits: u8,
        group_size: usize,
    ) -> Result<ZigQuantizedTensor, String> {
        let mut shape_ptr: *mut usize = std::ptr::null_mut();
        let mut shape_len: usize = 0;
        let mut packed_ptr: *mut u8 = std::ptr::null_mut();
        let mut packed_len: usize = 0;
        let mut scales_ptr: *mut f32 = std::ptr::null_mut();
        let mut scales_len: usize = 0;
        let mut zeros_ptr: *mut f32 = std::ptr::null_mut();
        let mut zeros_len: usize = 0;

        let ok = unsafe {
            tq_zig_encode(
                data.as_ptr(),
                data.len(),
                bits,
                group_size,
                &mut shape_ptr,
                &mut shape_len,
                &mut packed_ptr,
                &mut packed_len,
                &mut scales_ptr,
                &mut scales_len,
                &mut zeros_ptr,
                &mut zeros_len,
            )
        };
        if !ok {
            return Err("Zig tq_zig_encode returned false".to_string());
        }

        let shape = unsafe { std::slice::from_raw_parts(shape_ptr, shape_len) }.to_vec();
        let packed = unsafe { std::slice::from_raw_parts(packed_ptr, packed_len) }.to_vec();
        let scales = unsafe { std::slice::from_raw_parts(scales_ptr, scales_len) }.to_vec();
        let zeros = unsafe { std::slice::from_raw_parts(zeros_ptr, zeros_len) }.to_vec();

        unsafe {
            tq_zig_free(
                shape_ptr as *mut c_void,
                shape_len * std::mem::size_of::<usize>(),
            );
            tq_zig_free(
                packed_ptr as *mut c_void,
                packed_len * std::mem::size_of::<u8>(),
            );
            tq_zig_free(
                scales_ptr as *mut c_void,
                scales_len * std::mem::size_of::<f32>(),
            );
            tq_zig_free(
                zeros_ptr as *mut c_void,
                zeros_len * std::mem::size_of::<f32>(),
            );
        }

        Ok(ZigQuantizedTensor {
            shape,
            packed,
            scales,
            zeros,
        })
    }

    pub(super) fn zig_decode(
        packed: &[u8],
        scales: &[f32],
        zeros: &[f32],
        n: usize,
        group_size: usize,
        bits: u8,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        if n == 0 {
            return out;
        }
        let packed_ptr = if packed.is_empty() {
            std::ptr::null()
        } else {
            packed.as_ptr()
        };
        let scales_ptr = if scales.is_empty() {
            std::ptr::null()
        } else {
            scales.as_ptr()
        };
        let zeros_ptr = if zeros.is_empty() {
            std::ptr::null()
        } else {
            zeros.as_ptr()
        };
        unsafe {
            tq_zig_decode(
                packed_ptr,
                packed.len(),
                scales_ptr,
                zeros_ptr,
                n,
                group_size,
                bits,
                out.as_mut_ptr(),
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zig_encode_decode_roundtrip() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01 - 0.64).collect();
        let q = ZigQuantizedTensor::encode(&data, 4, 32)
            .expect("Zig encode failed — verify zig is installed and build.rs succeeded");
        assert_eq!(q.shape, vec![data.len()]);
        let decoded = q.decode(data.len(), 32, 4);
        for (a, b) in data.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.15, "roundtrip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn zig_decode_empty_tensor() {
        let q = ZigQuantizedTensor {
            shape: vec![8],
            packed: vec![],
            scales: vec![],
            zeros: vec![],
        };
        let r = q.decode(0, 8, 4);
        assert!(r.is_empty());
    }
}
