// turbo-quant-zig — Rust wrapper for the Zig implementation.
//
// Build pipeline (build.rs, when feature = "zig"):
//   1. `zig build-lib` produces `libturbo_quant_zig.a` (C ABI)
//   2. We `extern "C"` declare those functions
//   3. We expose them through a Rust-friendly API matching perf-core/turbo-quant
//
// All hot-path work runs in Zig (LLVM-compiled to native). The Rust wrapper
// is zero-cost.
//
// Without `--features zig`, the crate compiles to a no-op stub (so the
// workspace always builds, even when `zig` is not installed).

#[cfg(feature = "zig")]
use native::{zig_encode, zig_decode};

#[derive(Debug, Clone)]
pub struct ZigQuantizedTensor {
    pub shape: Vec<usize>,
    pub packed: Vec<u8>,
    pub scales: Vec<f32>,
    pub zeros: Vec<f32>,
}

impl ZigQuantizedTensor {
    /// Encode `data` to TurboQuant via the Zig kernel.
    /// Without feature "zig", returns an error stub.
    pub fn encode(data: &[f32], bits: u8, group_size: usize) -> Result<Self, String> {
        #[cfg(feature = "zig")]
        {
            zig_encode(data, bits, group_size)
        }
        #[cfg(not(feature = "zig"))]
        {
            let _ = (data, bits, group_size);
            Err("turbo-quant-zig: built without --features zig".to_string())
        }
    }

    /// Decode a TurboQuant tensor.
    /// Without feature "zig", returns zeros.
    pub fn decode(&self, n: usize, group_size: usize, bits: u8) -> Vec<f32> {
        #[cfg(feature = "zig")]
        {
            zig_decode(&self.packed, &self.scales, &self.zeros, n, group_size, bits)
        }
        #[cfg(not(feature = "zig"))]
        {
            let _ = (n, group_size, bits);
            vec![0.0; n]
        }
    }
}

// ── Zig-native implementation ──────────────────────────────────────────
#[cfg(feature = "zig")]
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

        fn tq_zig_free(ptr: *mut c_void, size: usize);
    }

    pub(super) fn zig_encode(
        data: &[f32], bits: u8, group_size: usize,
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

        let shape  = unsafe { std::slice::from_raw_parts(shape_ptr,  shape_len)  }.to_vec();
        let packed = unsafe { std::slice::from_raw_parts(packed_ptr, packed_len) }.to_vec();
        let scales = unsafe { std::slice::from_raw_parts(scales_ptr, scales_len) }.to_vec();
        let zeros  = unsafe { std::slice::from_raw_parts(zeros_ptr,  zeros_len)  }.to_vec();

        unsafe {
            tq_zig_free(shape_ptr  as *mut c_void, shape_len * std::mem::size_of::<usize>());
            tq_zig_free(packed_ptr as *mut c_void, packed_len * std::mem::size_of::<u8>());
            tq_zig_free(scales_ptr as *mut c_void, scales_len * std::mem::size_of::<f32>());
            tq_zig_free(zeros_ptr  as *mut c_void, zeros_len * std::mem::size_of::<f32>());
        }

        Ok(ZigQuantizedTensor { shape, packed, scales, zeros })
    }

    pub(super) fn zig_decode(
        packed: &[u8], scales: &[f32], zeros: &[f32],
        n: usize, group_size: usize, bits: u8,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        if n == 0 {
            return out;
        }
        let packed_ptr = if packed.is_empty() { std::ptr::null() } else { packed.as_ptr() };
        let scales_ptr = if scales.is_empty() { std::ptr::null() } else { scales.as_ptr() };
        let zeros_ptr = if zeros.is_empty() { std::ptr::null() } else { zeros.as_ptr() };
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

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zig_stub_returns_error_without_feature() {
        // When built without `--features zig`, encode returns an error.
        // This makes the crate's behavior deterministic on any host.
        let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01 - 0.64).collect();
        let r = ZigQuantizedTensor::encode(&data, 4, 32);
        #[cfg(feature = "zig")]
        {
            assert!(r.is_ok(), "encode failed: {:?}", r.err());
            let q = r.unwrap();
            let decoded = q.decode(data.len(), 32, 4);
            for (a, b) in data.iter().zip(decoded.iter()) {
                assert!((a - b).abs() < 0.15, "{} vs {}", a, b);
            }
        }
        #[cfg(not(feature = "zig"))]
        assert!(r.is_err(), "encode should fail without --features zig");
    }

    #[test]
    fn zig_decode_returns_zeros_without_feature() {
        let q = ZigQuantizedTensor { shape: vec![8], packed: vec![], scales: vec![], zeros: vec![] };
        let r = q.decode(8, 8, 4);
        #[cfg(feature = "zig")]
        assert!(r.iter().all(|&x| x == 0.0)); // since packed/scales/zeros are empty
        #[cfg(not(feature = "zig"))]
        assert!(r.iter().all(|&x| x == 0.0));
    }
}
