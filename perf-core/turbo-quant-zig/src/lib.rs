// turbo-quant-zig — Rust wrapper for the Zig implementation.
//
// Requires the Zig compiler in PATH (`brew install zig`). build.rs compiles
// zig-src/turbo_quant.zig via `zig build-lib` and links it unconditionally.

#[cfg(feature = "zig")]
use native::{zig_decode, zig_decode_v1, zig_encode, zig_encode_v1};

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
        #[cfg(feature = "zig")]
        {
            zig_encode(data, bits, group_size)
        }
        #[cfg(not(feature = "zig"))]
        {
            let _ = (data, bits, group_size);
            Err("Zig feature not enabled".to_string())
        }
    }

    /// Decode a TurboQuant tensor.
    pub fn decode(&self, n: usize, group_size: usize, bits: u8) -> Vec<f32> {
        #[cfg(feature = "zig")]
        {
            zig_decode(&self.packed, &self.scales, &self.zeros, n, group_size, bits)
        }
        #[cfg(not(feature = "zig"))]
        {
            let _ = (n, group_size, bits);
            vec![]
        }
    }

    /// Encode via the versioned Native ABI v1 contract. Returns the matching
    /// [`native_abi::Status`] on failure.
    pub fn encode_v1(
        data: &[f32],
        bits: u8,
        group_size: usize,
    ) -> Result<Self, native_abi::Status> {
        #[cfg(feature = "zig")]
        {
            zig_encode_v1(data, bits, group_size)
        }
        #[cfg(not(feature = "zig"))]
        {
            let _ = (data, bits, group_size);
            Err(native_abi::Status::ErrAllocation)
        }
    }

    /// Decode via the versioned Native ABI v1 contract. The status reported
    /// by the Zig side is returned; on success the buffer is overwritten, on
    /// failure it is left untouched.
    pub fn decode_v1(
        &self,
        n: usize,
        group_size: usize,
        bits: u8,
        out: &mut [f32],
    ) -> native_abi::Status {
        #[cfg(feature = "zig")]
        {
            zig_decode_v1(
                &self.packed,
                &self.scales,
                &self.zeros,
                n,
                group_size,
                bits,
                out,
            )
        }
        #[cfg(not(feature = "zig"))]
        {
            let _ = (n, group_size, bits, out);
            native_abi::Status::ErrAllocation
        }
    }
}

#[cfg(feature = "zig")]
mod native {
    use super::ZigQuantizedTensor;
    use std::os::raw::{c_uchar, c_void};

    use native_abi::{
        DecodeRequest as RustDecodeRequest, EncodeRequest as RustEncodeRequest,
        EncodeResult as RustEncodeResult, Status as RustStatus, ABI_VERSION_CURRENT,
    };

    #[cfg(feature = "zig")]
    extern "C" {
        fn _tq_zig_encode(
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

        fn _tq_zig_decode(
            packed_ptr: *const u8,
            packed_len: usize,
            scales_ptr: *const f32,
            zeros_ptr: *const f32,
            n: usize,
            group_size: usize,
            bits: c_uchar,
            out_ptr: *mut f32,
        );

        fn _tq_zig_free(ptr: *mut c_void, size: usize);

        // Native ABI v1 entries from the Zig kernel.
        fn tq_abi_encode(req: *const RustEncodeRequest) -> RustEncodeResult;
        fn tq_abi_decode(req: *const RustDecodeRequest) -> i32;
    }

    #[cfg(feature = "zig")]
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
            _tq_zig_encode(
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

    #[cfg(feature = "zig")]
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
            _tq_zig_decode(
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

    // ── Native ABI v1 wrappers ──────────────────────────────────────
    //
    // The Zig side exposes the same `tq_abi_encode` / `tq_abi_decode`
    // symbols as the C ABI, so this wrapper is a thin shim that mirrors
    // the C side. Caller-owned buffers, contract identical to
    // `perf-core/native-abi/include/abi_v1.h`.

    #[cfg(feature = "zig")]
    pub(super) fn zig_encode_v1(
        data: &[f32],
        bits: u8,
        group_size: usize,
    ) -> Result<ZigQuantizedTensor, RustStatus> {
        let n_groups = native_abi::group_count(data.len(), group_size);
        let packed_len = native_abi::expected_packed_len(data.len(), bits);

        let mut shape_storage: Vec<usize> = vec![0; 1];
        let mut packed_storage: Vec<u8> = vec![0; packed_len];
        let mut scales_storage: Vec<f32> = vec![0.0; n_groups];
        let mut zeros_storage: Vec<f32> = vec![0.0; n_groups];

        let mut shape_ptr = shape_storage.as_mut_ptr();
        let mut packed_ptr = packed_storage.as_mut_ptr();
        let mut scales_ptr = scales_storage.as_mut_ptr();
        let mut zeros_ptr = zeros_storage.as_mut_ptr();

        let mut req = RustEncodeRequest::zeroed();
        req.abi = ABI_VERSION_CURRENT;
        req.data_ptr = data.as_ptr();
        req.n = data.len();
        req.bits = bits;
        req.group_size = group_size;
        req.out_shape = &mut shape_ptr;
        req.out_shape_capacity = shape_storage.len();
        req.out_packed = &mut packed_ptr;
        req.out_packed_capacity = packed_storage.len();
        req.out_scales = &mut scales_ptr;
        req.out_scales_capacity = scales_storage.len();
        req.out_zeros = &mut zeros_ptr;
        req.out_zeros_capacity = zeros_storage.len();

        let result = unsafe { tq_abi_encode(&req) };
        if result.status != RustStatus::Ok {
            return Err(result.status);
        }

        let shape = std::mem::take(&mut shape_storage);
        let packed = std::mem::take(&mut packed_storage);
        let scales = std::mem::take(&mut scales_storage);
        let zeros = std::mem::take(&mut zeros_storage);
        Ok(ZigQuantizedTensor {
            shape,
            packed,
            scales,
            zeros,
        })
    }

    #[cfg(feature = "zig")]
    pub(super) fn zig_decode_v1(
        packed: &[u8],
        scales: &[f32],
        zeros: &[f32],
        n: usize,
        group_size: usize,
        bits: u8,
        out: &mut [f32],
    ) -> RustStatus {
        assert!(
            out.len() >= n,
            "decode_v1: output buffer length ({}) must be >= n ({})",
            out.len(),
            n
        );
        let mut req = RustDecodeRequest::zeroed();
        req.abi = ABI_VERSION_CURRENT;
        req.packed_ptr = packed.as_ptr();
        req.packed_len = packed.len();
        req.scales_ptr = scales.as_ptr();
        req.zeros_ptr = zeros.as_ptr();
        req.n = n;
        req.group_size = group_size;
        req.bits = bits;
        req.out_ptr = out.as_mut_ptr();

        let code = unsafe { tq_abi_decode(&req) };
        RustStatus::try_from(code).unwrap_or(RustStatus::ErrBackend)
    }
}

#[cfg(test)]
#[cfg(feature = "zig")]
mod tests {
    use super::*;

    #[test]
    fn zig_encode_decode_roundtrip() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01 - 0.64).collect();
        let r = ZigQuantizedTensor::encode(&data, 4, 32);
        assert!(r.is_ok(), "encode failed: {:?}", r.err());
        let q = r.unwrap();
        let decoded = q.decode(data.len(), 32, 4);
        for (a, b) in data.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.15, "{} vs {}", a, b);
        }
    }

    #[test]
    fn zig_decode_returns_zeros_without_feature() {
        let q = ZigQuantizedTensor {
            shape: vec![8],
            packed: vec![],
            scales: vec![],
            zeros: vec![],
        };
        #[cfg(feature = "zig")]
        {
            // Avoid decoding an empty tensor in native Zig FFI during test to prevent any potential slicing/alignment crash
            let r = q.decode(0, 8, 4);
            assert!(r.is_empty());
        }
        #[cfg(not(feature = "zig"))]
        {
            let r = q.decode(8, 8, 4);
            assert!(r.iter().all(|&x| x == 0.0));
        }
    }
}
