// turbo-quant-mojo — Rust wrapper for the Mojo implementation.
//
// Mojo can compile to a shared object or static library that exposes a
// C ABI; this crate wraps that and gives a Rust-friendly API matching
// perf-core/turbo-quant.
//
// Without `--features mojo`, this crate compiles to a no-op stub so the
// workspace always builds, even when `mojo` is not installed.

#[derive(Debug, Clone)]
pub struct MojoQuantizedTensor {
    pub shape: Vec<usize>,
    pub packed: Vec<u8>,
    pub scales: Vec<f32>,
    pub zeros: Vec<f32>,
}

impl MojoQuantizedTensor {
    pub fn encode(data: &[f32], bits: u8, group_size: usize) -> Result<Self, String> {
        #[cfg(feature = "mojo")]
        {
            mojo_encode(data, bits, group_size)
        }
        #[cfg(not(feature = "mojo"))]
        {
            let _ = (data, bits, group_size);
            Err("turbo-quant-mojo: built without --features mojo".to_string())
        }
    }

    pub fn decode(&self, n: usize, group_size: usize, bits: u8) -> Vec<f32> {
        #[cfg(feature = "mojo")]
        {
            mojo_decode(&self.packed, &self.scales, &self.zeros, n, group_size, bits)
        }
        #[cfg(not(feature = "mojo"))]
        {
            let _ = (n, group_size, bits);
            vec![0.0; n]
        }
    }
}

#[cfg(feature = "mojo")]
mod native {
    use super::MojoQuantizedTensor;
    use std::os::raw::c_uchar;

    extern "C" {
        fn tq_mojo_encode(
            data_ptr: *const f32, n: usize, bits: c_uchar, group_size: usize,
            out_shape: *mut *mut usize, out_shape_len: *mut usize,
            out_packed: *mut *mut u8, out_packed_len: *mut usize,
            out_scales: *mut *mut f32, out_scales_len: *mut usize,
            out_zeros: *mut *mut f32, out_zeros_len: *mut usize,
        ) -> bool;

        fn tq_mojo_decode(
            packed_ptr: *const u8, packed_len: usize,
            scales_ptr: *const f32, zeros_ptr: *const f32,
            n: usize, group_size: usize, bits: c_uchar,
            out_ptr: *mut f32,
        );

        fn tq_mojo_free(address: isize);
    }

    pub(super) fn mojo_encode(
        data: &[f32], bits: u8, group_size: usize,
    ) -> Result<MojoQuantizedTensor, String> {
        let mut shape_ptr: *mut usize = std::ptr::null_mut();
        let mut shape_len: usize = 0;
        let mut packed_ptr: *mut u8 = std::ptr::null_mut();
        let mut packed_len: usize = 0;
        let mut scales_ptr: *mut f32 = std::ptr::null_mut();
        let mut scales_len: usize = 0;
        let mut zeros_ptr: *mut f32 = std::ptr::null_mut();
        let mut zeros_len: usize = 0;

        let ok = unsafe {
            tq_mojo_encode(
                data.as_ptr(), data.len(), bits, group_size,
                &mut shape_ptr, &mut shape_len,
                &mut packed_ptr, &mut packed_len,
                &mut scales_ptr, &mut scales_len,
                &mut zeros_ptr, &mut zeros_len,
            )
        };
        if !ok { return Err("Mojo tq_mojo_encode returned false".to_string()); }

        let shape  = unsafe { std::slice::from_raw_parts(shape_ptr,  shape_len)  }.to_vec();
        let packed = unsafe { std::slice::from_raw_parts(packed_ptr, packed_len) }.to_vec();
        let scales = unsafe { std::slice::from_raw_parts(scales_ptr, scales_len) }.to_vec();
        let zeros  = unsafe { std::slice::from_raw_parts(zeros_ptr,  zeros_len)  }.to_vec();

        unsafe {
            tq_mojo_free(shape_addr);
            tq_mojo_free(packed_addr);
            tq_mojo_free(scales_addr);
            tq_mojo_free(zeros_addr);
        }

        Ok(MojoQuantizedTensor { shape, packed, scales, zeros })
    }

    pub(super) fn mojo_decode(
        packed: &[u8], scales: &[f32], zeros: &[f32],
        n: usize, group_size: usize, bits: u8,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        unsafe {
            tq_mojo_decode(
                packed.as_ptr(), packed.len(),
                scales.as_ptr(), zeros.as_ptr(),
                n, group_size, bits,
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
    fn mojo_stub_returns_error_without_feature() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01 - 0.64).collect();
        let r = MojoQuantizedTensor::encode(&data, 4, 32);
        #[cfg(feature = "mojo")]
        assert!(r.is_ok(), "encode failed: {:?}", r.err());
        #[cfg(not(feature = "mojo"))]
        assert!(r.is_err(), "encode should fail without --features mojo");
    }
}
