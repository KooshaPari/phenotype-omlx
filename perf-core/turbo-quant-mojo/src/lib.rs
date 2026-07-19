// turbo-quant-mojo — Rust wrapper for the Mojo implementation.
//
// Requires the Mojo SDK in PATH (`modular install mojo`). build.rs compiles
// mojo-src/turbo_quant.mojo to a shared library and links it unconditionally.

use native::{mojo_decode, mojo_encode};

#[derive(Debug, Clone)]
pub struct MojoQuantizedTensor {
    pub shape: Vec<usize>,
    pub packed: Vec<u8>,
    pub scales: Vec<f32>,
    pub zeros: Vec<f32>,
}

impl MojoQuantizedTensor {
    pub fn encode(data: &[f32], bits: u8, group_size: usize) -> Result<Self, String> {
        mojo_encode(data, bits, group_size)
    }

    pub fn decode(&self, n: usize, group_size: usize, bits: u8) -> Vec<f32> {
        mojo_decode(&self.packed, &self.scales, &self.zeros, n, group_size, bits)
    }
}

mod native {
    use super::MojoQuantizedTensor;
    use std::os::raw::{c_uchar, c_void};

    extern "C" {
        fn tq_mojo_encode(
            data_addr: isize,
            n: usize,
            bits: c_uchar,
            group_size: usize,
            shape_ptr_out: *mut isize,
            out_shape_len: *mut isize,
            packed_ptr_out: *mut isize,
            out_packed_len: *mut isize,
            scales_ptr_out: *mut isize,
            out_scales_len: *mut isize,
            zeros_ptr_out: *mut isize,
            out_zeros_len: *mut isize,
        ) -> bool;

        fn tq_mojo_decode(
            packed_ptr: *const u8,
            packed_len: usize,
            scales_ptr: *const f32,
            zeros_ptr: *const f32,
            n: usize,
            group_size: usize,
            bits: c_uchar,
            out_ptr: *mut f32,
        );

        fn free(ptr: *mut c_void);
    }

    pub(super) fn mojo_encode(
        data: &[f32],
        bits: u8,
        group_size: usize,
    ) -> Result<MojoQuantizedTensor, String> {
        let mut shape_addr: isize = 0;
        let mut shape_len: isize = 0;
        let mut packed_addr: isize = 0;
        let mut packed_len: isize = 0;
        let mut scales_addr: isize = 0;
        let mut scales_len: isize = 0;
        let mut zeros_addr: isize = 0;
        let mut zeros_len: isize = 0;

        let ok = unsafe {
            tq_mojo_encode(
                data.as_ptr() as usize as isize,
                data.len(),
                bits,
                group_size,
                &mut shape_addr,
                &mut shape_len,
                &mut packed_addr,
                &mut packed_len,
                &mut scales_addr,
                &mut scales_len,
                &mut zeros_addr,
                &mut zeros_len,
            )
        };
        if !ok {
            return Err("Mojo tq_mojo_encode returned false".to_string());
        }

        if shape_addr == 0 || packed_addr == 0 || scales_addr == 0 || zeros_addr == 0 {
            return Err(
                "Mojo tq_mojo_encode returned null output pointers — \
                 Mojo @export/out-pointer ABI still broken on this toolchain (TODO: FR-OMLX-POLY-001)"
                    .to_string(),
            );
        }

        let shape_len = shape_len as usize;
        let packed_len = packed_len as usize;
        let scales_len = scales_len as usize;
        let zeros_len = zeros_len as usize;

        let shape_ptr = shape_addr as *mut usize;
        let packed_ptr = packed_addr as *mut u8;
        let scales_ptr = scales_addr as *mut f32;
        let zeros_ptr = zeros_addr as *mut f32;

        let shape = unsafe { std::slice::from_raw_parts(shape_ptr, shape_len) }.to_vec();
        let packed = unsafe { std::slice::from_raw_parts(packed_ptr, packed_len) }.to_vec();
        let scales = unsafe { std::slice::from_raw_parts(scales_ptr, scales_len) }.to_vec();
        let zeros = unsafe { std::slice::from_raw_parts(zeros_ptr, zeros_len) }.to_vec();

        unsafe {
            free(shape_ptr as *mut c_void);
            free(packed_ptr as *mut c_void);
            free(scales_ptr as *mut c_void);
            free(zeros_ptr as *mut c_void);
        }

        Ok(MojoQuantizedTensor {
            shape,
            packed,
            scales,
            zeros,
        })
    }

    pub(super) fn mojo_decode(
        packed: &[u8],
        scales: &[f32],
        zeros: &[f32],
        n: usize,
        group_size: usize,
        bits: u8,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        unsafe {
            tq_mojo_decode(
                packed.as_ptr(),
                packed.len(),
                scales.as_ptr(),
                zeros.as_ptr(),
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
    use std::path::PathBuf;
    use std::process::Command;

    fn mojo_shared_lib_path() -> PathBuf {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let name = if cfg!(target_os = "macos") {
            "libturbo_quant_mojo.dylib"
        } else if cfg!(target_os = "windows") {
            "turbo_quant_mojo.dll"
        } else {
            "libturbo_quant_mojo.so"
        };
        manifest.join(name)
    }

    #[test]
    fn mojo_shared_lib_builds() {
        let lib = mojo_shared_lib_path();
        assert!(
            lib.exists(),
            "Mojo shared library missing at {} — build.rs compile gate failed",
            lib.display()
        );
    }

    #[test]
    fn mojo_smoke_script_roundtrips() {
        let mojo = std::env::var_os("MOJO_PATH")
            .map(PathBuf::from)
            .or_else(which)
            .expect("mojo not found in PATH — install with: modular install mojo");
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let smoke = manifest.join("mojo-src/turbo_quant_smoke.mojo");
        let status = Command::new(mojo)
            .arg("run")
            .arg(smoke.file_name().expect("smoke filename"))
            .current_dir(manifest.join("mojo-src"))
            .status()
            .expect("spawn mojo run smoke");
        assert!(status.success(), "mojo smoke script failed");
    }

    #[test]
    fn mojo_encode_decode_roundtrip_todo_fr_omlx_poly_001_null_out_pointers() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01 - 0.64).collect();
        match MojoQuantizedTensor::encode(&data, 4, 32) {
            Ok(q) => {
                let decoded = q.decode(data.len(), 32, 4);
                for (a, b) in data.iter().zip(decoded.iter()) {
                    assert!((a - b).abs() < 0.15, "roundtrip mismatch: {a} vs {b}");
                }
            }
            Err(e) if e.contains("null output pointers") => {
                panic!(
                    "TODO FR-OMLX-POLY-001: Mojo @export out-pointer ABI returns null on 1.0.0b3 — {e}"
                );
            }
            Err(e) => panic!("Mojo encode failed unexpectedly: {e}"),
        }
    }

    fn which() -> Option<PathBuf> {
        std::env::var_os("PATH").and_then(|path| {
            std::env::split_paths(&path).find_map(|dir| {
                let candidate = dir.join("mojo");
                if candidate.is_file() {
                    Some(candidate)
                } else {
                    None
                }
            })
        })
    }
}
