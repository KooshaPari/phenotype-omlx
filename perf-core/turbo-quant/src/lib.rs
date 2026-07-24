//! TurboQuant quantization kernels — Apple `arm64` + x86_64 SIMD.
//!
//! Implements the Metal-KV-cache quantization pipeline in Rust so that
//! quantization can be replayed off-device for benchmarking, training, and
//! CPU backends. Apple Metal implementation is provided as a Metal `.metallib`
//! in `shaders/turbo_quant.metallib` and consumed by `perf-core/spec-decode`'s
//! optional `metal` feature.

pub mod candidates;
pub mod echokv;
mod minmax;

use minmax::min_max;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurboMode {
    /// Asymmetric: K kept at FP16, V at 4-bit.
    Asymmetric4,
    /// Symmetric: K=V=4-bit (turbo4).
    Symmetric4,
    /// Symmetric: K=V=3-bit (turbo3).
    Symmetric3,
    /// Symmetric: K=V=2-bit (turbo2).
    Symmetric2,
}

impl TurboMode {
    pub fn bits(self) -> u8 {
        match self {
            TurboMode::Asymmetric4 | TurboMode::Symmetric4 => 4,
            TurboMode::Symmetric3 => 3,
            TurboMode::Symmetric2 => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            TurboMode::Asymmetric4 => "asym4 (K=FP16, V=4bit)",
            TurboMode::Symmetric4 => "sym4 (K=V=4bit)",
            TurboMode::Symmetric3 => "sym3 (K=V=3bit)",
            TurboMode::Symmetric2 => "sym2 (K=V=2bit)",
        }
    }
}

/// Quantization parameters applied to the entire model.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub mode: TurboMode,
    pub skip_last_n: usize,
    pub group_size: usize,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            mode: TurboMode::Asymmetric4,
            skip_last_n: 2,
            group_size: 64,
        }
    }
}

/// A single quantized tensor — implements `encode`/`decode` over a `&[f32]`
/// slice. This is a CPU fallback; the Metal path uses the `.metallib` kernels.
///
/// The `bits` and `group_size` fields are part of the on-disk / wire format:
/// decode reads them rather than guessing from the packed byte length, so
/// 2/3-bit and non-64 group encodings round-trip correctly.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub shape: Vec<usize>,
    pub bits: u8,
    pub group_size: usize,
    pub packed: Vec<u8>,
    pub scales: Vec<f32>,
    pub zeros: Vec<f32>,
}

impl QuantizedTensor {
    /// Round-to-nearest uniform quantization — fast, ~1.5% PPL hit vs Lloyd-Max.
    ///
    /// Validates programmer inputs:
    /// - `bits` must be in `2..=4`
    /// - `group_size` must be > 0
    /// - `data` must be non-empty and finite
    ///
    /// Panics on violation rather than silently corrupting the output.
    pub fn encode_uniform(data: &[f32], bits: u8, group_size: usize) -> Self {
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

        Self {
            shape: vec![data.len()],
            bits,
            group_size,
            packed,
            scales,
            zeros,
        }
    }

    /// Decode packed bytes back to f32 — symmetric recovery.
    ///
    /// Uses the `bits` and `group_size` fields stored on the tensor so that
    /// 2/3-bit and non-64-group encodings decode correctly. Out-of-range
    /// metadata or mismatched output length panics with a descriptive
    /// message instead of silently producing wrong numbers.
    pub fn decode_uniform(&self, out: &mut [f32]) {
        assert!(
            (2..=4).contains(&self.bits),
            "turbo-quant: stored bits must be in 2..=4, got {}",
            self.bits
        );
        assert!(
            self.group_size > 0,
            "turbo-quant: stored group_size must be > 0, got {}",
            self.group_size
        );
        let expected = self.shape.iter().product::<usize>();
        assert!(
            out.len() == expected,
            "turbo-quant: decode_uniform output length mismatch — expected \
                 {expected}, got {}",
            out.len()
        );

        let bits = self.bits;
        let qmax = ((1u32 << bits) - 1) as f32;
        let bp = bits as usize;

        for (i, v) in out.iter_mut().enumerate() {
            let g = i / self.group_size;
            let s = self.scales.get(g).copied().unwrap_or(1e-12);
            let z = self.zeros.get(g).copied().unwrap_or(0.0);

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
                let byte = self.packed.get(byte_idx).copied().unwrap_or(0) as u32;
                raw |= ((byte >> bit_off) & mask) << shift;
                shift += take;
                bc += take;
                remaining -= take;
            }

            let q = (raw as f32).min(qmax);
            *v = q * s + z;
        }
    }
}

/// Free function — encode the whole model at once.
pub fn quantize_tensor(data: &[f32], cfg: &QuantConfig) -> QuantizedTensor {
    QuantizedTensor::encode_uniform(data, cfg.mode.bits(), cfg.group_size)
}

/// Trait for unified quantization backends (CPU, C, Zig, Nim, Mojo).
pub trait PolyglotQuantizer {
    /// Encode a floating-point slice using the specified bit width and group size.
    fn encode(&self, data: &[f32], bits: u8, group_size: usize) -> Result<QuantizedTensor, String>;

    /// Decode a quantized tensor into the caller-provided output slice `out`.
    fn decode(&self, tensor: &QuantizedTensor, out: &mut [f32]) -> Result<(), String>;
}

/// Dispatcher across polyglot FFI backend implementations (Cpu, C, Zig, Nim, Mojo).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolyglotBackend {
    Cpu,
    C,
    Zig,
    Nim,
    Mojo,
}

impl PolyglotBackend {
    pub fn all() -> &'static [PolyglotBackend] {
        &[
            PolyglotBackend::Cpu,
            PolyglotBackend::C,
            PolyglotBackend::Zig,
            PolyglotBackend::Nim,
            PolyglotBackend::Mojo,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            PolyglotBackend::Cpu => "Cpu",
            PolyglotBackend::C => "C",
            PolyglotBackend::Zig => "Zig",
            PolyglotBackend::Nim => "Nim",
            PolyglotBackend::Mojo => "Mojo",
        }
    }
}

impl PolyglotQuantizer for PolyglotBackend {
    fn encode(&self, data: &[f32], bits: u8, group_size: usize) -> Result<QuantizedTensor, String> {
        match self {
            PolyglotBackend::Cpu => {
                if data.is_empty() {
                    return Err("turbo-quant: data must be non-empty".to_string());
                }
                if !(2..=4).contains(&bits) {
                    return Err(format!("turbo-quant: bits must be 2..=4, got {bits}"));
                }
                if group_size == 0 {
                    return Err("turbo-quant: group_size must be > 0".to_string());
                }
                if data.iter().any(|v| !v.is_finite()) {
                    return Err("turbo-quant: data contains non-finite values".to_string());
                }
                Ok(QuantizedTensor::encode_uniform(data, bits, group_size))
            }
            PolyglotBackend::C => {
                let c_res = turbo_quant_c::encode_v1(data, bits, group_size);
                match c_res {
                    Ok(ct) => Ok(QuantizedTensor {
                        shape: ct.shape,
                        bits,
                        group_size,
                        packed: ct.packed,
                        scales: ct.scales,
                        zeros: ct.zeros,
                    }),
                    Err(st) => Err(format!("C backend encode failed with status {:?}", st)),
                }
            }
            PolyglotBackend::Zig => {
                let z_res = turbo_quant_zig::ZigQuantizedTensor::encode_v1(data, bits, group_size);
                match z_res {
                    Ok(zt) => Ok(QuantizedTensor {
                        shape: zt.shape,
                        bits,
                        group_size,
                        packed: zt.packed,
                        scales: zt.scales,
                        zeros: zt.zeros,
                    }),
                    Err(st) => Err(format!("Zig backend encode failed with status {:?}", st)),
                }
            }
            PolyglotBackend::Nim => {
                let n_res = turbo_quant_nim::NimQuantizedTensor::encode(data, bits, group_size);
                match n_res {
                    Ok(nt) => Ok(QuantizedTensor {
                        shape: nt.shape,
                        bits,
                        group_size,
                        packed: nt.packed,
                        scales: nt.scales,
                        zeros: nt.zeros,
                    }),
                    Err(err) => Err(format!("Nim backend encode failed: {err}")),
                }
            }
            PolyglotBackend::Mojo => {
                let m_res = turbo_quant_mojo::MojoQuantizedTensor::encode(data, bits, group_size);
                match m_res {
                    Ok(mt) => Ok(QuantizedTensor {
                        shape: mt.shape,
                        bits,
                        group_size,
                        packed: mt.packed,
                        scales: mt.scales,
                        zeros: mt.zeros,
                    }),
                    Err(err) => Err(format!("Mojo backend encode failed: {err}")),
                }
            }
        }
    }

    fn decode(&self, tensor: &QuantizedTensor, out: &mut [f32]) -> Result<(), String> {
        let expected_len = tensor.shape.iter().product::<usize>();
        if out.len() != expected_len {
            return Err(format!(
                "output slice length ({}) does not match tensor element count ({})",
                out.len(),
                expected_len
            ));
        }

        match self {
            PolyglotBackend::Cpu => {
                tensor.decode_uniform(out);
                Ok(())
            }
            PolyglotBackend::C => {
                let status = turbo_quant_c::decode_v1(
                    &tensor.packed,
                    &tensor.scales,
                    &tensor.zeros,
                    expected_len,
                    tensor.group_size,
                    tensor.bits,
                    out,
                );
                if status == native_abi::Status::Ok {
                    Ok(())
                } else {
                    Err(format!("C backend decode failed with status {:?}", status))
                }
            }
            PolyglotBackend::Zig => {
                let zt = turbo_quant_zig::ZigQuantizedTensor {
                    shape: tensor.shape.clone(),
                    packed: tensor.packed.clone(),
                    scales: tensor.scales.clone(),
                    zeros: tensor.zeros.clone(),
                };
                let status = zt.decode_v1(expected_len, tensor.group_size, tensor.bits, out);
                if status == native_abi::Status::Ok {
                    Ok(())
                } else {
                    Err(format!("Zig backend decode failed with status {:?}", status))
                }
            }
            PolyglotBackend::Nim => {
                let status = turbo_quant_nim::decode_v1(
                    &tensor.packed,
                    &tensor.scales,
                    &tensor.zeros,
                    expected_len,
                    tensor.group_size,
                    tensor.bits,
                    out,
                );
                if status == native_abi::Status::Ok {
                    Ok(())
                } else {
                    Err(format!("Nim backend decode failed with status {:?}", status))
                }
            }
            PolyglotBackend::Mojo => {
                let mt = turbo_quant_mojo::MojoQuantizedTensor {
                    shape: tensor.shape.clone(),
                    packed: tensor.packed.clone(),
                    scales: tensor.scales.clone(),
                    zeros: tensor.zeros.clone(),
                };
                match mt.try_decode(expected_len, tensor.group_size, tensor.bits) {
                    Ok(decoded) => {
                        out.copy_from_slice(&decoded);
                        Ok(())
                    }
                    Err(err) => Err(format!("Mojo backend decode failed: {err}")),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tolerance_for(bits: u8, range: f32) -> f32 {
        // ~1.5 LSBs in the quantised domain, scaled by range.
        let qmax = ((1u32 << bits) - 1) as f32;
        2.0 * range / qmax
    }

    #[test]
    fn roundtrip_uniform() {
        let data: Vec<f32> = (0..512).map(|i| (i as f32) * 0.01).collect();
        let q = QuantizedTensor::encode_uniform(&data, 4, 64);
        let mut out = vec![0f32; data.len()];
        q.decode_uniform(&mut out);
        for (a, b) in data.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() < 0.1,
                "{} vs {} (delta={})",
                a,
                b,
                (a - b).abs()
            );
        }
    }

    #[test]
    fn mode_bits() {
        assert_eq!(TurboMode::Asymmetric4.bits(), 4);
        assert_eq!(TurboMode::Symmetric3.bits(), 3);
        assert_eq!(TurboMode::Symmetric2.bits(), 2);
    }

    fn linspace(n: usize, lo: f32, hi: f32) -> Vec<f32> {
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![lo];
        }
        let step = (hi - lo) / (n as f32 - 1.0);
        (0..n).map(|i| lo + step * i as f32).collect()
    }

    #[test]
    fn roundtrip_bits_2_group_3() {
        let data = linspace(10, -1.0, 1.0);
        let q = QuantizedTensor::encode_uniform(&data, 2, 3);
        assert_eq!(q.bits, 2);
        assert_eq!(q.group_size, 3);
        let mut out = vec![0f32; data.len()];
        q.decode_uniform(&mut out);
        let range = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - data.iter().cloned().fold(f32::INFINITY, f32::min);
        let tol = tolerance_for(2, range).max(1e-5);
        for (a, b) in data.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() <= tol + 1e-6,
                "{} vs {} (delta={})",
                a,
                b,
                (a - b).abs()
            );
        }
    }

    #[test]
    fn roundtrip_bits_3_group_7() {
        let data = linspace(20, -2.0, 2.0);
        let q = QuantizedTensor::encode_uniform(&data, 3, 7);
        assert_eq!(q.bits, 3);
        assert_eq!(q.group_size, 7);
        let mut out = vec![0f32; data.len()];
        q.decode_uniform(&mut out);
        let range = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - data.iter().cloned().fold(f32::INFINITY, f32::min);
        let tol = tolerance_for(3, range).max(1e-5);
        for (a, b) in data.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() <= tol + 1e-6,
                "{} vs {} (delta={})",
                a,
                b,
                (a - b).abs()
            );
        }
    }

    #[test]
    fn roundtrip_bits_4_group_64() {
        let data = linspace(257, -3.0, 3.0);
        let q = QuantizedTensor::encode_uniform(&data, 4, 64);
        assert_eq!(q.bits, 4);
        assert_eq!(q.group_size, 64);
        let mut out = vec![0f32; data.len()];
        q.decode_uniform(&mut out);
        let range = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - data.iter().cloned().fold(f32::INFINITY, f32::min);
        let tol = tolerance_for(4, range).max(1e-5);
        for (a, b) in data.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() <= tol + 1e-6,
                "{} vs {} (delta={})",
                a,
                b,
                (a - b).abs()
            );
        }
    }

    #[test]
    fn metadata_preserved_through_encode() {
        for &(bits, group_size) in &[(2u8, 3usize), (3, 7), (4, 64), (2, 64), (4, 1)] {
            let data = linspace(13, -1.0, 1.0);
            let q = QuantizedTensor::encode_uniform(&data, bits, group_size);
            assert_eq!(q.bits, bits, "bits mismatch for ({}, {})", bits, group_size);
            assert_eq!(
                q.group_size, group_size,
                "group_size mismatch for ({}, {})",
                bits, group_size
            );
            // scale/zero length must equal ceil(n / group_size)
            let expected_groups = data.len().div_ceil(group_size);
            assert_eq!(q.scales.len(), expected_groups);
            assert_eq!(q.zeros.len(), expected_groups);
        }
    }

    #[test]
    fn roundtrip_partial_trailing_group() {
        // 10 elements, group_size=3 → groups of [3,3,3,1] — last is partial.
        let data: Vec<f32> = (0..10).map(|i| i as f32 * 0.25 - 1.0).collect();
        let q = QuantizedTensor::encode_uniform(&data, 4, 3);
        assert_eq!(q.group_size, 3);
        let mut out = vec![0f32; data.len()];
        q.decode_uniform(&mut out);
        for (a, b) in data.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() < 0.1,
                "{} vs {} (delta={})",
                a,
                b,
                (a - b).abs()
            );
        }
    }

    #[test]
    fn encode_validates_programmer_inputs() {
        // bits out of range — panic, do not silently corrupt.
        let data = vec![1.0f32, 2.0, 3.0];
        let result = std::panic::catch_unwind(|| {
            QuantizedTensor::encode_uniform(&data, 5, 4);
        });
        assert!(result.is_err(), "bits=5 should panic");
        // group_size == 0 — panic.
        let result = std::panic::catch_unwind(|| {
            QuantizedTensor::encode_uniform(&data, 4, 0);
        });
        assert!(result.is_err(), "group_size=0 should panic");
        // Non-finite input — panic.
        let bad = vec![1.0f32, f32::NAN, 3.0];
        let result = std::panic::catch_unwind(|| {
            QuantizedTensor::encode_uniform(&bad, 4, 2);
        });
        assert!(result.is_err(), "NaN input should panic");
    }

    #[test]
    fn polyglot_backends_roundtrip() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.02 - 1.28).collect();
        for &backend in PolyglotBackend::all() {
            let encoded = backend.encode(&data, 4, 32);
            assert!(
                encoded.is_ok(),
                "Backend {} failed encode: {:?}",
                backend.name(),
                encoded.err()
            );
            let tensor = encoded.unwrap();
            let mut decoded = vec![0.0f32; data.len()];
            let dec_res = backend.decode(&tensor, &mut decoded);
            assert!(
                dec_res.is_ok(),
                "Backend {} failed decode: {:?}",
                backend.name(),
                dec_res.err()
            );
            for (a, b) in data.iter().zip(decoded.iter()) {
                assert!(
                    (a - b).abs() < 0.25,
                    "Backend {} mismatch: {} vs {}",
                    backend.name(),
                    a,
                    b
                );
            }
        }
    }
}
