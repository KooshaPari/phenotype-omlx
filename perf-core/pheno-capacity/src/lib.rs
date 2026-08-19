//! Pure capacity math — embedded into the synthetic monolith (ADR-006).
//! Replaces deleted `KooshaPari/pheno-capacity` public API surface.

#![deny(unsafe_code)]

/// Parameter element width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
    I8,
    I4,
}

impl Dtype {
    pub fn bytes(self) -> f64 {
        match self {
            Dtype::F32 => 4.0,
            Dtype::F16 | Dtype::Bf16 => 2.0,
            Dtype::I8 => 1.0,
            Dtype::I4 => 0.5,
        }
    }
}

/// Training optimizer memory multiplier relative to weight bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Optimizer {
    AdamW,
    LoRA,
    QLoRA,
    Adafactor,
}

impl Optimizer {
    pub fn weight_multiplier(self) -> f64 {
        match self {
            Optimizer::AdamW => 8.0,
            Optimizer::LoRA | Optimizer::QLoRA => 0.0,
            Optimizer::Adafactor => 1.0,
        }
    }
}

/// Inference weight memory in bytes (weights only; no KV/activations).
pub fn vram_estimate(params: u64, dtype: Dtype) -> u64 {
    ((params as f64) * dtype.bytes()).round() as u64
}

pub fn model_fits_in(params: u64, available_bytes: u64, dtype: Dtype) -> bool {
    vram_estimate(params, dtype) <= available_bytes
}

pub fn optimizer_state_vram(weights_bytes: u64, optimizer: Optimizer) -> u64 {
    ((weights_bytes as f64) * optimizer.weight_multiplier()).round() as u64
}

/// Chinchilla-style optimal tokens ≈ ratio × params (default ratio 20).
pub fn chinchilla_tokens(params: u64, ratio: f64) -> u64 {
    ((params as f64) * ratio).round() as u64
}

pub fn dtype_bytes(dtype: Dtype) -> f64 {
    dtype.bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mistral_7b_f16_approx_14gb() {
        let bytes = vram_estimate(7_240_000_000, Dtype::F16);
        // Decimal GB (docs table); GiB ≈ 13.49
        let gb = bytes as f64 / 1_000_000_000.0;
        assert!((gb - 14.48).abs() < 0.05, "got {gb} decimal GB");
    }

    #[test]
    fn fits_16gb() {
        assert!(model_fits_in(7_000_000_000, 16 * 1024 * 1024 * 1024, Dtype::F16));
        assert!(!model_fits_in(70_000_000_000, 16 * 1024 * 1024 * 1024, Dtype::F16));
    }

    #[test]
    fn adamw_is_8x() {
        assert_eq!(optimizer_state_vram(1_000, Optimizer::AdamW), 8_000);
    }

    #[test]
    fn chinchilla_default() {
        assert_eq!(chinchilla_tokens(1_000_000_000, 20.0), 20_000_000_000);
    }
}
