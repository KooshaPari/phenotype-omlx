//! Candidate builders for turbo-quant kernels.
//!
//! Each [`kernel_registry::Candidate`] returned here advertises
//! `dynamic_eviction = "true"` when the quantization mode integrates
//! with [`super::echokv::EchoKVCache`] for on-the-fly KV cache
//! management.

use kernel_registry::candidate::{BackendKind, Capability};
use kernel_registry::compat::DType;
use kernel_registry::key::ShapeSignature;
use kernel_registry::Candidate;

/// Minimum shape envelope (all zeros — accepts everything).
fn zero_shape() -> ShapeSignature {
    ShapeSignature {
        m: 0,
        n: 0,
        k: 0,
        batch: 0,
        seq: 0,
        group: 0,
    }
}

/// Maximum shape envelope (4096 in every dimension).
fn wide_shape() -> ShapeSignature {
    ShapeSignature {
        m: 4096,
        n: 4096,
        k: 4096,
        batch: 4096,
        seq: 4096,
        group: 64,
    }
}

/// Build an EchoKV-aware quantization candidate.
///
/// The candidate carries `dynamic_eviction = "true"` in its
/// [`Candidate::properties`] so that
/// [`KernelRegistry::select_with_kv_state`](kernel_registry::KernelRegistry::select_with_kv_state)
/// can prefer it when KV cache utilization exceeds 80 %.
fn echokv_candidate(
    name: &str,
    backend: BackendKind,
    source_hash: &str,
    requires: Vec<Capability>,
    dtypes: Vec<DType>,
) -> Candidate {
    Candidate::new(
        name,
        backend,
        source_hash,
        requires,
        zero_shape(),
        wide_shape(),
        dtypes,
        true,
    )
    .with_property("dynamic_eviction", "true")
}

/// Asymmetric4 quantization candidate (K=FP16, V=4-bit) with EchoKV
/// eviction support.
pub fn asymmetric4_echokv(backend: BackendKind, source_hash: &str) -> Candidate {
    echokv_candidate(
        "turbo-quant-asym4-echokv",
        backend,
        source_hash,
        vec![],
        vec![DType::Fp16, DType::Bf16],
    )
}

/// Symmetric4 quantization candidate (K=V=4-bit) with EchoKV eviction
/// support.
pub fn symmetric4_echokv(backend: BackendKind, source_hash: &str) -> Candidate {
    echokv_candidate(
        "turbo-quant-sym4-echokv",
        backend,
        source_hash,
        vec![],
        vec![DType::Fp16, DType::Bf16],
    )
}

/// Symmetric3 quantization candidate (K=V=3-bit) with EchoKV eviction
/// support.
pub fn symmetric3_echokv(backend: BackendKind, source_hash: &str) -> Candidate {
    echokv_candidate(
        "turbo-quant-sym3-echokv",
        backend,
        source_hash,
        vec![],
        vec![DType::Fp16, DType::Bf16],
    )
}

/// Symmetric2 quantization candidate (K=V=2-bit) with EchoKV eviction
/// support.
pub fn symmetric2_echokv(backend: BackendKind, source_hash: &str) -> Candidate {
    echokv_candidate(
        "turbo-quant-sym2-echokv",
        backend,
        source_hash,
        vec![],
        vec![DType::Fp16, DType::Bf16],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_echokv_candidates_have_dynamic_eviction() {
        for cand in [
            asymmetric4_echokv(BackendKind::Metal, "hash-a"),
            symmetric4_echokv(BackendKind::Metal, "hash-b"),
            symmetric3_echokv(BackendKind::Metal, "hash-c"),
            symmetric2_echokv(BackendKind::Metal, "hash-d"),
        ] {
            assert!(
                cand.has_property("dynamic_eviction"),
                "{} must advertise dynamic_eviction",
                cand.name
            );
        }
    }

    #[test]
    fn candidates_support_fp16_and_bf16() {
        let c = asymmetric4_echokv(BackendKind::Cpu, "h");
        assert!(c.supports_dtype(DType::Fp16));
        assert!(c.supports_dtype(DType::Bf16));
    }

    #[test]
    fn candidates_are_tunable() {
        let c = symmetric2_echokv(BackendKind::Cuda, "h");
        assert!(c.tunable);
    }

    #[test]
    fn different_backends_produce_different_ids() {
        let c_cpu = asymmetric4_echokv(BackendKind::Cpu, "h");
        let c_metal = asymmetric4_echokv(BackendKind::Metal, "h");
        assert_ne!(c_cpu.id, c_metal.id);
    }
}
