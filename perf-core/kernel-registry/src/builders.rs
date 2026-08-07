//! Builders that bridge [`crate::KernelKey`] construction from a
//! model-plan operator description.
//!
//! These helpers exist so the runtime never has to hand-construct a
//! `KernelKey` with magic numbers when dispatching a known operator
//! family (sliding-window GQA, batched DeltaNet, single-chunk DeltaNet).
//! All builders return a fully-formed `KernelKey` whose
//! `quantization = QuantizationPolicy::None` and
//! `state_layout_version = 1` — these are the defaults for the
//! unquantized reference layout the kernel-registry ships with today.
//!
//! Every builder panics with a structured message on invalid input.
//! Panicking is deliberate: builders are called from runtime hot paths
//! where a malformed argument signals a programmer error upstream
//! (e.g. wrong head_dim from a model-plan reader) — failing loud beats
//! silently returning a degenerate key that the selector would just
//! fall back from.

use crate::compat::{AttentionKind, DType, OperatorKind, QuantizationPolicy};
use crate::key::{KernelKey, ShapeSignature};

/// Build a [`KernelKey`] for a SlidingWindow attention layer.
///
/// Encoding (matches the Qwen3-Next sliding-window contract pinned in
/// `tests/sota_operators/attention_sliding_window.rs`):
///
/// - `m = q_heads * head_dim / 8` (packed byte-row count)
/// - `n = group_size`
/// - `k = kv_heads`
/// - `batch = batch_size`
/// - `seq = seq_len`
/// - `group = window_size` clamped to `[1, seq_len]` so a degenerate
///   `window_size = 0` key still matches scalar fallbacks
///
/// # Panics
/// Panics if `q_heads`, `kv_heads`, `head_dim`, `batch_size`, `seq_len`,
/// or `group_size` is zero. `window_size == 0` is silently clamped to 1
/// so the key remains usable.
#[allow(clippy::too_many_arguments)]
pub fn sliding_window_key(
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    batch_size: usize,
    seq_len: usize,
    group_size: usize,
    window_size: usize,
    dtype: DType,
    device_fingerprint: &str,
    policy_version: u32,
) -> KernelKey {
    assert!(q_heads > 0, "sliding_window_key: q_heads must be > 0");
    assert!(kv_heads > 0, "sliding_window_key: kv_heads must be > 0");
    assert!(head_dim > 0, "sliding_window_key: head_dim must be > 0");
    assert!(batch_size > 0, "sliding_window_key: batch_size must be > 0");
    assert!(seq_len > 0, "sliding_window_key: seq_len must be > 0");
    assert!(group_size > 0, "sliding_window_key: group_size must be > 0");

    let m = q_heads * head_dim / 8;
    let n = group_size;
    let k = kv_heads;
    let group = window_size.clamp(1, seq_len);

    KernelKey {
        operator_kind: OperatorKind::Attention,
        attention_kind: Some(AttentionKind::Gqa),
        shape_signature: ShapeSignature {
            m,
            n,
            k,
            batch: batch_size,
            seq: seq_len,
            group,
        },
        dtype,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: device_fingerprint.to_string(),
        policy_version,
    }
}

/// Build a [`KernelKey`] for a batched DeltaNet layer
/// (Qwen3-Coder-Next style hybrid).
///
/// Encoding (matches the contract pinned in
/// `tests/sota_operators/recurrent.rs::deltanet_batched_key`):
///
/// - `m = head_dim`
/// - `n = head_dim`
/// - `k = head_dim`
/// - `batch = batch_size`
/// - `seq = chunk_size`
/// - `group = num_heads`
///
/// # Panics
/// Panics if any of `batch_size`, `num_heads`, `chunk_size`, or
/// `head_dim` is zero.
#[allow(clippy::too_many_arguments)]
pub fn deltanet_batched_key(
    batch_size: usize,
    num_heads: usize,
    chunk_size: usize,
    head_dim: usize,
    dtype: DType,
    device_fingerprint: &str,
    policy_version: u32,
) -> KernelKey {
    assert!(
        batch_size > 0,
        "deltanet_batched_key: batch_size must be > 0"
    );
    assert!(num_heads > 0, "deltanet_batched_key: num_heads must be > 0");
    assert!(
        chunk_size > 0,
        "deltanet_batched_key: chunk_size must be > 0"
    );
    assert!(head_dim > 0, "deltanet_batched_key: head_dim must be > 0");

    KernelKey {
        operator_kind: OperatorKind::DeltaNet,
        attention_kind: None,
        shape_signature: ShapeSignature {
            m: head_dim,
            n: head_dim,
            k: head_dim,
            batch: batch_size,
            seq: chunk_size,
            group: num_heads,
        },
        dtype,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: device_fingerprint.to_string(),
        policy_version,
    }
}

/// Build a [`KernelKey`] for a single-chunk DeltaNet layer
/// (canonical Mamba-style state update).
///
/// Encoding (matches the Mamba selective-scan shape pinned in
/// `tests/sota_operators/recurrent.rs::mamba_key`):
///
/// - `m = head_dim`
/// - `n = head_dim`
/// - `k = head_dim`
/// - `batch = 1`
/// - `seq = chunk_size`
/// - `group = 1`
///
/// # Panics
/// Panics if `head_dim` or `chunk_size` is zero.
pub fn deltanet_key(
    head_dim: usize,
    chunk_size: usize,
    dtype: DType,
    device_fingerprint: &str,
    policy_version: u32,
) -> KernelKey {
    assert!(head_dim > 0, "deltanet_key: head_dim must be > 0");
    assert!(chunk_size > 0, "deltanet_key: chunk_size must be > 0");

    KernelKey {
        operator_kind: OperatorKind::DeltaNet,
        attention_kind: None,
        shape_signature: ShapeSignature {
            m: head_dim,
            n: head_dim,
            k: head_dim,
            batch: 1,
            seq: chunk_size,
            group: 1,
        },
        dtype,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: device_fingerprint.to_string(),
        policy_version,
    }
}

/// Build a [`KernelKey`] for a RetNet retention step.
///
/// The recurrent state is indexed by `(batch, heads, head_dim, head_dim)`;
/// `seq` records the number of tokens in the dispatched chunk and `group`
/// records the head count.  Keeping this as `OperatorKind::Recurrent` avoids
/// introducing a new serialized enum variant while still distinguishing the
/// shape from RWKV-style recurrent candidates.
pub fn retnet_key(
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    chunk_size: usize,
    dtype: DType,
    device_fingerprint: &str,
    policy_version: u32,
) -> KernelKey {
    assert!(batch_size > 0, "retnet_key: batch_size must be > 0");
    assert!(num_heads > 0, "retnet_key: num_heads must be > 0");
    assert!(head_dim > 0, "retnet_key: head_dim must be > 0");
    assert!(chunk_size > 0, "retnet_key: chunk_size must be > 0");

    KernelKey {
        operator_kind: OperatorKind::Recurrent,
        attention_kind: None,
        shape_signature: ShapeSignature {
            m: head_dim,
            n: head_dim,
            k: head_dim,
            batch: batch_size,
            seq: chunk_size,
            group: num_heads,
        },
        dtype,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: device_fingerprint.to_string(),
        policy_version,
    }
}

/// Build a [`KernelKey`] for a chunked Mamba selective scan.
pub fn mamba_scan_key(
    batch_size: usize,
    state_dim: usize,
    chunk_size: usize,
    dtype: DType,
    device_fingerprint: &str,
    policy_version: u32,
) -> KernelKey {
    assert!(batch_size > 0, "mamba_scan_key: batch_size must be > 0");
    assert!(state_dim > 0, "mamba_scan_key: state_dim must be > 0");
    assert!(chunk_size > 0, "mamba_scan_key: chunk_size must be > 0");

    KernelKey {
        operator_kind: OperatorKind::Scan,
        attention_kind: None,
        shape_signature: ShapeSignature {
            m: state_dim,
            n: state_dim,
            k: state_dim,
            batch: batch_size,
            seq: chunk_size,
            group: 1,
        },
        dtype,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: device_fingerprint.to_string(),
        policy_version,
    }
}

/// Build a [`KernelKey`] for a Mixture-of-Experts grouped GEMM layer.
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_gemm_key(
    m: usize,
    n: usize,
    k: usize,
    batch_size: usize,
    seq_len: usize,
    num_experts: usize,
    dtype: DType,
    device_fingerprint: &str,
    policy_version: u32,
) -> KernelKey {
    assert!(m > 0, "moe_grouped_gemm_key: m must be > 0");
    assert!(n > 0, "moe_grouped_gemm_key: n must be > 0");
    assert!(k > 0, "moe_grouped_gemm_key: k must be > 0");
    assert!(
        batch_size > 0,
        "moe_grouped_gemm_key: batch_size must be > 0"
    );
    assert!(seq_len > 0, "moe_grouped_gemm_key: seq_len must be > 0");
    assert!(
        num_experts > 0,
        "moe_grouped_gemm_key: num_experts must be > 0"
    );

    KernelKey {
        operator_kind: OperatorKind::Moe,
        attention_kind: None,
        shape_signature: ShapeSignature {
            m,
            n,
            k,
            batch: batch_size,
            seq: seq_len,
            group: num_experts,
        },
        dtype,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: device_fingerprint.to_string(),
        policy_version,
    }
}

/// Build a [`KernelKey`] for a packed ternary GEMM (Bonsai layout).
#[allow(clippy::too_many_arguments)]
pub fn ternary_gemm_key(
    m: usize,
    n: usize,
    k: usize,
    batch_size: usize,
    seq_len: usize,
    group_size: usize,
    dtype: DType,
    device_fingerprint: &str,
    policy_version: u32,
) -> KernelKey {
    assert!(m > 0, "ternary_gemm_key: m must be > 0");
    assert!(n > 0, "ternary_gemm_key: n must be > 0");
    assert!(k > 0, "ternary_gemm_key: k must be > 0");
    assert!(batch_size > 0, "ternary_gemm_key: batch_size must be > 0");
    assert!(seq_len > 0, "ternary_gemm_key: seq_len must be > 0");
    assert!(group_size > 0, "ternary_gemm_key: group_size must be > 0");

    KernelKey {
        operator_kind: OperatorKind::Quantized,
        attention_kind: None,
        shape_signature: ShapeSignature {
            m,
            n,
            k,
            batch: batch_size,
            seq: seq_len,
            group: group_size,
        },
        dtype,
        quantization: QuantizationPolicy::Ternary,
        state_layout_version: 1,
        device_fingerprint: device_fingerprint.to_string(),
        policy_version,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat::AttentionKind;

    const FP: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    // -- sliding_window_key ----------------------------------------------------

    #[test]
    fn sliding_window_valid_round_trips_expected_shape() {
        // q_heads=8, kv_heads=2, head_dim=64, batch_size=1, seq_len=8,
        // group_size=4, window_size=4 → pinned shape (64, 4, 2, 1, 8, 4).
        let key = sliding_window_key(8, 2, 64, 1, 8, 4, 4, DType::Bf16, FP, 1);
        assert_eq!(key.shape_signature.m, 64);
        assert_eq!(key.shape_signature.n, 4);
        assert_eq!(key.shape_signature.k, 2);
        assert_eq!(key.shape_signature.batch, 1);
        assert_eq!(key.shape_signature.seq, 8);
        assert_eq!(key.shape_signature.group, 4);
    }

    #[test]
    fn sliding_window_window_size_above_seq_len_is_clamped() {
        let key = sliding_window_key(8, 2, 64, 1, 8, 4, 999, DType::Bf16, FP, 1);
        assert_eq!(
            key.shape_signature.group, 8,
            "window_size > seq_len must clamp to seq_len (8)"
        );
    }

    #[test]
    fn sliding_window_window_size_zero_is_clamped_to_one() {
        let key = sliding_window_key(8, 2, 64, 1, 8, 4, 0, DType::Bf16, FP, 1);
        assert_eq!(
            key.shape_signature.group, 1,
            "window_size == 0 must clamp to 1 so scalar fallbacks still match"
        );
    }

    #[test]
    fn sliding_window_forwards_dtype_fingerprint_and_policy_version() {
        let key = sliding_window_key(8, 2, 64, 1, 8, 4, 4, DType::Fp16, "custom-fp", 7);
        assert_eq!(key.dtype, DType::Fp16);
        assert_eq!(key.device_fingerprint, "custom-fp");
        assert_eq!(key.policy_version, 7);
    }

    #[test]
    fn sliding_window_operator_kind_attention_gqa_discriminant() {
        let key = sliding_window_key(8, 2, 64, 1, 8, 4, 4, DType::Bf16, FP, 1);
        assert_eq!(key.operator_kind, OperatorKind::Attention);
        assert_eq!(key.attention_kind, Some(AttentionKind::Gqa));
        assert_eq!(key.quantization, QuantizationPolicy::None);
        assert_eq!(key.state_layout_version, 1);
    }

    #[test]
    #[should_panic(expected = "q_heads must be > 0")]
    fn sliding_window_panics_on_zero_q_heads() {
        let _ = sliding_window_key(0, 2, 64, 1, 8, 4, 4, DType::Bf16, FP, 1);
    }

    // -- deltanet_batched_key --------------------------------------------------

    #[test]
    fn deltanet_batched_valid_round_trips_expected_shape() {
        // (B=2, H=2, C=4, D=8) → pinned shape (8, 8, 8, 2, 4, 2).
        let key = deltanet_batched_key(2, 2, 4, 8, DType::Bf16, FP, 1);
        assert_eq!(key.shape_signature.m, 8);
        assert_eq!(key.shape_signature.n, 8);
        assert_eq!(key.shape_signature.k, 8);
        assert_eq!(key.shape_signature.batch, 2);
        assert_eq!(key.shape_signature.seq, 4);
        assert_eq!(key.shape_signature.group, 2);
    }

    #[test]
    fn deltanet_batched_forwards_dtype_fingerprint_and_policy_version() {
        let key = deltanet_batched_key(2, 2, 4, 8, DType::Fp32, "fp-x", 9);
        assert_eq!(key.dtype, DType::Fp32);
        assert_eq!(key.device_fingerprint, "fp-x");
        assert_eq!(key.policy_version, 9);
    }

    #[test]
    fn deltanet_batched_operator_kind_deltanet_discriminant() {
        let key = deltanet_batched_key(2, 2, 4, 8, DType::Bf16, FP, 1);
        assert_eq!(key.operator_kind, OperatorKind::DeltaNet);
        assert_eq!(key.attention_kind, None);
        assert_eq!(key.quantization, QuantizationPolicy::None);
        assert_eq!(key.state_layout_version, 1);
    }

    #[test]
    #[should_panic(expected = "head_dim must be > 0")]
    fn deltanet_batched_panics_on_zero_head_dim() {
        let _ = deltanet_batched_key(2, 2, 4, 0, DType::Bf16, FP, 1);
    }

    #[test]
    #[should_panic(expected = "batch_size must be > 0")]
    fn deltanet_batched_panics_on_zero_batch_size() {
        let _ = deltanet_batched_key(0, 2, 4, 8, DType::Bf16, FP, 1);
    }

    // -- deltanet_key (single chunk) -------------------------------------------

    #[test]
    fn deltanet_valid_round_trips_expected_shape() {
        // head_dim=8, chunk_size=16 → pinned shape (8, 8, 8, 1, 16, 1).
        let key = deltanet_key(8, 16, DType::Bf16, FP, 1);
        assert_eq!(key.shape_signature.m, 8);
        assert_eq!(key.shape_signature.n, 8);
        assert_eq!(key.shape_signature.k, 8);
        assert_eq!(key.shape_signature.batch, 1);
        assert_eq!(key.shape_signature.seq, 16);
        assert_eq!(key.shape_signature.group, 1);
    }

    #[test]
    fn deltanet_forwards_dtype_fingerprint_and_policy_version() {
        let key = deltanet_key(8, 16, DType::Bf16, "fp-y", 3);
        assert_eq!(key.dtype, DType::Bf16);
        assert_eq!(key.device_fingerprint, "fp-y");
        assert_eq!(key.policy_version, 3);
    }

    #[test]
    fn deltanet_operator_kind_deltanet_discriminant() {
        let key = deltanet_key(8, 16, DType::Bf16, FP, 1);
        assert_eq!(key.operator_kind, OperatorKind::DeltaNet);
        assert_eq!(key.attention_kind, None);
        assert_eq!(key.quantization, QuantizationPolicy::None);
        assert_eq!(key.state_layout_version, 1);
    }

    #[test]
    #[should_panic(expected = "chunk_size must be > 0")]
    fn deltanet_panics_on_zero_chunk_size() {
        let _ = deltanet_key(8, 0, DType::Bf16, FP, 1);
    }

    #[test]
    fn retnet_key_uses_recurrent_shape_contract() {
        let key = retnet_key(2, 4, 16, 8, DType::Bf16, FP, 3);
        assert_eq!(key.operator_kind, OperatorKind::Recurrent);
        assert_eq!(
            key.shape_signature,
            ShapeSignature {
                m: 16,
                n: 16,
                k: 16,
                batch: 2,
                seq: 8,
                group: 4
            }
        );
        assert_eq!(key.policy_version, 3);
    }

    #[test]
    fn mamba_scan_key_uses_scan_shape_contract() {
        let key = mamba_scan_key(1, 257, 32, DType::Fp32, FP, 4);
        assert_eq!(key.operator_kind, OperatorKind::Scan);
        assert_eq!(
            key.shape_signature,
            ShapeSignature {
                m: 257,
                n: 257,
                k: 257,
                batch: 1,
                seq: 32,
                group: 1
            }
        );
    }
}
