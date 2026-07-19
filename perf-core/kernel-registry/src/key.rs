//! [`KernelKey`] — the cache and selector identity.
//!
//! A `KernelKey` is constructed for every operator invocation. It captures
//! everything that determines kernel eligibility and tunability:
//!
//! - the operator kind and (where relevant) attention variant;
//! - the operand shape bundle ([`ShapeSignature`]);
//! - the data type and quantization policy;
//! - the state-layout version (changes invalidate prior evidence);
//! - the device fingerprint (a SHA-256 of the host+device tuple);
//! - the selection-policy version (changes invalidate prior evidence).
//!
//! Two `KernelKey` values are equal iff every field is equal. Equality and
//! [`KernelKey::fast_hash`] both treat the `policy_version` as
//! distinguishing, because a policy bump is itself evidence that prior
//! decisions may not be valid under the new rules.

use serde::{Deserialize, Serialize};

use crate::compat::{AttentionKind, DType, OperatorKind, QuantizationPolicy};

/// Operand shape bundle carried by [`KernelKey::shape_signature`].
///
/// `m`, `n`, `k` map to the canonical matmul operands. `batch`, `seq`,
/// `group` cover transformer / MoE axes. `group` may be zero for operators
/// that do not partition along that axis; non-grouped candidates must then
/// be gated through the shape filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShapeSignature {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: usize,
    pub seq: usize,
    pub group: usize,
}

impl ShapeSignature {
    /// `true` when `self >= other` component-wise. Used to check that
    /// `other` is at least as large as the candidate's `min_shape`.
    pub fn ge(&self, other: &ShapeSignature) -> bool {
        self.m >= other.m
            && self.n >= other.n
            && self.k >= other.k
            && self.batch >= other.batch
            && self.seq >= other.seq
            && self.group >= other.group
    }

    /// `true` when `self` lies inside `[min, max]` component-wise. The
    /// caller passes `self = key.shape_signature`, `min = candidate.min_shape`,
    /// `max = candidate.max_shape`.
    pub fn within(&self, min: &ShapeSignature, max: &ShapeSignature) -> bool {
        self.ge(min) && max.ge(self)
    }

    /// `true` when every axis is zero — the smallest legal degenerate shape.
    /// Selectors that probe an operator with `Zero` must fall back to a
    /// reference kernel because no tuned record will exist.
    pub fn is_zero(&self) -> bool {
        self.m == 0 && self.n == 0 && self.k == 0
            && self.batch == 0 && self.seq == 0 && self.group == 0
    }
}

/// Marker for "no specific attention variant". Distinct from
/// `AttentionKind::Standard` so the key remains hashable.
pub const ATTENTION_NONE: Option<AttentionKind> = None;

/// Cache / selector identity. Hashable, serializable, and stable across
/// processes (see [`KernelKey::fast_hash`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KernelKey {
    pub operator_kind: OperatorKind,
    pub attention_kind: Option<AttentionKind>,
    pub shape_signature: ShapeSignature,
    pub dtype: DType,
    pub quantization: QuantizationPolicy,
    pub state_layout_version: u32,
    pub device_fingerprint: String,
    pub policy_version: u32,
}

impl KernelKey {
    /// Deterministic 64-bit hash of this key. Stable across builds and
    /// platforms. Uses FNV-1a over a fixed serialization of every field so
    /// accidental field reordering is loud at test time.
    pub fn fast_hash(&self) -> u64 {
        let mut buf: Vec<u8> = Vec::with_capacity(128);
        // operator_kind discriminant
        buf.push(discriminant_of_op(self.operator_kind));
        // attention_kind: 0xff = none, 0x00..0x05 = variants
        match self.attention_kind {
            None => buf.push(0xff),
            Some(AttentionKind::Gqa) => buf.push(0x01),
            Some(AttentionKind::Mla) => buf.push(0x02),
            Some(AttentionKind::Cca) => buf.push(0x03),
            Some(AttentionKind::Tree) => buf.push(0x04),
            Some(AttentionKind::Paged) => buf.push(0x05),
            Some(AttentionKind::Standard) => buf.push(0x06),
        }
        // shape axes
        for axis in [
            self.shape_signature.m,
            self.shape_signature.n,
            self.shape_signature.k,
            self.shape_signature.batch,
            self.shape_signature.seq,
            self.shape_signature.group,
        ] {
            buf.extend_from_slice(&axis.to_le_bytes());
        }
        buf.push(discriminant_of_dtype(self.dtype));
        buf.push(discriminant_of_quant(self.quantization));
        buf.extend_from_slice(&self.state_layout_version.to_le_bytes());
        buf.extend_from_slice(self.device_fingerprint.as_bytes());
        buf.extend_from_slice(&self.policy_version.to_le_bytes());
        fast_hash_bytes(&buf)
    }
}

fn discriminant_of_op(op: OperatorKind) -> u8 {
    match op {
        OperatorKind::DenseMatmul => 0x01,
        OperatorKind::GroupedMatmul => 0x02,
        OperatorKind::Attention => 0x03,
        OperatorKind::Gqa => 0x04,
        OperatorKind::Mla => 0x05,
        OperatorKind::Cca => 0x06,
        OperatorKind::TreeAttention => 0x07,
        OperatorKind::PagedAttention => 0x08,
        OperatorKind::Moe => 0x09,
        OperatorKind::MoeSharedExpert => 0x0a,
        OperatorKind::DeltaNet => 0x0b,
        OperatorKind::ShortConv => 0x0c,
        OperatorKind::Scan => 0x0d,
        OperatorKind::Recurrent => 0x0e,
        OperatorKind::Diffusion => 0x0f,
        OperatorKind::DiscreteDiffusion => 0x12,
        OperatorKind::Speculative => 0x10,
        OperatorKind::Quantized => 0x11,
        OperatorKind::Unknown => 0x7f,
    }
}

fn discriminant_of_dtype(d: DType) -> u8 {
    match d {
        DType::Fp32 => 0x10,
        DType::Fp16 => 0x11,
        DType::Bf16 => 0x12,
        DType::Fp8 => 0x13,
        DType::Int8 => 0x14,
        DType::Int4 => 0x15,
        DType::Bool => 0x16,
        DType::Unknown => 0x7f,
    }
}

fn discriminant_of_quant(q: QuantizationPolicy) -> u8 {
    match q {
        QuantizationPolicy::None => 0x00,
        QuantizationPolicy::Fp8 => 0x01,
        QuantizationPolicy::Int8 => 0x02,
        QuantizationPolicy::Int4 => 0x03,
        QuantizationPolicy::Ternary => 0x04,
        QuantizationPolicy::SubByte => 0x05,
        QuantizationPolicy::Unknown => 0x7f,
    }
}

/// FNV-1a 64-bit hash over an arbitrary byte slice. Stable across builds
/// and platforms; used as the underlying primitive for both
/// [`KernelKey::fast_hash`] and `CandidateId` derivation.
pub fn fast_hash_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}