//! Candidate metadata: identity, backend, capability requirements, and shape
//! bounds.
//!
//! A [`Candidate`] is the immutable description of a kernel that *might* be
//! usable for a given [`crate::KernelKey`]. The selector consults `requires`
//! and `min_shape` / `max_shape` to filter eligibility and falls back to
//! tunability to decide whether evidence can be promoted.

use serde::{Deserialize, Serialize};

use crate::compat::DType;
use crate::key::{fast_hash_bytes, ShapeSignature};

/// Newtype wrapping a stable candidate identifier. Derived from
/// `Candidate::name` (and the backend discriminant where collision is
/// possible) so two candidates with identical content are detected as the
/// same id and can be deduplicated at registration time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CandidateId(pub u64);

impl CandidateId {
    /// Derive an id from a candidate's `(name, backend)` pair. Stable
    /// across builds and platforms because it relies on FNV-1a over the
    /// backend discriminant followed by the name bytes.
    pub fn derive(name: &str, backend: BackendKind) -> Self {
        let mut buf = Vec::with_capacity(name.len() + 1);
        buf.push(backend_discriminant(backend));
        buf.extend_from_slice(name.as_bytes());
        CandidateId(fast_hash_bytes(&buf))
    }
}

impl core::fmt::Display for CandidateId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Candidate({:016x})", self.0)
    }
}

/// Backend in which a candidate kernel is implemented. Mirrors the
/// multi-engine reality of the runtime (MLX/Metal + polyglot kernels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    Cpu,
    Metal,
    Cuda,
    Vulkan,
    Zig,
    Mojo,
    Nim,
    Go,
    /// Reference / scalar oracle — always considered last.
    Reference,
}

fn backend_discriminant(b: BackendKind) -> u8 {
    match b {
        BackendKind::Cpu => 0x01,
        BackendKind::Metal => 0x02,
        BackendKind::Cuda => 0x03,
        BackendKind::Vulkan => 0x04,
        BackendKind::Zig => 0x05,
        BackendKind::Mojo => 0x06,
        BackendKind::Nim => 0x07,
        BackendKind::Go => 0x08,
        BackendKind::Reference => 0x7f,
    }
}

impl BackendKind {
    /// `true` for the always-eligible scalar reference backend.
    pub fn is_reference(&self) -> bool {
        matches!(self, BackendKind::Reference)
    }

    /// Stable short tag, used in traces and reports.
    pub fn as_tag(&self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Metal => "metal",
            BackendKind::Cuda => "cuda",
            BackendKind::Vulkan => "vulkan",
            BackendKind::Zig => "zig",
            BackendKind::Mojo => "mojo",
            BackendKind::Nim => "nim",
            BackendKind::Go => "go",
            BackendKind::Reference => "ref",
        }
    }
}

/// Hardware or numerical capability a kernel requires. The selector
/// requires `candidate.requires ⊆ device_capabilities` to deem a candidate
/// eligible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    Avx2,
    Avx512,
    Amx,
    Neon,
    MetalGpu,
    /// Metal MSL 3.0+ feature set required by some quantized kernels.
    MetalMs3,
    Cuda,
    Vulkan,
    Bf16,
    Fp16,
}

impl Capability {
    /// Canonical string for human-readable rejection explanations.
    pub fn as_str(&self) -> &'static str {
        match self {
            Capability::Avx2 => "avx2",
            Capability::Avx512 => "avx512",
            Capability::Amx => "amx",
            Capability::Neon => "neon",
            Capability::MetalGpu => "metal-gpu",
            Capability::MetalMs3 => "metal-ms3",
            Capability::Cuda => "cuda",
            Capability::Vulkan => "vulkan",
            Capability::Bf16 => "bf16",
            Capability::Fp16 => "fp16",
        }
    }
}

/// Immutable description of a kernel that *might* be usable for a given
/// [`crate::KernelKey`].
///
/// `id` should be derived via [`CandidateId::derive`] so identical
/// `(name, backend)` pairs dedupe at registration. `source_hash` is the
/// SHA-256 of the kernel artifact or `"ref"` for reference kernels.
/// `requires` lists every capability the device must advertise for the
/// candidate to be eligible. `min_shape` and `max_shape` bound the
/// shape envelope. `supports_dtypes` is the dtype whitelist. `tunable`
/// indicates whether the candidate may acquire tuning evidence at runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: CandidateId,
    pub name: String,
    pub backend: BackendKind,
    pub source_hash: String,
    pub requires: Vec<Capability>,
    pub min_shape: ShapeSignature,
    pub max_shape: ShapeSignature,
    pub supports_dtypes: Vec<DType>,
    pub tunable: bool,
}

impl Candidate {
    /// Convenience constructor that derives `id` from `name` + `backend`.
    pub fn new(
        name: impl Into<String>,
        backend: BackendKind,
        source_hash: impl Into<String>,
        requires: Vec<Capability>,
        min_shape: ShapeSignature,
        max_shape: ShapeSignature,
        supports_dtypes: Vec<DType>,
        tunable: bool,
    ) -> Self {
        let name = name.into();
        let id = CandidateId::derive(&name, backend);
        Self {
            id,
            name,
            backend,
            source_hash: source_hash.into(),
            requires,
            min_shape,
            max_shape,
            supports_dtypes,
            tunable,
        }
    }

    /// `true` when every capability in `requires` is present in `caps`.
    pub fn capabilities_satisfied(&self, caps: &super::registry::DeviceCaps) -> bool {
        self.requires
            .iter()
            .all(|req| caps.capabilities.contains(req))
    }

    /// `true` when `shape` lies inside `[min_shape, max_shape]`.
    pub fn supports_shape(&self, shape: &ShapeSignature) -> bool {
        shape.within(&self.min_shape, &self.max_shape)
    }

    /// `true` when `dtype` is in `supports_dtypes`.
    pub fn supports_dtype(&self, dtype: DType) -> bool {
        self.supports_dtypes.contains(&dtype)
    }
}