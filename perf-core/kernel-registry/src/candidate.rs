//! Candidate metadata: identity, backend, capability requirements, and shape
//! bounds.
//!
//! A [`Candidate`] is the immutable description of a kernel that *might* be
//! usable for a given [`crate::KernelKey`]. The selector consults `requires`
//! and `min_shape` / `max_shape` to filter eligibility and falls back to
//! tunability to decide whether evidence can be promoted.

use std::collections::HashMap;

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
/// `engine_name` is an optional metadata tag identifying an *external*
/// inference engine (e.g. `SGLang`, `vLLM`, `TRT-LLM`, `llama.cpp`) that
/// the candidate represents for audit/observability. This is **not** a
/// [`BackendKind`] variant — `BackendKind` describes the *kernel
/// substrate* (Metal, Cuda, Zig, ...) while `engine_name` describes the
/// *external serving engine* the kernel is associated with. When `Some`,
/// the engine name is folded into `source_hash` deterministically and
/// surfaced in [`crate::ExecutionTrace::human_explanation`] so the audit
/// trail records which engine was selected.
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
    /// Optional external-engine tag (e.g. `SGLang`, `vLLM`, `TRT-LLM`,
    /// `llama.cpp`). `None` for in-tree MLX/Metal/CPU/etc. candidates.
    /// See the module-level docs for the BackendKind vs engine_name
    /// distinction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_name: Option<String>,
    /// Extensible key-value metadata attached to a candidate. Selectors
    /// and policy logic can consult arbitrary properties without changing
    /// the core struct. For example, `"dynamic_eviction" => "true"` marks
    /// candidates that integrate with EchoKV cache management.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub properties: HashMap<String, String>,
}

impl Candidate {
    /// Convenience constructor that derives `id` from `name` + `backend`
    /// and leaves `engine_name` as `None`. Use [`Candidate::with_engine`]
    /// when registering a candidate that represents an external inference
    /// engine.
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
            engine_name: None,
            properties: HashMap::new(),
        }
    }

    /// Build a candidate with an external-engine tag. When `engine_name` is
    /// `Some`, the value is folded into `source_hash` deterministically
    /// (suffix `[engine:<name>]`) so two candidates with identical source
    /// bytes but different engines are distinguishable on disk and in
    /// audit logs. When `engine_name` is `None`, the source hash is used
    /// unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn with_engine(
        name: impl Into<String>,
        backend: BackendKind,
        source_hash: impl Into<String>,
        engine_name: Option<impl Into<String>>,
        requires: Vec<Capability>,
        min_shape: ShapeSignature,
        max_shape: ShapeSignature,
        supports_dtypes: Vec<DType>,
        tunable: bool,
    ) -> Self {
        let name = name.into();
        let id = CandidateId::derive(&name, backend);
        let base_hash: String = source_hash.into();
        let engine_name: Option<String> = engine_name.map(Into::into);
        let source_hash = Self::fold_engine_into_source_hash(&base_hash, engine_name.as_deref());
        Self {
            id,
            name,
            backend,
            source_hash,
            requires,
            min_shape,
            max_shape,
            supports_dtypes,
            tunable,
            engine_name,
            properties: HashMap::new(),
        }
    }

    /// Deterministic `[engine:<name>]` suffix applied to `source_hash`
    /// when an `engine_name` is present. Centralized so tests can pin
    /// the exact form.
    fn fold_engine_into_source_hash(base: &str, engine: Option<&str>) -> String {
        match engine {
            Some(name) => format!("{base}[engine:{name}]"),
            None => base.to_string(),
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

    /// Attach a key-value property to this candidate (builder pattern).
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// `true` when property `key` exists and equals `"true"`.
    pub fn has_property(&self, key: &str) -> bool {
        self.properties
            .get(key)
            .map(|v| v == "true")
            .unwrap_or(false)
    }
}
