//! CUDA-native sibling of `concurrent-exec`.
//!
//! The Rust `concurrent-exec` crate handles *orchestration* — DAG scheduling,
//! agent registration, fan-out / first-success / chain strategies. This crate
//! handles the *per-agent math* on NVIDIA GPUs:
//!
//! - [`CudaLatentMasBackend`] — parallel top-k fan-out across N latent agents.
//! - [`CudaSsdBackend`]      — draft verification for self-speculative decoding.
//! - [`CudaJetSpecBackend`]  — tree-attention speculative decoding pre-pass.
//!
//! All CUDA-specific symbols are gated behind `#[cfg(feature = "cuda")]`. On
//! macOS or any non-NV platform, the crate compiles to an empty shim — the
//! orchestration APIs from `concurrent-exec` are still usable via the path
//! dependency declared in `Cargo.toml`, but every `run_*` here returns an
//! `Err("CUDA backend requires NVIDIA GPU")`.

use async_trait::async_trait;
use concurrent_exec::{AgentId, ExecBackend, ExecRequest, ExecResult, JobError};
use thiserror::Error;

/// Compile-time marker used by the loader. We refuse to dlopen on macOS even
/// when the `cuda` feature is enabled, because Apple's `libcudart.dylib`
/// does not ship and there is no CUDA toolkit for M-series silicon.
#[allow(dead_code)]
const REQUIRES_NVIDIA_GPU: &str = "CUDA backend requires NVIDIA GPU";

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("libloading: {0}")]
    LibLoading(#[from] concurrent_exec::JobError),
    #[error("platform check: {0}")]
    Platform(&'static str),
}

// ---------------------------------------------------------------------------
// Placeholder types — always available so the crate type-checks on macOS.
// ---------------------------------------------------------------------------

/// Stub. On macOS / non-NV, construction is allowed but `run` errors out.
pub struct CudaLatentMasBackend {
    pub n_agents: usize,
}

impl CudaLatentMasBackend {
    pub fn new(n_agents: usize) -> Self {
        Self { n_agents }
    }
}

/// Stub. On macOS / non-NV, construction is allowed but `run` errors out.
pub struct CudaSsdBackend {
    pub gamma: usize,
}

impl CudaSsdBackend {
    pub fn new(gamma: usize) -> Self {
        Self { gamma }
    }
}

/// Stub. On macOS / non-NV, construction is allowed but `run` errors out.
pub struct CudaJetSpecBackend {
    pub tree_width: usize,
    pub tree_depth: usize,
}

impl CudaJetSpecBackend {
    pub fn new(tree_width: usize, tree_depth: usize) -> Self {
        Self {
            tree_width,
            tree_depth,
        }
    }
}

// ---------------------------------------------------------------------------
// Trait impls — always compiled. Real CUDA dispatch lives behind `cfg`.
// ---------------------------------------------------------------------------

#[async_trait]
impl ExecBackend for CudaLatentMasBackend {
    async fn run(&self, _id: AgentId, _req: ExecRequest) -> Result<ExecResult, JobError> {
        Err(JobError::Backend(REQUIRES_NVIDIA_GPU.into()))
    }
}

#[async_trait]
impl ExecBackend for CudaSsdBackend {
    async fn run(&self, _id: AgentId, _req: ExecRequest) -> Result<ExecResult, JobError> {
        Err(JobError::Backend(REQUIRES_NVIDIA_GPU.into()))
    }
}

#[async_trait]
impl ExecBackend for CudaJetSpecBackend {
    async fn run(&self, _id: AgentId, _req: ExecRequest) -> Result<ExecResult, JobError> {
        Err(JobError::Backend(REQUIRES_NVIDIA_GPU.into()))
    }
}

// ---------------------------------------------------------------------------
// CUDA-only loader — the real libloading / cuLaunchKernel wiring.
// Gated so macOS builds never touch `libloading`.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub mod loader {
    //! Dynamic loader for `libphenotype_omlx_cuda.so` and `libcudart.so`.
    //!
    //! On Linux + NVIDIA, the expected dlopen sequence is:
    //!
    //! ```ignore
    //! use libloading::Library;
    //! use concurrent_exec_cuda::loader::{open_runtime, open_kernels};
    //!
    //! let rt  = open_runtime()?;     // libcudart.so / cudart64_*.dll
    //! let kup = open_kernels()?;     // libphenotype_omlx_cuda.so / .dll
    //! // resolve cuLaunchKernel, dispatch to a kernel by name, …
    //! ```
    //!
    //! The FFI surface intentionally mirrors the C ABI exported by
    //! `cuda/kernels/*.cu` so that `dlsym` lookups are stable across
    //! nvcc versions.

    use std::ffi::CString;
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum LoaderError {
        #[error("dlopen failed: {0}")]
        DlOpen(String),
        #[error("dlsym failed for `{symbol}`: {source}")]
        DlSym {
            symbol: String,
            #[source]
            source: libloading::Error,
        },
        #[error("platform check: {0}")]
        Platform(&'static str),
    }

    pub type KernelHandle = *mut std::ffi::c_void;

    /// `libcudart.so` on Linux, `cudart64_12.dll` on Windows.
    pub fn open_runtime() -> Result<libloading::Library, LoaderError> {
        #[cfg(target_os = "macos")]
        {
            return Err(LoaderError::Platform(
                "libcudart is not available on macOS — CUDA requires Linux/Windows + NVIDIA",
            ));
        }
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        {
            // SAFETY: we never call this on macOS (early return above).
            unsafe {
                libloading::Library::new("libcudart.so")
                    .map_err(|e| LoaderError::DlOpen(e.to_string()))
            }
        }
        #[cfg(all(target_os = "windows", feature = "cuda"))]
        {
            unsafe {
                libloading::Library::new("cudart64_12.dll")
                    .map_err(|e| LoaderError::DlOpen(e.to_string()))
            }
        }
    }

    /// `./libphenotype_omlx_cuda.so` on Linux, `.dll` on Windows.
    /// Built by `cuda/build.sh` / `cuda/CMakeLists.txt`.
    pub fn open_kernels() -> Result<libloading::Library, LoaderError> {
        #[cfg(target_os = "macos")]
        {
            return Err(LoaderError::Platform(
                "libphenotype_omlx_cuda is not built on macOS — CUDA requires Linux/Windows + NVIDIA",
            ));
        }
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        {
            unsafe {
                libloading::Library::new("./libphenotype_omlx_cuda.so")
                    .map_err(|e| LoaderError::DlOpen(e.to_string()))
            }
        }
        #[cfg(all(target_os = "windows", feature = "cuda"))]
        {
            unsafe {
                libloading::Library::new("./phenotype_omlx_cuda.dll")
                    .map_err(|e| LoaderError::DlOpen(e.to_string()))
            }
        }
    }

    /// Resolve a kernel symbol by name. Caller is responsible for casting to
    /// the correct `extern "C" fn(...)` signature declared in
    /// `cuda/kernels/*.cu`.
    pub fn resolve(lib: &libloading::Library, name: &str) -> Result<KernelHandle, LoaderError> {
        let cstr = CString::new(name).map_err(|_| LoaderError::DlSym {
            symbol: name.into(),
            source: libloading::Error::DlSymUnknown,
        })?;
        // SAFETY: caller ensures `name` matches an `extern "C"` export.
        unsafe {
            lib.get::<KernelHandle>(cstr.as_bytes_with_nul())
                .map(|sym| *sym)
                .map_err(|e| LoaderError::DlSym {
                    symbol: name.into(),
                    source: e,
                })
        }
    }

    /// cuLaunchKernel signature mirror — exact arg layout matches the CUDA
    /// Runtime API so we can wrap it from Rust via libloading.
    #[allow(non_snake_case, dead_code)]
    pub type CuLaunchKernelFn = unsafe extern "C" fn(
        KernelHandle,            // function handle
        *const std::ffi::c_void, // kernel args (packed by driver)
    ) -> i32;
}
