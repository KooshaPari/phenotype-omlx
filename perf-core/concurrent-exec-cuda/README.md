# concurrent-exec-cuda

CUDA backend for `concurrent-exec` — Linux/Windows + NVIDIA only.

This module is the CUDA-native sibling of `perf-core/concurrent-exec/`. The
Rust crate handles orchestration (DAG scheduling, agent registration,
fan-out / first-success / chain strategies); CUDA handles the per-agent
math (top-k fan-out, draft verification, tree-attention pre-pass).

## Layout

```
concurrent-exec-cuda/
├── Cargo.toml                  # path-deps ../concurrent-exec; optional libloading
├── src/lib.rs                  # CudaLatentMasBackend / CudaSsdBackend / CudaJetSpecBackend
├── examples/load_cuda_runtime.rs   # dlopen libcudart + libphenotype_omlx_cuda
└── cuda/
    ├── CMakeLists.txt          # find_package(CUDA) + add_library(...SHARED)
    ├── build.sh                # one-shot cmake + make
    └── kernels/
        ├── latentmas_fanout.cu # top-k per agent (one CUDA block per agent)
        └── ssd_verify.cu       # draft-token verification pass
```

## Build

### 1. CUDA shared library

```bash
cd cuda
./build.sh          # produces libphenotype_omlx_cuda.so
```

Requires the CUDA Toolkit (`nvcc` ≥ 12.0) and an NVIDIA driver on the build
host. The artifact is copied next to `build.sh` so the Rust example can
`dlopen("./libphenotype_omlx_cuda.so")` without extra path config.

### Mixed GTX 1080 Ti + RTX 3090 Ti desktop

The Pascal GTX 1080 Ti requires CUDA 12.x and `sm_61`; the Ampere RTX 3090 Ti
uses `sm_86`. Build one portable artifact containing both profiles explicitly:

```bash
CUDA_PROFILE=portable CUDA_ARCH='61;86' ./build.sh
```

Do not use a CUDA 13.x toolkit for this mixed build: CUDA 13 no longer emits
Pascal `sm_61` code. The `pascal` and `ampere` profiles remain available for
single-GPU builds, while `CUDA_ARCH` is the explicit escape hatch for a mixed
or otherwise customized architecture set.

### 2. Rust crate

```bash
cargo build --features cuda --release
cargo run  --example load_cuda_runtime --features cuda
```

## Platform behavior

| Platform                       | `cargo build` (default) | `cargo build --features cuda` |
| ------------------------------ | ----------------------- | ----------------------------- |
| macOS (any arch)               | OK — empty shim         | OK at compile, errors at dlopen (`CUDA backend requires NVIDIA GPU`) |
| Linux + no NVIDIA driver       | OK — empty shim         | OK at compile, errors at `cudaMalloc` time |
| Linux/Windows + NVIDIA + CUDA  | OK — empty shim         | Loads `libphenotype_omlx_cuda.so` and dispatches kernels |

Default `cargo build` on macOS must succeed and does — every CUDA-specific
symbol is gated behind `#[cfg(feature = "cuda")]`.

## Loader pattern

The intended runtime sequence (see `src/lib.rs::loader`) is:

```rust,ignore
let rt  = loader::open_runtime()?;   // libcudart.so   (Linux) / cudart64_*.dll (Win)
let kup = loader::open_kernels()?;   // libphenotype_omlx_cuda.so  / .dll
let h   = loader::resolve(&kup, "latentmas_fanout_kernel")?;
// cuLaunchKernel(h, packed_args) → device
```

All kernels in `cuda/kernels/*.cu` are declared `extern "C"` so `dlsym`
lookups match by raw symbol name.
