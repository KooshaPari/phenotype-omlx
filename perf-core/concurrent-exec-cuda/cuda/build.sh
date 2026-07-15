#!/usr/bin/env bash
# =============================================================================
#  Build script for the phenotype-omlx CUDA shared library.
#
#  Produces: libphenotype_omlx_cuda.so (Linux) / phenotype_omlx_cuda.dll (Win)
#  Consumed by: perf-core/concurrent-exec-cuda via libloading::Library::new
#
#  Requirements:
#    - CUDA Toolkit (nvcc) on PATH. Tested with CUDA 12.x.
#    - cmake >= 3.18
#    - A build host with an NVIDIA driver/GPU available.
#
#  On Apple Silicon (M1/M2/M3) or any host without an NVIDIA driver this
#  script is NOT expected to succeed — concurrent-exec-cuda is gated to
#  Linux/Windows + NVIDIA only.
# =============================================================================
set -euo pipefail

# Resolve script directory regardless of where it's invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "[phenotype_omlx_cuda] CUDA toolkit:"
nvcc --version || { echo "nvcc not found on PATH — install CUDA Toolkit first"; exit 1; }

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-75;80;86;89;90}"

cmake --build "${BUILD_DIR}" --parallel "$(nproc 2>/dev/null || echo 4)"

# Stash the artifact next to the build script so the Rust example can
# dlopen("./libphenotype_omlx_cuda.so") without further path config.
cp -v "${BUILD_DIR}/libphenotype_omlx_cuda.so" "${SCRIPT_DIR}/" 2>/dev/null || true

echo "[phenotype_omlx_cuda] OK — ${BUILD_DIR}/libphenotype_omlx_cuda.so"