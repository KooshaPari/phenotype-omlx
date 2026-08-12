#!/usr/bin/env bash
# =============================================================================
#  Build script for the phenotype-omlx CUDA shared library.
#
#  Produces: libphenotype_omlx_cuda.so (Linux) / phenotype_omlx_cuda.dll (Win)
#  Consumed by: perf-core/concurrent-exec-cuda via libloading::Library::new
#
#  Requirements:
#    - CUDA Toolkit (nvcc) on PATH. CUDA 12.x is required for Pascal/sm_61;
#      CUDA 12.x or 13.x may be used for Turing-and-newer profiles.
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
NVCC_VERSION="$(nvcc --version 2>&1)" || {
    echo "nvcc not found on PATH — install CUDA Toolkit first" >&2
    exit 1
}
printf '%s\n' "${NVCC_VERSION}"

# Keep architecture selection explicit for the supported hardware lanes. A
# Pascal 1080 Ti requires a CUDA 12.x compiler and sm_61; CUDA 13 removed
# offline compilation for Pascal. An Ampere 3090 Ti uses sm_86 and works with
# either CUDA 12.x or 13.x. CUDA_ARCH remains an escape hatch for additional
# architectures, while CUDA_PROFILE is the reproducible operator-facing path.
CUDA_PROFILE="${CUDA_PROFILE:-portable}"
case "${CUDA_PROFILE}" in
    pascal|sm61|1080ti)
        [[ -z "${CUDA_ARCH:-}" || "${CUDA_ARCH}" == "61" ]] || {
            echo "CUDA_PROFILE=${CUDA_PROFILE} requires CUDA_ARCH=61 (got ${CUDA_ARCH})" >&2
            exit 2
        }
        CUDA_ARCH="61"
        ;;
    ampere|sm86|3090ti)
        [[ -z "${CUDA_ARCH:-}" || "${CUDA_ARCH}" == "86" ]] || {
            echo "CUDA_PROFILE=${CUDA_PROFILE} requires CUDA_ARCH=86 (got ${CUDA_ARCH})" >&2
            exit 2
        }
        CUDA_ARCH="86"
        ;;
    portable|default) CUDA_ARCH="${CUDA_ARCH:-75;80;86;89;90}" ;;
    *)
        echo "unknown CUDA_PROFILE=${CUDA_PROFILE}; use pascal, ampere, or portable" >&2
        exit 2
        ;;
esac

CUDA_MAJOR="$(printf '%s\n' "${NVCC_VERSION}" | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n 1)"
if [[ "${CUDA_MAJOR}" =~ ^[0-9]+$ ]] && (( CUDA_MAJOR >= 13 )) && [[ ";${CUDA_ARCH};" == *";61;"* ]]; then
    echo "CUDA_ARCH=${CUDA_ARCH} includes sm_61, but CUDA ${CUDA_MAJOR} no longer compiles Pascal; use CUDA 12.x." >&2
    exit 2
fi

echo "[phenotype_omlx_cuda] profile=${CUDA_PROFILE} arch=${CUDA_ARCH} toolkit_major=${CUDA_MAJOR:-unknown}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"

cmake --build "${BUILD_DIR}" --parallel "$(nproc 2>/dev/null || echo 4)"

# Stash the artifact next to the build script so the Rust example can
# dlopen("./libphenotype_omlx_cuda.so") without further path config.
cp -v "${BUILD_DIR}/libphenotype_omlx_cuda.so" "${SCRIPT_DIR}/" 2>/dev/null || true

echo "[phenotype_omlx_cuda] OK — ${BUILD_DIR}/libphenotype_omlx_cuda.so"
