# =============================================================================
#  CUDA shared library for phenotype-omlx perf-core
# -----------------------------------------------------------------------------
#  Builds libphenotype_omlx_cuda.so (Linux) / phenotype_omlx_cuda.dll
#  (Windows) from the kernels under cuda/kernels/*.cu.
#
#  The Rust crate `concurrent-exec-cuda` dlopens this library at runtime via
#  libloading; see src/lib.rs::loader and examples/load_cuda_runtime.rs.
#
#  Requirements:
#    - CUDA Toolkit (nvcc) >= 12.0
#    - cmake >= 3.18
#    - An NVIDIA driver + GPU on the build host
# =============================================================================

cmake_minimum_required(VERSION 3.18)
project(phenotype_omlx_cuda LANGUAGES C CXX CUDA)

# Use the modern CMake FindCUDA path (introduced in CMake 3.17) so we get
# separable compilation per source file. find_package(CUDA) is the legacy
# module; we still call it for backward compatibility with older toolkits.
find_package(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(phenotype_omlx_cuda SHARED
    kernels/latentmas_fanout.cu
    kernels/ssd_verify.cu
)

target_include_directories(phenotype_omlx_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

# Shared-library + PIC flags. nvcc forwards --compiler-options to the host
# compiler, which is what makes -fPIC land on Linux; MSVC ignores it.
target_compile_options(phenotype_omlx_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--shared>
    $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options,-fPIC>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# Linux: libphenotype_omlx_cuda.so  Windows: phenotype_omlx_cuda.dll
set_target_properties(phenotype_omlx_cuda PROPERTIES
    OUTPUT_NAME "phenotype_omlx_cuda"
    PREFIX "lib"
)

# Export every kernel as `extern "C"` so the Rust side can dlsym by name.
# nvcc keeps the unmangled name when the function is declared `extern "C"`
# in the .cu file (see kernels/*.cu).