# gemv_decode.mojo — GPU-optimized GEMV (General Matrix-Vector Multiply) kernel.
#
# Used in the decode phase of quantized inference: output = W * input.
#
# Build:
#   mojo build mojo-src/gemv_decode.mojo --emit shared-lib -o libturbo_quant_mojo.dylib
#
# The actual Mojo FFI bridge requires the Mojo runtime and `mojo-ffi` crate.
# This file provides the API surface and the reference scalar implementation.

from math import sqrt


fn gemv_decode(
    weights: UnsafePointer[Float32, MutUntrackedOrigin],
    input: UnsafePointer[Float32, MutUntrackedOrigin],
    output: UnsafePointer[Float32, MutUntrackedOrigin],
    rows: Int,
    cols: Int,
):
    """GPU-optimized GEMV for decode phase.

    weights: (rows, cols) weight matrix in row-major layout
    input: (cols,) input vector
    output: (rows,) output vector

    Computes: output[r] = sum_over_c(weights[r * cols + c] * input[c])
    """
    for row in range(rows):
        var sum: Float32 = 0.0
        for col in range(cols):
            sum += weights.load(row * cols + col)[0] * input.load(col)[0]
        output.store(row, sum)


@export("tq_gemv_decode")
def tq_gemv_decode(
    weights_ptr: UnsafePointer[Float32, MutUntrackedOrigin],
    input_ptr: UnsafePointer[Float32, MutUntrackedOrigin],
    output_ptr: UnsafePointer[Float32, MutUntrackedOrigin],
    rows: Int,
    cols: Int,
) -> Bool:
    """C ABI entry point for GEMV decode kernel."""
    gemv_decode(weights_ptr, input_ptr, output_ptr, rows, cols)
    return True
