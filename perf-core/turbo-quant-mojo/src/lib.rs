// turbo-quant-mojo — Rust wrapper for the Mojo implementation.
//
// Requires the Mojo SDK in PATH (`modular install mojo`). build.rs compiles
// mojo-src/turbo_quant.mojo to a shared library and links it unconditionally.
//
// ABI safety: caller-controlled dimensions crossing the Mojo boundary are
// validated before entering unsafe Mojo. See [`crate::validation`].

#[derive(Debug, Clone)]
pub struct MojoQuantizedTensor {
    pub shape: Vec<usize>,
    pub packed: Vec<u8>,
    pub scales: Vec<f32>,
    pub zeros: Vec<f32>,
}

impl MojoQuantizedTensor {
    pub fn encode(data: &[f32], bits: u8, group_size: usize) -> Result<Self, String> {
        #[cfg(feature = "mojo")]
        {
            native::mojo_encode(data, bits, group_size)
        }
        #[cfg(not(feature = "mojo"))]
        {
            validation::validate_encode_inputs(data.len(), bits, group_size)?;
            Err("Mojo feature not enabled".to_string())
        }
    }

    pub fn try_decode(&self, n: usize, group_size: usize, bits: u8) -> Result<Vec<f32>, String> {
        #[cfg(feature = "mojo")]
        {
            native::mojo_try_decode(
                &self.shape,
                &self.packed,
                &self.scales,
                &self.zeros,
                n,
                group_size,
                bits,
            )
        }
        #[cfg(not(feature = "mojo"))]
        {
            validation::validate_decode_inputs(
                &self.shape,
                self.packed.len(),
                self.scales.len(),
                self.zeros.len(),
                n,
                group_size,
                bits,
            )?;
            Err("Mojo feature not enabled".to_string())
        }
    }

    pub fn decode(&self, n: usize, group_size: usize, bits: u8) -> Vec<f32> {
        match self.try_decode(n, group_size, bits) {
            Ok(v) => v,
            Err(e) => panic!("MojoQuantizedTensor::decode: invalid input ({e})"),
        }
    }
}

/// GEMV decode kernel: output = W * input
///
/// `weights` is a row-major (rows × cols) matrix.
/// `input` is a (cols,) vector.
/// `output` is a (rows,) vector.
///
/// When the `mojo-ffi` feature is enabled, attempts the Mojo FFI path first.
/// If the Mojo kernel panics or is unavailable, falls back to the reference
/// Rust implementation.
pub fn gemv_decode(weights: &[f32], input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(weights.len(), rows * cols, "weights shape mismatch");
    assert_eq!(input.len(), cols, "input length mismatch");
    assert_eq!(output.len(), rows, "output length mismatch");

    #[cfg(feature = "mojo-ffi")]
    {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            native::mojo_gemv_decode(weights, input, output, rows, cols);
        })) {
            Ok(()) => return,
            Err(_) => eprintln!("Mojo GEMV FFI failed, falling back to Rust"),
        }
    }

    gemv_decode_rust(weights, input, output, rows, cols);
}

fn gemv_decode_rust(weights: &[f32], input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += weights[r * cols + c] * input[c];
        }
        output[r] = sum;
    }
}

/// SIMD-style chunked GEMV decode for benchmarking comparison.
///
/// Processes 32 elements at a time for better cache utilization.
pub fn gemv_decode_rust_simd(
    weights: &[f32],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    assert_eq!(weights.len(), rows * cols, "weights shape mismatch");
    assert_eq!(input.len(), cols, "input length mismatch");
    assert_eq!(output.len(), rows, "output length mismatch");
    const CHUNK: usize = 32;
    for r in 0..rows {
        let mut sum = 0.0f32;
        let mut c = 0;
        while c + CHUNK <= cols {
            let chunk_sum: f32 = (0..CHUNK)
                .map(|i| weights[r * cols + c + i] * input[c + i])
                .sum();
            sum += chunk_sum;
            c += CHUNK;
        }
        while c < cols {
            sum += weights[r * cols + c] * input[c];
            c += 1;
        }
        output[r] = sum;
    }
}

#[cfg(feature = "mojo")]
mod native;
mod validation;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_validation;
