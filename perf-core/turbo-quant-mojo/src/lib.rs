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
        native::mojo_encode(data, bits, group_size)
    }

    pub fn try_decode(&self, n: usize, group_size: usize, bits: u8) -> Result<Vec<f32>, String> {
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

    pub fn decode(&self, n: usize, group_size: usize, bits: u8) -> Vec<f32> {
        match self.try_decode(n, group_size, bits) {
            Ok(v) => v,
            Err(e) => panic!("MojoQuantizedTensor::decode: invalid input ({e})"),
        }
    }
}

mod native;
mod validation;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_validation;
