//! Ternary (Bonsai-style) and sub-byte (2/3/4/5/6/7/8-bit) quantization
//! kernels.
//!
//! Both pack formats are *symmetric* per group: each group of
//! `group_size` values is quantized relative to its own `(min, max)`
//! pair with `bits` bits per element. Ternary is a special case that
//! only carries the sign-magnitude ternary code in 2 bits and stores
//! trivial `scale = 1.0` / `zero = 0.0` per group.
//!
//! All functions are pure: no allocation outside the returned buffers,
//! no global state, deterministic.

pub mod ternary_matmul;

mod subbyte;
mod ternary;

pub use subbyte::{subbyte_pack, subbyte_unpack};
pub use ternary::{ternary_pack, ternary_unpack, SignedTernary};
pub use ternary_matmul::ternary_matmul;

#[cfg(test)]
mod tests;
