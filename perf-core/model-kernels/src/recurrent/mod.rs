//! Recurrent-family kernels: DeltaNet, short convolution, Mamba scan,
//! RWKV time-mixing.
//!
//! All kernels in this module operate on contiguous `f32` slices and
//! are pure functions of their inputs. Determinism is enforced
//! structurally — no randomness is involved.

pub mod conv;
pub mod deltanet;
pub mod mamba;
pub mod rwkv;

pub use conv::short_conv1d_step;
pub use deltanet::{deltanet_chunk, deltanet_step};
pub use mamba::mamba_scan;
pub use rwkv::rwkv_time_mix;
