//! Recurrent-family kernels: DeltaNet, short convolution, Mamba scan,
//! RWKV time-mixing.
//!
//! All kernels in this module operate on contiguous `f32` slices and
//! are pure functions of their inputs. Determinism is enforced
//! structurally — no randomness is involved.

pub mod conv;
pub mod deltanet;
pub mod mamba;
pub mod mamba_selective;
pub mod rwkv;

pub use conv::{gated_short_conv1d_step, short_conv1d_step};
pub use deltanet::{deltanet_chunk, deltanet_step};
pub use mamba::mamba_scan;
pub use mamba_selective::{mamba_selective_scan, mamba_selective_scan_chunk, MambaSelectiveParams};
pub use rwkv::{rwkv7_time_mix, rwkv_time_mix};
