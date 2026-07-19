//! Recurrent-family kernels — single-file facade (spec: `recurrent.rs`).
//!
//! Re-exports the recurrent operators implemented under `recurrent/`:
//!
//! - [`deltanet_step`] / [`deltanet_chunk`] — DeltaNet recurrent update
//!   (Qwen-style linear attention) with a `[head_dim, head_dim]` running
//!   state.
//! - [`short_conv1d_step`] — LFM2-style gated short convolution with a
//!   lazily-grown state buffer.
//! - [`mamba_scan`] — selective state-space (Mamba) recurrence
//!   `y[t] = a * y[t-1] + b * u[t]`.
//! - [`rwkv_time_mix`] — RWKV time-mixing update with per-channel mix
//!   coefficients.
//!
//! All functions operate on contiguous `&[f32]` slices with state passed
//! in/out as `&mut [f32]`. Determinism is structural: there is no
//! randomness.

pub use crate::recurrent::conv::{gated_short_conv1d_step, short_conv1d_step};
pub use crate::recurrent::deltanet::{deltanet_chunk, deltanet_step};
pub use crate::recurrent::deltanet_batched::deltanet_batched_chunk;
pub use crate::recurrent::mamba::mamba_scan;
pub use crate::recurrent::mamba_selective::{
    mamba_selective_scan, mamba_selective_scan_chunk, MambaSelectiveParams,
};
pub use crate::recurrent::rwkv::{rwkv7_time_mix, rwkv_time_mix};
