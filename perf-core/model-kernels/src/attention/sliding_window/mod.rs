//! Sliding-window GQA attention (Qwen3-Next, Mistral, Gemma-2 long-context).
//!
//! For each query position `s` the valid key range is
//! `[max(0, seq_k - seq_q + s - window_size + 1), min(seq_k, seq_k - seq_q + s + 1))`.
//! When `seq_q == seq_k` this collapses to `[s - window_size + 1, s + 1)`
//! clamped into `[0, seq_k)`. Numerical contract matches
//! [`gqa_attention`](crate::attention::gqa::gqa_attention): plain
//! dot-product, no `1/sqrt(d)` scaling.

mod sliding_window_mask;
mod sliding_window_ops;

pub use sliding_window_mask::sliding_window_range;

use crate::error::{KernelError, Result};
use sliding_window_ops::sliding_window_attention_unchecked;

/// Causal sliding-window GQA attention. `q` is `[seq_q, q_heads, head_dim]`,
/// `k` / `v` are `[seq_k, kv_heads, head_dim]`, `out` is
/// `[seq_q, q_heads, head_dim]`; `window_size` is the per-row width.
#[allow(clippy::too_many_arguments)]
pub fn sliding_window_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    seq_q: usize,
    seq_k: usize,
    group_size: usize,
    window_size: usize,
    out: &mut [f32],
) -> Result<()> {
    if head_dim == 0 {
        return Err(KernelError::ZeroDimension {
            what: "head_dim",
            got: 0,
        });
    }
    if seq_q == 0 {
        return Err(KernelError::EmptySequence { what: "seq_q" });
    }
    if seq_k == 0 {
        return Err(KernelError::EmptySequence { what: "seq_k" });
    }
    if q_heads == 0 {
        return Err(KernelError::ZeroDimension {
            what: "q_heads",
            got: 0,
        });
    }
    if kv_heads == 0 {
        return Err(KernelError::ZeroDimension {
            what: "kv_heads",
            got: 0,
        });
    }
    if group_size == 0 {
        return Err(KernelError::ZeroDimension {
            what: "group_size",
            got: 0,
        });
    }
    if window_size == 0 {
        return Err(KernelError::ZeroDimension {
            what: "window_size",
            got: 0,
        });
    }
    if kv_heads != q_heads / group_size {
        return Err(KernelError::BadGqaGrouping { q_heads, kv_heads });
    }
    let q_len = seq_q * q_heads * head_dim;
    let k_len = seq_k * kv_heads * head_dim;
    if q.len() != q_len {
        return Err(KernelError::BadBufferLength {
            what: "q",
            expected: q_len,
            got: q.len(),
        });
    }
    if k.len() != k_len || v.len() != k_len {
        return Err(KernelError::BadBufferLength {
            what: "k/v",
            expected: k_len,
            got: if k.len() != k_len { k.len() } else { v.len() },
        });
    }
    if out.len() != q_len {
        return Err(KernelError::BadBufferLength {
            what: "out",
            expected: q_len,
            got: out.len(),
        });
    }
    sliding_window_attention_unchecked(
        q,
        k,
        v,
        q_heads,
        kv_heads,
        head_dim,
        seq_q,
        seq_k,
        group_size,
        window_size,
        out,
    );
    Ok(())
}

#[cfg(test)]
mod tests;
