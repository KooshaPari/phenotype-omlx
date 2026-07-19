//! Batched DeltaNet chunked linear-recurrent update.
//!
//! Qwen3-Coder-Next and similar hybrid DeltaNet models evaluate many
//! `(batch, head)` chunks in parallel on Metal. This module exposes a
//! pure-Rust oracle that takes contiguous `[batch, num_heads, chunk,
//! head_dim]` buffers and runs
//! [`crate::recurrent::deltanet::deltanet_step`] per `(batch, head,
//! chunk)` — byte-for-byte identical to running the sequential
//! [`crate::recurrent::deltanet::deltanet_chunk`] `N = batch_size *
//! num_heads` times and stacking the outputs.
//!
//! The oracle is the scalar reference the Metal kernel will be measured
//! against; keeping the two numerically identical means downstream
//! tests can substitute either backend without changing expectations.

use crate::error::{KernelError, Result};
use crate::recurrent::deltanet::{deltanet_chunk, deltanet_step};

/// Run [`deltanet_chunk`] for every `(batch, head)` pair in parallel
/// over a contiguous layout.
///
/// Layouts:
///
/// - `q`, `k`, `v`: row-major `[batch, num_heads, chunk_size, head_dim]`,
///   total length `batch * num_heads * chunk_size * head_dim`.
/// - `initial_state`: row-major `[batch, num_heads, head_dim, head_dim]`,
///   total length `batch * num_heads * head_dim * head_dim`.
/// - Returns `(outputs, final_state)` with the same row-major layouts.
///
/// This function is a thin oracle: every per-`(batch, head)` update
/// delegates to [`deltanet_chunk`], so the result is byte-for-byte
/// identical to running the sequential path `batch_size * num_heads`
/// times. The implementation is intentionally scalar so the optimized
/// Metal kernel can be measured against the same reference.
pub fn deltanet_batched_chunk(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    initial_state: &[f32],
    batch_size: usize,
    num_heads: usize,
    chunk_size: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    // 1. Validate dimensions.
    if batch_size == 0 {
        return Err(KernelError::ZeroDimension {
            what: "batch_size",
            got: 0,
        });
    }
    if num_heads == 0 {
        return Err(KernelError::ZeroDimension {
            what: "num_heads",
            got: 0,
        });
    }
    if chunk_size == 0 {
        return Err(KernelError::ZeroDimension {
            what: "chunk_size",
            got: 0,
        });
    }
    if head_dim == 0 {
        return Err(KernelError::ZeroDimension {
            what: "head_dim",
            got: 0,
        });
    }

    // 2. Validate buffer lengths.
    let expected_qkv = batch_size * num_heads * chunk_size * head_dim;
    if q.len() != expected_qkv || k.len() != expected_qkv || v.len() != expected_qkv {
        return Err(KernelError::BadBufferLength {
            what: "q/k/v",
            expected: expected_qkv,
            got: q.len().max(k.len()).max(v.len()),
        });
    }
    let expected_state = batch_size * num_heads * head_dim * head_dim;
    if initial_state.len() != expected_state {
        return Err(KernelError::BadBufferLength {
            what: "initial_state",
            expected: expected_state,
            got: initial_state.len(),
        });
    }

    // 3. Allocate output buffers up front (single allocation per output).
    let mut outs = Vec::with_capacity(expected_qkv);
    let mut final_states = Vec::with_capacity(expected_state);

    // 4. Run the per-(batch, head) chunk sequentially. Each call
    // delegates to the scalar oracle `deltanet_chunk`, so the resulting
    // outputs and final_states match stacking N=batch_size*num_heads
    // sequential chunks exactly.
    for b in 0..batch_size {
        for h in 0..num_heads {
            let qc = &q[(b * num_heads + h) * chunk_size * head_dim
                ..(b * num_heads + h) * chunk_size * head_dim + chunk_size * head_dim];
            let kc = &k[(b * num_heads + h) * chunk_size * head_dim
                ..(b * num_heads + h) * chunk_size * head_dim + chunk_size * head_dim];
            let vc = &v[(b * num_heads + h) * chunk_size * head_dim
                ..(b * num_heads + h) * chunk_size * head_dim + chunk_size * head_dim];
            let sc = &initial_state[(b * num_heads + h) * head_dim * head_dim
                ..(b * num_heads + h) * head_dim * head_dim + head_dim * head_dim];

            // Use `deltanet_chunk` so the body matches the sequential
            // path exactly — the byte-for-byte oracle contract.
            let (o, s) = deltanet_chunk(qc, kc, vc, chunk_size, head_dim, sc)?;
            outs.extend_from_slice(&o);
            final_states.extend_from_slice(&s);
        }
    }

    Ok((outs, final_states))
}

/// Convenience wrapper that calls [`deltanet_step`] per `(batch, head,
/// chunk)` rather than [`deltanet_chunk`]. Provided as a sanity-check
/// path: the two functions must produce identical results because
/// `deltanet_chunk` is just a `chunk_size`-fold of `deltanet_step`.
/// Exposed for tests and for callers that want to instrument the
/// per-step inner loop directly.
#[allow(dead_code)]
pub(crate) fn deltanet_batched_chunk_stepwise(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    initial_state: &[f32],
    batch_size: usize,
    num_heads: usize,
    chunk_size: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    // Same validation as the primary entry point.
    if batch_size == 0 {
        return Err(KernelError::ZeroDimension {
            what: "batch_size",
            got: 0,
        });
    }
    if num_heads == 0 {
        return Err(KernelError::ZeroDimension {
            what: "num_heads",
            got: 0,
        });
    }
    if chunk_size == 0 {
        return Err(KernelError::ZeroDimension {
            what: "chunk_size",
            got: 0,
        });
    }
    if head_dim == 0 {
        return Err(KernelError::ZeroDimension {
            what: "head_dim",
            got: 0,
        });
    }

    let expected_qkv = batch_size * num_heads * chunk_size * head_dim;
    if q.len() != expected_qkv || k.len() != expected_qkv || v.len() != expected_qkv {
        return Err(KernelError::BadBufferLength {
            what: "q/k/v",
            expected: expected_qkv,
            got: q.len().max(k.len()).max(v.len()),
        });
    }
    let expected_state = batch_size * num_heads * head_dim * head_dim;
    if initial_state.len() != expected_state {
        return Err(KernelError::BadBufferLength {
            what: "initial_state",
            expected: expected_state,
            got: initial_state.len(),
        });
    }

    let mut outs = Vec::with_capacity(expected_qkv);
    let mut final_states = Vec::with_capacity(expected_state);

    for b in 0..batch_size {
        for h in 0..num_heads {
            let qb = &q[(b * num_heads + h) * chunk_size * head_dim
                ..(b * num_heads + h) * chunk_size * head_dim + chunk_size * head_dim];
            let kb = &k[(b * num_heads + h) * chunk_size * head_dim
                ..(b * num_heads + h) * chunk_size * head_dim + chunk_size * head_dim];
            let vb = &v[(b * num_heads + h) * chunk_size * head_dim
                ..(b * num_heads + h) * chunk_size * head_dim + chunk_size * head_dim];

            // Copy the (b, h) initial state slice into a working buffer.
            let sc = &initial_state[(b * num_heads + h) * head_dim * head_dim
                ..(b * num_heads + h) * head_dim * head_dim + head_dim * head_dim];
            let mut state = sc.to_vec();

            for c in 0..chunk_size {
                let qc = &qb[c * head_dim..c * head_dim + head_dim];
                let kc = &kb[c * head_dim..c * head_dim + head_dim];
                let vc = &vb[c * head_dim..c * head_dim + head_dim];
                let o = deltanet_step(qc, kc, vc, &mut state, 0.5, head_dim)?;
                outs.extend_from_slice(&o);
            }
            final_states.extend_from_slice(&state);
        }
    }

    Ok((outs, final_states))
}
