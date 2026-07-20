//! DeltaNet chunked linear-recurrent update (Qwen3-Coder-Next style).
//!
//! Canonical DeltaNet update for head_dim `d`:
//!
//! ```text
//! state'[i, j] = state[i, j] - beta * k[i] * sum_p k[p] * state[p, j]
//!              + beta * v[i] * k[j]
//! out[j]      = sum_i q[i] * state'[i, j]
//! ```
//!
//! `state` is laid out row-major `[head_dim, head_dim]`.

use crate::error::{KernelError, Result};

/// Run one chunked DeltaNet step. Mutates `state` in place and returns
/// the output vector `[head_dim]`. Beta is fixed at `0.5` per the
/// reference implementation; pass it via the [`deltanet_chunk`]
/// boundary if a different value is needed (TODO: thread beta through).
pub fn deltanet_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    state: &mut [f32],
    beta: f32,
    head_dim: usize,
) -> Result<Vec<f32>> {
    if head_dim == 0 {
        return Err(KernelError::ZeroDimension { what: "head_dim", got: 0 });
    }
    let expected_state = head_dim * head_dim;
    if state.len() != expected_state {
        return Err(KernelError::BadBufferLength {
            what: "state",
            expected: expected_state,
            got: state.len(),
        });
    }
    if q.len() != head_dim || k.len() != head_dim || v.len() != head_dim {
        return Err(KernelError::BadBufferLength {
            what: "q/k/v",
            expected: head_dim,
            got: q.len().max(k.len()).max(v.len()),
        });
    }
    let mut new_state = state.to_vec();
    for i in 0..head_dim {
        for j in 0..head_dim {
            let mut kk = 0.0;
            for p in 0..head_dim {
                kk += k[p] * state[p * head_dim + j];
            }
            new_state[i * head_dim + j] = state[i * head_dim + j]
                - beta * k[i] * kk
                + beta * v[i] * k[j];
        }
    }
    state.copy_from_slice(&new_state);
    // Output is `q^T @ state`, i.e. `out[i] = sum_j q[j] * state[j, i]`
    // (matches the contract test's reference computation).
    let mut out = vec![0.0f32; head_dim];
    for i in 0..head_dim {
        let mut acc = 0.0;
        for j in 0..head_dim {
            acc += q[j] * new_state[j * head_dim + i];
        }
        out[i] = acc;
    }
    Ok(out)
}

/// Run `chunk_size` sequential [`deltanet_step`]s starting from
/// `initial_state`. Returns the stacked outputs `[chunk_size, head_dim]`
/// and the final state.
pub fn deltanet_chunk(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    chunk_size: usize,
    head_dim: usize,
    initial_state: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    if chunk_size == 0 {
        return Err(KernelError::ZeroDimension {
            what: "chunk_size",
            got: 0,
        });
    }
    let expected = chunk_size * head_dim;
    if q.len() != expected || k.len() != expected || v.len() != expected {
        return Err(KernelError::BadBufferLength {
            what: "q/k/v",
            expected,
            got: q.len().max(k.len()).max(v.len()),
        });
    }
    if initial_state.len() != head_dim * head_dim {
        return Err(KernelError::BadBufferLength {
            what: "initial_state",
            expected: head_dim * head_dim,
            got: initial_state.len(),
        });
    }
    let mut state = initial_state.to_vec();
    let mut outs = Vec::with_capacity(expected);
    for c in 0..chunk_size {
        let qc = &q[c * head_dim..c * head_dim + head_dim];
        let kc = &k[c * head_dim..c * head_dim + head_dim];
        let vc = &v[c * head_dim..c * head_dim + head_dim];
        let o = deltanet_step(qc, kc, vc, &mut state, /* beta */ 0.5, head_dim)?;
        outs.extend_from_slice(&o);
    }
    Ok((outs, state))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_head_dim() {
        let mut s: [f32; 0] = [];
        let err = deltanet_step(&[], &[], &[], &mut s, 0.5, 0).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn rejects_wrong_state_length() {
        let mut s = [0.0f32; 3];
        let err = deltanet_step(&[0.0; 2], &[0.0; 2], &[0.0; 2], &mut s, 0.5, 2).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }
}
