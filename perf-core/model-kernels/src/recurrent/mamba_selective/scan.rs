//! Scalar reference kernels for the multi-channel selective state-space
//! scan. See the [`crate::recurrent::mamba_selective`] module docs for
//! the recurrence and the meaning of each parameter.
//!
//! ## Determinism
//!
//! Pure function of inputs. No randomness, no global state.

use crate::error::{KernelError, Result};

use super::params::MambaSelectiveParams;

/// Run the multi-channel selective scan end-to-end.
///
/// `u` has length `n`. `initial_state` has length `state_dim ==
/// a_log.len()` and is mutated in place; the returned vector is the
/// final state (use [`mamba_selective_scan_chunk`] if you need both
/// the stacked outputs and the final state).
pub fn mamba_selective_scan(
    params: &MambaSelectiveParams<'_>,
    u: &[f32],
    initial_state: &mut [f32],
) -> Result<Vec<f32>> {
    let n = u.len();
    let state_dim = params.a_log.len();
    if n == 0 {
        return Err(KernelError::EmptySequence { what: "u" });
    }
    if state_dim == 0 {
        return Err(KernelError::ZeroDimension {
            what: "state_dim (a_log)",
            got: 0,
        });
    }
    if params.dt.len() != n
        || params.b.len() != n
        || params.c.len() != n
        || params.d.len() != n
    {
        return Err(KernelError::BadBufferLength {
            what: "dt/b/c/d",
            expected: n,
            got: params
                .dt
                .len()
                .max(params.b.len())
                .max(params.c.len())
                .max(params.d.len()),
        });
    }
    if initial_state.len() != state_dim {
        return Err(KernelError::BadBufferLength {
            what: "initial_state",
            expected: state_dim,
            got: initial_state.len(),
        });
    }

    let mut ys = Vec::with_capacity(n);
    for t in 0..n {
        // Per-channel decay at this step.
        let dt = params.dt[t];
        // Discretized B: dt * b[t] (Mamba-1 convention).
        let dbu = dt * params.b[t] * u[t];
        // Per-channel skip-gain (D is per-time-step in our convention).
        let d_skip = params.d[t] * u[t];
        // Sum over channels of c[t] * state[c].
        let mut acc = d_skip;
        for c_idx in 0..state_dim {
            let decay = (dt * params.a_log[c_idx].exp()).exp();
            initial_state[c_idx] = decay * initial_state[c_idx] + dbu;
            acc += params.c[t] * initial_state[c_idx];
        }
        ys.push(acc);
    }
    Ok(ys)
}

/// Run the multi-channel selective scan for exactly `chunk_size` steps
/// starting from `initial_state`. Returns `(stacked_outputs,
/// final_state)` where `stacked_outputs.len() == chunk_size`.
pub fn mamba_selective_scan_chunk(
    params: &MambaSelectiveParams<'_>,
    u: &[f32],
    initial_state: &mut [f32],
    chunk_size: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if chunk_size == 0 {
        return Err(KernelError::ZeroDimension {
            what: "chunk_size",
            got: 0,
        });
    }
    if u.len() != chunk_size {
        return Err(KernelError::BadBufferLength {
            what: "u",
            expected: chunk_size,
            got: u.len(),
        });
    }
    let outs = mamba_selective_scan(params, u, initial_state)?;
    Ok((outs, initial_state.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params_1channel<'a>(
        dt: &'a [f32],
        b: &'a [f32],
        c: &'a [f32],
        d: &'a [f32],
    ) -> MambaSelectiveParams<'a> {
        // 1-channel: a_log length 1. Choose a_log so that exp(a_log) = 1,
        // so the decay is exp(dt). With dt = 0 the update reduces to
        // `state += b[t] * u[t]`, y = c[t] * state.
        MambaSelectiveParams {
            dt,
            a_log: &[0.0],
            b,
            c,
            d,
        }
    }

    #[test]
    fn one_channel_reduces_to_explicit_recurrence() {
        // With a single channel, dt = 1, and a_log = [0] (so decay
        // = exp(1 * exp(0)) = exp(1) = e ≈ 2.71828), initial state 0,
        // b = c = 1, d = 0, u = 1: the per-step state update is
        // `state = e * 0 + 1 * 1 * 1 = 1` and the output is
        // `c * state = 1 * 1 = 1`.
        let dt = [1.0f32];
        let b = [1.0f32];
        let c = [1.0f32];
        let d = [0.0f32];
        let u = [1.0f32];
        let mut state = [0.0f32; 1];
        let ys =
            mamba_selective_scan(&params_1channel(&dt, &b, &c, &d), &u, &mut state).unwrap();
        assert!((ys[0] - 1.0).abs() < 1e-5, "got {}", ys[0]);
        assert!((state[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn multi_channel_deterministic_replay() {
        // 3-channel state, 4 time steps. Hand-replay the recurrence in
        // the test to verify the kernel matches the documented
        // recurrence byte-for-byte.
        let state_dim = 3;
        let a_log = [0.1f32, -0.2, 0.05];
        let dt = [0.5, 0.5, 0.5, 0.5];
        let b = [0.1, 0.2, 0.3, 0.4];
        let c = [1.0, 1.0, 1.0, 1.0];
        let d = [0.0, 0.0, 0.0, 0.0];
        let u = [1.0, -1.0, 0.5, 2.0];
        let params = MambaSelectiveParams {
            dt: &dt,
            a_log: &a_log,
            b: &b,
            c: &c,
            d: &d,
        };
        let mut state = vec![0.0f32; state_dim];
        let ys = mamba_selective_scan(&params, &u, &mut state).unwrap();

        // Manual reference replay.
        let mut ref_state = vec![0.0f32; state_dim];
        let mut exp = Vec::new();
        for t in 0..u.len() {
            let dbu = dt[t] * b[t] * u[t];
            let mut acc = d[t] * u[t];
            for c_idx in 0..state_dim {
                let decay = (dt[t] * a_log[c_idx].exp()).exp();
                ref_state[c_idx] = decay * ref_state[c_idx] + dbu;
                acc += c[t] * ref_state[c_idx];
            }
            exp.push(acc);
        }
        for (i, (&g, &e)) in ys.iter().zip(exp.iter()).enumerate() {
            assert!((g - e).abs() < 1e-5, "step {i}: got {g}, expected {e}");
        }
        for (i, (&g, &e)) in state.iter().zip(ref_state.iter()).enumerate() {
            assert!((g - e).abs() < 1e-5, "state[{i}]: got {g}, expected {e}");
        }
    }

    #[test]
    fn chunked_equals_repeated_full_scan() {
        // 2-channel state, 6 time steps; run as one 6-step call and as
        // three 2-step chunks starting from the same initial state. Each
        // chunk receives its own slice of the per-step params.
        let state_dim = 2;
        let a_log_full = [0.0f32, 0.0];
        let dt_full = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let b_full = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let c_full = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let d_full = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let u_full = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

        let full_params = MambaSelectiveParams {
            dt: &dt_full,
            a_log: &a_log_full,
            b: &b_full,
            c: &c_full,
            d: &d_full,
        };
        let mut s_full = vec![0.0f32; state_dim];
        let full_ys = mamba_selective_scan(&full_params, &u_full, &mut s_full).unwrap();

        let (chunk_ys, chunk_state) = {
            let mut acc = Vec::new();
            let mut s = vec![0.0f32; state_dim];
            let mut pos = 0;
            while pos < u_full.len() {
                let cs = 2;
                let chunk_params = MambaSelectiveParams {
                    dt: &dt_full[pos..pos + cs],
                    a_log: &a_log_full,
                    b: &b_full[pos..pos + cs],
                    c: &c_full[pos..pos + cs],
                    d: &d_full[pos..pos + cs],
                };
                let (chunk_out, new_state) = mamba_selective_scan_chunk(
                    &chunk_params,
                    &u_full[pos..pos + cs],
                    &mut s,
                    cs,
                )
                .unwrap();
                acc.extend_from_slice(&chunk_out);
                s = new_state;
                pos += cs;
            }
            (acc, s)
        };
        assert_eq!(full_ys.len(), chunk_ys.len());
        for (i, (&a, &b_v)) in full_ys.iter().zip(chunk_ys.iter()).enumerate() {
            assert!(
                (a - b_v).abs() < 1e-5,
                "step {i}: full {a} vs chunk {b_v}"
            );
        }
        for (i, (&a, &b_v)) in s_full.iter().zip(chunk_state.iter()).enumerate() {
            assert!(
                (a - b_v).abs() < 1e-5,
                "final state[{i}]: {a} vs {b_v}"
            );
        }
    }

    #[test]
    fn rejects_empty_u() {
        let params = MambaSelectiveParams {
            dt: &[],
            a_log: &[0.0],
            b: &[],
            c: &[],
            d: &[],
        };
        let mut state = [0.0f32; 1];
        let err = mamba_selective_scan(&params, &[], &mut state).unwrap_err();
        assert!(matches!(err, KernelError::EmptySequence { .. }));
    }

    #[test]
    fn rejects_zero_state_dim() {
        let params = MambaSelectiveParams {
            dt: &[0.1],
            a_log: &[], // state_dim = 0
            b: &[0.1],
            c: &[0.1],
            d: &[0.0],
        };
        let mut state: [f32; 0] = [];
        let err = mamba_selective_scan(&params, &[1.0], &mut state).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn rejects_length_mismatch() {
        let params = MambaSelectiveParams {
            dt: &[0.1, 0.1, 0.1],
            a_log: &[0.0],
            b: &[0.1, 0.1], // wrong length
            c: &[0.1, 0.1, 0.1],
            d: &[0.0, 0.0, 0.0],
        };
        let mut state = [0.0f32; 1];
        let err = mamba_selective_scan(&params, &[1.0, 2.0, 3.0], &mut state).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }

    #[test]
    fn rejects_state_length_mismatch() {
        let params = MambaSelectiveParams {
            dt: &[0.1],
            a_log: &[0.0, 0.0], // state_dim = 2
            b: &[0.1],
            c: &[0.1],
            d: &[0.0],
        };
        let mut state = [0.0f32; 1];
        let err = mamba_selective_scan(&params, &[1.0], &mut state).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }

    #[test]
    fn rejects_zero_chunk_size() {
        let params = MambaSelectiveParams {
            dt: &[],
            a_log: &[0.0],
            b: &[],
            c: &[],
            d: &[],
        };
        let mut state = [0.0f32; 1];
        let err = mamba_selective_scan_chunk(&params, &[], &mut state, 0).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { .. }));
    }

    #[test]
    fn chunk_rejects_u_length_mismatch() {
        let params = MambaSelectiveParams {
            dt: &[0.1, 0.1, 0.1],
            a_log: &[0.0],
            b: &[0.1, 0.1, 0.1],
            c: &[0.1, 0.1, 0.1],
            d: &[0.0, 0.0, 0.0],
        };
        let mut state = [0.0f32; 1];
        let err =
            mamba_selective_scan_chunk(&params, &[1.0, 2.0], &mut state, 3).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { .. }));
    }
}