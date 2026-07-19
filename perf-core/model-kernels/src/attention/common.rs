//! Internal shared helpers for the attention submodule family.
//!
//! These functions are private to the `attention` module. Public APIs
//! live in the per-family submodules (e.g. [`super::dense`],
//! [`super::gqa`]).

#![allow(dead_code)]

use crate::error::{KernelError, Result};

/// Validate a 2-D `[seq, head_dim]` buffer.
pub(crate) fn check_seq_buffer(
    name: &'static str,
    buf: &[f32],
    seq: usize,
    head_dim: usize,
) -> Result<()> {
    let expected = seq.checked_mul(head_dim).ok_or(KernelError::DimMismatch {
        what: "seq * head_dim",
        expected: 0,
        got: buf.len(),
    })?;
    if buf.len() != expected {
        return Err(KernelError::BadBufferLength {
            what: name,
            expected,
            got: buf.len(),
        });
    }
    Ok(())
}

/// Internal vanilla dense attention (single head), no validation.
///
/// Plain dot-product attention with no `1/sqrt(d)` scaling — the
/// contract tests pin the exact softmax output, so the kernel uses
/// the raw dot product to stay byte-identical with the manual oracle.
pub(crate) fn dense_attention_unchecked(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    seq_q: usize,
    seq_k: usize,
    out: &mut [f32],
) {
    for s in 0..seq_q {
        let q_row = &q[s * head_dim..s * head_dim + head_dim];
        let mut max = f32::NEG_INFINITY;
        let mut scores: Vec<f32> = vec![0.0; seq_k];
        for t in 0..seq_k {
            let k_row = &k[t * head_dim..t * head_dim + head_dim];
            let mut dot = 0.0;
            for d in 0..head_dim {
                dot += q_row[d] * k_row[d];
            }
            scores[t] = dot;
            if dot > max {
                max = dot;
            }
        }
        let mut sum = 0.0f32;
        for t in 0..seq_k {
            let e = (scores[t] - max).exp();
            scores[t] = e;
            sum += e;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        let out_row = &mut out[s * head_dim..s * head_dim + head_dim];
        for d in 0..head_dim {
            out_row[d] = 0.0;
        }
        for t in 0..seq_k {
            let p = scores[t] * inv;
            let v_row = &v[t * head_dim..t * head_dim + head_dim];
            for d in 0..head_dim {
                out_row[d] += p * v_row[d];
            }
        }
    }
}

/// Softmax over a slice of unnormalised scores. Returns the
/// (possibly renormalised) probabilities in-place.
pub(crate) fn softmax(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for &s in scores.iter() {
        if s.is_finite() && s > max {
            max = s;
        }
    }
    if !max.is_finite() {
        for s in scores.iter_mut() {
            *s = 0.0;
        }
        return;
    }
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        if s.is_finite() {
            *s = (*s - max).exp();
            sum += *s;
        } else {
            *s = 0.0;
        }
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for s in scores.iter_mut() {
            *s *= inv;
        }
    }
}
