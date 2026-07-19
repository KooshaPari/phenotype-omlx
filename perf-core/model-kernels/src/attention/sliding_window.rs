//! Sliding-window GQA attention (Qwen3-Next, Mistral, Gemma-2 long-context).
//!
//! For each query position `s` the valid key range is
//! `[max(0, seq_k - seq_q + s - window_size + 1), min(seq_k, seq_k - seq_q + s + 1))`.
//! When `seq_q == seq_k` this collapses to `[s - window_size + 1, s + 1)`
//! clamped into `[0, seq_k)`. Numerical contract matches
//! [`gqa_attention`](crate::attention::gqa::gqa_attention): plain
//! dot-product, no `1/sqrt(d)` scaling.

use crate::attention::common::softmax;
use crate::error::{KernelError, Result};

/// Causal sliding-window GQA attention. `q` is `[seq_q, q_heads, head_dim]`,
/// `k` / `v` are `[seq_k, kv_heads, head_dim]`, `out` is
/// `[seq_q, q_heads, head_dim]`; `window_size` is the per-row width.
#[allow(clippy::too_many_arguments)]
pub fn sliding_window_attention(
    q: &[f32], k: &[f32], v: &[f32],
    q_heads: usize, kv_heads: usize, head_dim: usize,
    seq_q: usize, seq_k: usize, group_size: usize,
    window_size: usize, out: &mut [f32],
) -> Result<()> {
    // Mirror gqa_attention validation order, plus a window_size==0 check.
    if head_dim == 0 { return Err(KernelError::ZeroDimension { what: "head_dim", got: 0 }); }
    if seq_q == 0 { return Err(KernelError::EmptySequence { what: "seq_q" }); }
    if seq_k == 0 { return Err(KernelError::EmptySequence { what: "seq_k" }); }
    if q_heads == 0 { return Err(KernelError::ZeroDimension { what: "q_heads", got: 0 }); }
    if kv_heads == 0 { return Err(KernelError::ZeroDimension { what: "kv_heads", got: 0 }); }
    if group_size == 0 { return Err(KernelError::ZeroDimension { what: "group_size", got: 0 }); }
    if window_size == 0 { return Err(KernelError::ZeroDimension { what: "window_size", got: 0 }); }
    if kv_heads != q_heads / group_size {
        return Err(KernelError::BadGqaGrouping { q_heads, kv_heads });
    }
    let q_len = seq_q * q_heads * head_dim;
    let k_len = seq_k * kv_heads * head_dim;
    if q.len() != q_len {
        return Err(KernelError::BadBufferLength { what: "q", expected: q_len, got: q.len() });
    }
    if k.len() != k_len || v.len() != k_len {
        return Err(KernelError::BadBufferLength {
            what: "k/v", expected: k_len,
            got: if k.len() != k_len { k.len() } else { v.len() },
        });
    }
    if out.len() != q_len {
        return Err(KernelError::BadBufferLength { what: "out", expected: q_len, got: out.len() });
    }
    sliding_window_attention_unchecked(
        q, k, v, q_heads, kv_heads, head_dim, seq_q, seq_k, group_size, window_size, out,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn sliding_window_attention_unchecked(
    q: &[f32], k: &[f32], v: &[f32],
    q_heads: usize, kv_heads: usize, head_dim: usize,
    seq_q: usize, seq_k: usize, group_size: usize,
    window_size: usize, out: &mut [f32],
) {
    // Window per row s ∈ [seq_q):
    //   lo = max(0, seq_k - seq_q + s + 1 - window_size)
    //   hi = min(seq_k, seq_k - seq_q + s + 1)
    // (half-open: column s+1 itself is the "next" causal step and
    // excluded.) Scores outside the window are set to -inf so softmax
    // collapses them to 0 mass.
    for kh in 0..kv_heads {
        for qh_off in 0..group_size {
            let qh = kh * group_size + qh_off;
            for s in 0..seq_q {
                let offset = seq_k - seq_q + s;
                let lo = offset
                    .checked_add(1)
                    .and_then(|x| x.checked_sub(window_size))
                    .unwrap_or(0);
                let hi = (offset + 1).min(seq_k);
                let q_row = &q[s * q_heads * head_dim + qh * head_dim
                    ..s * q_heads * head_dim + qh * head_dim + head_dim];
                let mut scores = vec![f32::NEG_INFINITY; seq_k];
                for t in lo..hi {
                    let k_row = &k[t * kv_heads * head_dim + kh * head_dim
                        ..t * kv_heads * head_dim + kh * head_dim + head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..head_dim { dot += q_row[d] * k_row[d]; }
                    scores[t] = dot;
                }
                softmax(&mut scores);
                let out_row = &mut out[s * q_heads * head_dim + qh * head_dim
                    ..s * q_heads * head_dim + qh * head_dim + head_dim];
                for d in out_row.iter_mut() { *d = 0.0; }
                for t in lo..hi {
                    let p = scores[t];
                    if p == 0.0 { continue; }
                    let v_row = &v[t * kv_heads * head_dim + kh * head_dim
                        ..t * kv_heads * head_dim + kh * head_dim + head_dim];
                    for d in 0..head_dim { out_row[d] += p * v_row[d]; }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic `[seq, heads, head_dim]` tensor with `t[s,h,d] =
    /// (s+1)*17 + (h+1)*5 + (d+1)*3` (all positive so softmax leaves
    /// every in-window score non-zero and masked-out positions stay 0).
    fn tensor(seq: usize, heads: usize, hd: usize) -> Vec<f32> {
        let mut t = Vec::with_capacity(seq * heads * hd);
        for s in 0..seq { for h in 0..heads { for d in 0..hd {
            t.push(((s + 1) * 17 + (h + 1) * 5 + (d + 1) * 3) as f32);
        }}}
        t
    }

    #[allow(clippy::too_many_arguments)]
    fn run_sw(q: &[f32], k: &[f32], v: &[f32], qh_: usize, kvh: usize, hd: usize,
              sq: usize, sk: usize, ws: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; sq * qh_ * hd];
        sliding_window_attention(q, k, v, qh_, kvh, hd, sq, sk, qh_ / kvh, ws, &mut out)
            .expect("sw ok");
        out
    }

    /// Causal GQA reference: for each Q row `s`, attention is over
    /// K[0..=s] only. Matches the upper bound of sliding-window when
    /// `window_size >= seq_k` (the canonical half-open window covers
    /// the entire causal past). Used by test (1) to express the
    /// intended "byte-identical" property.
    fn run_causal_gqa(q: &[f32], k: &[f32], v: &[f32], qh_: usize, kvh: usize, hd: usize,
                      seq: usize) -> Vec<f32> {
        use crate::attention::common::softmax;
        let group = qh_ / kvh;
        let mut out = vec![0.0f32; seq * qh_ * hd];
        for kh in 0..kvh { for qh_off in 0..group { let qh = kh * group + qh_off;
            for s in 0..seq {
                let q_row = &q[s * qh_ * hd + qh * hd..s * qh_ * hd + qh * hd + hd];
                let mut scores = vec![0.0f32; seq];
                for t in 0..=s {
                    let k_row = &k[t * kvh * hd + kh * hd..t * kvh * hd + kh * hd + hd];
                    let mut dot = 0.0f32;
                    for d in 0..hd { dot += q_row[d] * k_row[d]; }
                    scores[t] = dot;
                }
                softmax(&mut scores);
                let out_row = &mut out[s * qh_ * hd + qh * hd..s * qh_ * hd + qh * hd + hd];
                for d in out_row.iter_mut().take(hd) { *d = 0.0; }
                for t in 0..=s {
                    let p = scores[t];
                    if p == 0.0 { continue; }
                    let v_row = &v[t * kvh * hd + kh * hd..t * kvh * hd + kh * hd + hd];
                    for d in 0..hd { out_row[d] += p * v_row[d]; }
                }
            }
        }}
        out
    }

    /// (1) Full-window equivalence: window_size >= seq_k with
    /// seq_q == seq_k is byte-identical to causal GQA (each Q row
    /// attends to K[0..=s]). NOTE: `gqa_attention` in this crate is
    /// non-causal (full-context), so the canonical byte-equivalence
    /// reference is causal GQA — the natural upper bound of the
    /// sliding window formula.
    #[test]
    fn sliding_window_matches_gqa_when_window_is_full() {
        for &(seq, qh_, kvh, hd, ws) in &[
            (4usize, 4usize, 2usize, 3usize, 4usize),
            (8, 8, 2, 64, 8),
            (6, 6, 3, 8, 100),
            (1, 4, 2, 5, 1),
        ] {
            let q = tensor(seq, qh_, hd);
            let k = tensor(seq, kvh, hd);
            let v = tensor(seq, kvh, hd);
            let sw = run_sw(&q, &k, &v, qh_, kvh, hd, seq, seq, ws);
            let causal = run_causal_gqa(&q, &k, &v, qh_, kvh, hd, seq);
            assert_eq!(sw, causal,
                "window={} >= seq={} (seq_q==seq_k) must equal causal GQA",
                ws, seq);
        }
    }

    /// (2) Causal masking: with window=1 and seq_q==seq_k, each row
    /// attends only to K[s] (softmax over one element == 1.0), so
    /// out[s, h, d] must equal V[s, kh, d] where kh = h / group_size.
    /// Concretely for qh_=2, kvh=1, hd=3: out[0, h, *] == V[0, 0, *]
    /// for every query head h, since both qh=0 and qh=1 share kv_head 0.
    #[test]
    fn sliding_window_mask_future_tokens() {
        let (seq, qh_, kvh, hd) = (4usize, 2usize, 1usize, 3usize);
        let q = tensor(seq, qh_, hd);
        let k = tensor(seq, kvh, hd);
        let v = tensor(seq, kvh, hd);
        let out = run_sw(&q, &k, &v, qh_, kvh, hd, seq, seq, 1);
        // V[t, kh, d] = v[t * kvh * hd + kh * hd + d] = v[t * hd + d] here.
        let v0 = |d| v[d]; // V[0, kh=0, d]
        for h in 0..qh_ { for d in 0..hd {
            let got = out[h * hd + d];
            let want = v0(d);
            assert!((got - want).abs() < 1e-5,
                "Q[0,h={h}] window=1 must produce V[0,kh=0,*]: got {got} want {want} at d={d}");
        }}
    }

    /// (3) Window isolation: Q3 attends only to K[2..4].
    #[test]
    fn sliding_window_attends_only_to_window() {
        let (seq, qh_, kvh, hd) = (6usize, 1usize, 1usize, 2usize);
        // V defaults to sentinel=50 except V[2] and V[3] which carry
        // distinct values; Q3 must be a convex combo of those two.
        let mut q = vec![0.0f32; seq * qh_ * hd];
        for d in 0..hd { q[3 * qh_ * hd + d] = 1.0 + d as f32 * 0.1; }
        let mut v = vec![0.0f32; seq * kvh * hd];
        for d in 0..hd {
            v[2 * kvh * hd + d] = 1.0 + d as f32;
            v[3 * kvh * hd + d] = 2.0 + d as f32 * 0.5;
        }
        for x in v.iter_mut() { if *x == 0.0 { *x = 50.0; } }
        let out = run_sw(&q, &vec![0.5; seq * kvh * hd], &v, qh_, kvh, hd, seq, seq, 2);
        for d in 0..hd {
            let got = out[3 * qh_ * hd + d];
            assert!(got < 49.0, "Q3 leaked sentinel: got {got} at d={d}");
            assert!(got > 0.5 && got < 3.0, "Q3 not convex combo: got {got} at d={d}");
        }
        for (d, got) in out.iter().take(hd).copied().enumerate() {
            assert!((got - 50.0).abs() < 1e-4, "Q0 must output V[0]=50: got {got} at d={d}");
        }
    }

    /// (4) Zero window rejected.
    #[test]
    fn sliding_window_zero_width_rejected() {
        let (seq, qh_, kvh, hd) = (2usize, 1usize, 1usize, 1usize);
        let q = tensor(seq, qh_, hd); let k = tensor(seq, kvh, hd);
        let v = tensor(seq, kvh, hd); let mut out = vec![0.0f32; seq * qh_ * hd];
        let err = sliding_window_attention(&q, &k, &v, qh_, kvh, hd, seq, seq, 1, 0, &mut out)
            .unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { what: "window_size", got: 0 }));
    }

    #[test] fn sliding_window_zero_d_rejected() {
        let err = sliding_window_attention(&[], &[], &[], 2, 1, 0, 1, 1, 2, 4, &mut []).unwrap_err();
        assert!(matches!(err, KernelError::ZeroDimension { what: "head_dim", got: 0 }));
    }

    #[test] fn sliding_window_zero_heads_rejected() {
        let err1 = sliding_window_attention(&[], &[], &[], 0, 0, 4, 1, 1, 1, 4, &mut []).unwrap_err();
        assert!(matches!(err1, KernelError::ZeroDimension { what: "q_heads", got: 0 }));
        let err2 = sliding_window_attention(&[], &[], &[], 4, 0, 4, 1, 1, 1, 4, &mut []).unwrap_err();
        assert!(matches!(err2, KernelError::ZeroDimension { what: "kv_heads", got: 0 }));
    }

    #[test] fn sliding_window_bad_grouping_rejected() {
        let q = vec![0.0f32; 4 * 2]; let k = vec![0.0f32; 4 * 2];
        let v = vec![0.0f32; 4 * 2]; let mut out = vec![0.0f32; 4 * 2];
        let err = sliding_window_attention(&q, &k, &v, 4, 3, 2, 4, 4, 1, 2, &mut out).unwrap_err();
        assert!(matches!(err, KernelError::BadGqaGrouping { q_heads: 4, kv_heads: 3 }));
    }

    /// (6) All four buffer-length mismatch surfaces.
    #[test]
    fn sliding_window_buffer_length_mismatches_rejected() {
        let q = tensor(2, 2, 2); let k = tensor(2, 1, 2); let v = tensor(2, 1, 2);
        let cap = q.len();
        let err = sliding_window_attention(
            &vec![0.0; cap - 1], &k, &v, 2, 1, 2, 2, 2, 2, 3, &mut vec![0.0; cap]).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { what: "q", .. }));
        let mut out = vec![0.0; cap];
        let err = sliding_window_attention(
            &q, &[0.0; 3 * 2], &v, 2, 1, 2, 2, 2, 2, 3, &mut out).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { what: "k/v", .. }));
        let err = sliding_window_attention(
            &q, &k, &[0.0; 4 + 1], 2, 1, 2, 2, 2, 2, 3, &mut out).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { what: "k/v", .. }));
        let err = sliding_window_attention(
            &q, &k, &v, 2, 1, 2, 2, 2, 2, 3, &mut vec![0.0; cap - 1]).unwrap_err();
        assert!(matches!(err, KernelError::BadBufferLength { what: "out", .. }));
    }

    /// (7) window > seq_k allowed: must not panic and must equal the
    /// causal-GQA reference (since seq_q == seq_k, the canonical
    /// half-open window already covers the entire causal past).
    #[test]
    fn sliding_window_larger_than_seq_k_does_not_panic() {
        let (seq, qh_, kvh, hd) = (4usize, 2usize, 1usize, 2usize);
        let q = tensor(seq, qh_, hd); let k = tensor(seq, kvh, hd); let v = tensor(seq, kvh, hd);
        let sw = run_sw(&q, &k, &v, qh_, kvh, hd, seq, seq, 1000);
        let causal = run_causal_gqa(&q, &k, &v, qh_, kvh, hd, seq);
        assert_eq!(sw, causal, "window >> seq_k with seq_q==seq_k must equal causal GQA");
    }

    /// (8) Prefill (seq_q == seq_k == 8, window == 4): Q[7] attends
    /// only to K[4..8]; Q[0] only to K[0].
    #[test]
    fn sliding_window_prefill_shape() {
        let (seq, qh_, kvh, hd) = (8usize, 1usize, 1usize, 4usize);
        let q = tensor(seq, qh_, hd);
        let k = tensor(seq, kvh, hd);
        let out = run_sw(&q, &k, &tensor(seq, kvh, hd), qh_, kvh, hd, seq, seq, 4);
        assert_eq!(out.len(), seq * qh_ * hd);
        // Plant sentinel 99 on V[0..4]; Q[7] must NOT see it.
        let mut v_probe = vec![99.0f32; seq * kvh * hd];
        for t in 4..seq { for d in 0..hd {
            v_probe[t * kvh * hd + d] = ((t * 11 + d * 3 + 1) as f32) * 0.1 + 1.0;
        }}
        let out2 = run_sw(&q, &k, &v_probe, qh_, kvh, hd, seq, seq, 4);
        for (d, got) in out2[7 * qh_ * hd..].iter().take(hd).copied().enumerate() {
            assert!(got < 50.0, "Q[7] must not see sentinel V[0..4]=99: got {got} at d={d}");
        }
        for (d, got) in out2.iter().take(hd).copied().enumerate() {
            assert!((got - 99.0).abs() < 1e-4, "Q[0] must output V[0]=99: got {got} at d={d}");
        }
    }

    /// (9) Decode (seq_q == 1, seq_k > window). Window:
    ///   lo = max(0, seq_k - seq_q + 0 - window + 1)
    ///   hi = min(seq_k, seq_k - seq_q + 0 + 1)
    /// With seq_k=6, seq_q=1, window=4 -> [max(0, 2), min(6, 6)) = [2, 6):
    /// the last `window` K positions, exactly as the task spec requires.
    #[test]
    fn sliding_window_decode_shape() {
        let seq_k = 6; let seq_q = 1; let qh_ = 1; let kvh = 1; let hd = 2; let window = 4;
        let q = tensor(seq_q, qh_, hd);
        let k = tensor(seq_k, kvh, hd);
        // Plant sentinel 88 on K positions [0..2) — outside the window
        // — and unique per-(t,d) values on the window positions [2..6).
        let mut v_probe = vec![88.0f32; seq_k * kvh * hd];
        for t in 2..seq_k { for d in 0..hd {
            v_probe[t * kvh * hd + d] = ((t * 11 + d * 3 + 1) as f32) * 0.1 + 1.0;
        }}
        let out = run_sw(&q, &k, &v_probe, qh_, kvh, hd, seq_q, seq_k, window);
        assert_eq!(out.len(), seq_q * qh_ * hd);
        // Output must be a convex combination of V[2..6] — strictly
        // between min and max of those values, and never equal 88.0.
        let mut min_w = f32::INFINITY;
        let mut max_w = f32::NEG_INFINITY;
        for t in 2..seq_k {
            for d in 0..hd {
                let v = v_probe[t * kvh * hd + d];
                if v < min_w { min_w = v; }
                if v > max_w { max_w = v; }
            }
        }
        for (d, got) in out.iter().take(hd).copied().enumerate() {
            assert!((got - 88.0).abs() > 1e-4,
                "decode Q[0] window=[2,6) must not see sentinel V[0..2]=88: got {got}");
            assert!(got >= min_w - 1e-5 && got <= max_w + 1e-5,
                "decode out[{d}]={got} not in convex range [{min_w}, {max_w}]");
        }
    }
}
