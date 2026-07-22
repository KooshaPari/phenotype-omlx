//! Sliding-window mask computation: window range derivation for each
//! query position.
//!
//! Split from `sliding_window.rs` to isolate the mask/offset geometry from
//! the public API and the inner compute kernel.

/// Compute the half-open window `[lo, hi)` for query position `s` in the
/// sliding-window attention formula.
///
/// Given `seq_q`, `seq_k`, `s` (query position), and `window_size`:
/// - `lo = max(0, seq_k - seq_q + s + 1 - window_size)`
/// - `hi = min(seq_k, seq_k - seq_q + s + 1)`
///
/// The window is half-open: column `s+1` itself is the "next" causal step
/// and is excluded. Scores outside the window are set to `-inf` so
/// softmax collapses them to zero mass.
#[inline]
pub fn sliding_window_range(
    seq_q: usize,
    seq_k: usize,
    s: usize,
    window_size: usize,
) -> (usize, usize) {
    let offset = seq_k - seq_q + s;
    let lo = offset
        .checked_add(1)
        .and_then(|x| x.checked_sub(window_size))
        .unwrap_or(0);
    let hi = (offset + 1).min(seq_k);
    (lo, hi)
}
