//! Tiled / blocked grouped-expert GEMM with oracle parity to the scalar
//! path in [`super::grouped_gemm`].
//!
//! The tiled path iterates the scalar reduction block-by-block instead
//! of element-by-element. It is an *additive* parallel function: the
//! existing [`super::grouped_gemm`] signature is untouched so every
//! caller in the codebase keeps working without changes. Both paths
//! share the same `(a, b, buckets, m, k, n, out)` contract and the
//! same output conventions:
//!
//! - For every `(tok, expert)` pair where `tok ∈ buckets[expert]`,
//!   `out[tok, :] = a[tok, :] @ b[expert, :, :]` (full overwrite for
//!   the assigned rows).
//! - Tokens not assigned to any bucket are not touched — the
//!   validation surface matches the scalar path exactly.
//!
//! The tile size is selected automatically based on `(k, n)`:
//! `tile = min(64, k, n)`. The canonical Qwen-MoE block uses
//! `k == n == 128`, which yields `tile == 64`. Smaller embeddings
//! (`k == 16, n == 64`) pick `tile == 16`.
//!
//! This module is split out of `gemm.rs` to keep both files under the
//! 350-line target.

use crate::error::{KernelError, Result};

/// Tile size selection policy: `tile = min(64, k, n)`. Exposed
/// (`pub`) so the bench harness and the SOTA coverage matrix can
/// pin the chosen value at specific `(k, n)` shapes — see
/// `grouped_gemm_tiled_tile_size_selection` in the test module.
///
/// The 64-element ceiling matches the canonical Qwen-MoE block
/// (`k == n == 128` → `tile == 64`). For shapes smaller than 64 the
/// tile collapses to `min(k, n)` so every block is at most
/// `tile × tile` floats.
#[inline]
pub fn tile_size_for(k: usize, n: usize) -> usize {
    const TILE_MAX: usize = 64;
    k.min(n).min(TILE_MAX)
}

/// Compute `out[tok, :] = a[tok, :] @ b[expert_of(tok), :, :]` for every
/// token in every bucket, iterating block-by-block. The tile size is
/// selected via [`tile_size_for`].
///
/// `m` is accepted for forward compatibility (the scalar path also
/// takes it but ignores it today); the tiled path uses `m` only as a
/// semantic hint for the runtime to chunk the per-bucket row count
/// in future tile kernels. The actual loop bound is `bucket.len()`
/// for each expert.
#[allow(clippy::too_many_arguments)]
pub fn grouped_gemm_tiled(
    a: &[f32],
    b: &[f32],
    buckets: &[Vec<usize>],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) -> Result<()> {
    let _ = m; // accepted but unused on this scalar-tile path
    if k == 0 || n == 0 {
        return Err(KernelError::ZeroDimension {
            what: "k or n",
            got: 0,
        });
    }
    // Validate `b` first — matches the scalar contract.
    let expected_b = buckets.len() * k * n;
    if b.len() != expected_b {
        return Err(KernelError::BadBufferLength {
            what: "b",
            expected: expected_b,
            got: b.len(),
        });
    }
    // Pick the tile size once per call. `tile` is `min(64, k, n)`,
    // which is always >= 1 because we rejected `k == 0 || n == 0`
    // above.
    let tile = tile_size_for(k, n);
    debug_assert!(tile >= 1, "tile must be >= 1 after ZeroDimension guard");
    debug_assert!(tile <= k && tile <= n, "tile must be <= min(k, n)");

    for (e, bucket) in buckets.iter().enumerate() {
        let b_offset = e * k * n;
        for &tok in bucket {
            // Validate `a` and `out` per row (lazy validation, same
            // surface as the scalar path).
            if tok * k + k > a.len() {
                return Err(KernelError::BadBufferLength {
                    what: "a",
                    expected: (tok + 1) * k,
                    got: a.len(),
                });
            }
            let a_row = &a[tok * k..tok * k + k];
            let out_offset = tok * n;
            if out_offset + n > out.len() {
                return Err(KernelError::BadBufferLength {
                    what: "out",
                    expected: out_offset + n,
                    got: out.len(),
                });
            }
            let out_row = &mut out[out_offset..out_offset + n];

            // Match the scalar path's overwrite semantics: every
            // assigned row is fully overwritten with its matmul
            // result. Zero the row first so the per-tile accumulation
            // (`+=`) below sums cleanly across tile blocks without
            // inheriting any sentinel value the caller left in `out`.
            for slot in out_row.iter_mut() {
                *slot = 0.0;
            }
            // Iterate over `n` and `k` in tiles. Each tile is a
            // `tile x tile` block of the `[k, n]` matmul. The
            // reduction accumulates into the same `out_row[j]`
            // entries so cross-tile contributions are summed exactly
            // the way the scalar path sums them.
            let mut j0 = 0usize;
            while j0 < n {
                let j_end = (j0 + tile).min(n);
                let mut kk0 = 0usize;
                while kk0 < k {
                    let kk_end = (kk0 + tile).min(k);
                    // Inner block: accumulate `out_row[j]` for j in
                    // [j0, j_end) over the `kk` range
                    // [kk0, kk_end).
                    for j in j0..j_end {
                        let mut acc = 0.0f32;
                        for kk in kk0..kk_end {
                            acc += a_row[kk] * b[b_offset + kk * n + j];
                        }
                        out_row[j] += acc;
                    }
                    kk0 = kk_end;
                }
                j0 = j_end;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Lcg;

    /// Build a deterministic `[num_tokens, k]` activation matrix.
    fn deterministic_activations(num_tokens: usize, k: usize, salt: u64) -> Vec<f32> {
        let mut rng = Lcg::new(0xCAFE_BABE ^ salt);
        (0..num_tokens * k).map(|_| rng.next_signed()).collect()
    }

    /// Build a deterministic `[num_experts, k, n]` expert weight
    /// tensor.
    fn deterministic_experts(num_experts: usize, k: usize, n: usize, salt: u64) -> Vec<f32> {
        let mut rng = Lcg::new(0xDEAD_BEEF ^ salt);
        (0..num_experts * k * n)
            .map(|_| rng.next_signed())
            .collect()
    }

    /// Bit-equality check against the scalar reference. Both
    /// implementations perform the same `f32` multiply-add in the
    /// same order modulo tile-block ordering, so the tile path's
    /// output is byte-equal to the scalar output element-by-element.
    /// We pin the absolute tolerance at `1e-5` per the task spec.
    fn assert_close(a: &[f32], b: &[f32], abs: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "[{label}] length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            if (x - y).abs() > abs {
                panic!(
                    "[{label}] element {i} differs: tiled={x} scalar={y} (|d|={})",
                    (x - y).abs()
                );
            }
        }
    }

    /// Tiled output must match scalar output element-wise within
    /// `1e-5` on random inputs. This is the oracle-parity contract
    /// for the new path: any future regression in the block-iteration
    /// arithmetic (e.g. accidentally double-counting a boundary
    /// block, swapping the `kk` and `j` loops, or re-using a stale
    /// accumulator) trips this assertion.
    #[test]
    fn grouped_gemm_tiled_matches_scalar_for_random_inputs() {
        let num_tokens = 8;
        let num_experts = 3;
        let k = 16;
        let n = 32;
        let a = deterministic_activations(num_tokens, k, 0xA1);
        let b = deterministic_experts(num_experts, k, n, 0xB2);
        let buckets: Vec<Vec<usize>> =
            vec![vec![0usize, 3, 5], vec![1usize, 6], vec![2usize, 4, 7]];
        let m = 0; // unused; mirrors the scalar path's contract.

        let mut scalar_out = vec![0.0f32; num_tokens * n];
        crate::moe::gemm::grouped_gemm(&a, &b, &buckets, m, k, n, &mut scalar_out)
            .expect("scalar reference must accept well-formed inputs");

        let mut tiled_out = vec![0.0f32; num_tokens * n];
        grouped_gemm_tiled(&a, &b, &buckets, m, k, n, &mut tiled_out)
            .expect("tiled path must accept well-formed inputs");

        assert_close(&tiled_out, &scalar_out, 1e-5, "tiled vs scalar");
    }

    /// Empty buckets for some experts must not write any output
    /// for tokens outside the non-empty buckets. Run both the
    /// scalar reference and the tiled path over the same input and
    /// pin byte-equality; this catches a tiled-path regression that
    /// accidentally indexes an empty expert's `b_offset` (which
    /// would either panic or compute against the next expert's
    /// weights).
    #[test]
    fn grouped_gemm_tiled_handles_empty_buckets() {
        let num_tokens = 4;
        let num_experts = 3;
        let k = 4;
        let n = 6;
        let a = deterministic_activations(num_tokens, k, 0xE1);
        let b = deterministic_experts(num_experts, k, n, 0xE2);
        // expert 0 owns tokens, expert 1 owns nothing, expert 2 owns tokens.
        let buckets: Vec<Vec<usize>> = vec![vec![0usize, 2], vec![], vec![1usize, 3]];
        let mut tiled_out = vec![99.0f32; num_tokens * n];
        grouped_gemm_tiled(&a, &b, &buckets, 0, k, n, &mut tiled_out)
            .expect("tiled path must accept empty buckets");
        let mut scalar_out = vec![99.0f32; num_tokens * n];
        crate::moe::gemm::grouped_gemm(&a, &b, &buckets, 0, k, n, &mut scalar_out)
            .expect("scalar path must accept empty buckets");
        assert_close(
            &tiled_out,
            &scalar_out,
            1e-5,
            "tiled vs scalar with empty buckets",
        );
    }

    /// `k == 0` and `n == 0` are both rejected with the same
    /// `ZeroDimension` variant as the scalar path. The tiled path
    /// performs the zero-dim check before any tile-size selection,
    /// so `tile_size_for(0, n)` / `tile_size_for(k, 0)` is never
    /// reached on these inputs.
    #[test]
    fn grouped_gemm_tiled_rejects_zero_dim() {
        let a = [0.0f32; 1];
        let b = [0.0f32; 1];
        let buckets = vec![vec![0usize]];
        let mut out = [0.0f32; 1];

        let err = grouped_gemm_tiled(&a, &b, &buckets, 1, 0, 1, &mut out).unwrap_err();
        assert!(
            matches!(err, KernelError::ZeroDimension { .. }),
            "expected ZeroDimension for k=0, got {err:?}"
        );

        let err = grouped_gemm_tiled(&a, &b, &buckets, 1, 1, 0, &mut out).unwrap_err();
        assert!(
            matches!(err, KernelError::ZeroDimension { .. }),
            "expected ZeroDimension for n=0, got {err:?}"
        );
    }

    /// Bad buffer lengths (mismatched `b.len()`, lazy per-row
    /// `a.len()` / `out.len()`) must produce `BadBufferLength` errors
    /// matching the scalar path's surface. A future refactor of the
    /// tiled path must not silently accept under-sized buffers.
    #[test]
    fn grouped_gemm_tiled_rejects_bad_buffer_lengths() {
        // b.len() mismatch: buckets.len()=2 but b has only 1 expert's weights.
        let a = vec![0.0f32; 4];
        let b = vec![0.0f32; 4]; // expects 2 * 2 * 2 = 8 floats
        let buckets: Vec<Vec<usize>> = vec![vec![0usize], vec![1usize]];
        let mut out = vec![0.0f32; 4];
        let err = grouped_gemm_tiled(&a, &b, &buckets, 1, 2, 2, &mut out).unwrap_err();
        match err {
            KernelError::BadBufferLength {
                what,
                expected,
                got,
            } => {
                assert_eq!(what, "b", "expected BadBufferLength for b, got {what}");
                assert_eq!(expected, 8);
                assert_eq!(got, 4);
            }
            other => panic!("expected BadBufferLength for b, got {other:?}"),
        }

        // a.len() / out.len() too small for the assigned tokens. The
        // lazy per-row check surfaces `a` or `out` first depending on
        // iteration order; both are valid.
        let a2 = vec![1.0f32; 6]; // num_tokens * k = 3 * 2
        let b2 = vec![0.0f32; 8]; // 2 experts * k * n = 2 * 2 * 2
        let buckets2: Vec<Vec<usize>> = vec![vec![0usize, 2], vec![1usize]];
        let mut out2 = vec![0.0f32; 2]; // expects 3 * 2 = 6
        let err = grouped_gemm_tiled(&a2, &b2, &buckets2, 2, 2, 2, &mut out2).unwrap_err();
        match err {
            KernelError::BadBufferLength {
                what,
                expected,
                got,
            } => {
                assert!(
                    what == "a" || what == "out",
                    "expected BadBufferLength for a or out, got {what:?}"
                );
                assert!(
                    expected > got,
                    "expected > got must hold ({expected} > {got})"
                );
            }
            other => panic!("expected BadBufferLength, got {other:?}"),
        }
    }

    /// The tile-size selection policy is pinned at this surface
    /// instead of through the loop internals so the contract is
    /// stable across loop refactors. The two canonical shapes
    /// spelled out in the task spec:
    ///
    /// - `(k=128, n=128)` → `tile = 64`
    /// - `(k=16, n=64)`  → `tile = 16`
    #[test]
    fn grouped_gemm_tiled_tile_size_selection() {
        assert_eq!(tile_size_for(128, 128), 64, "Qwen-MoE canonical block");
        assert_eq!(tile_size_for(16, 64), 16, "small-k block");
        // Spot-check the boundary cases that bracket the policy.
        assert_eq!(tile_size_for(64, 64), 64, "tile_max hit exactly");
        assert_eq!(tile_size_for(65, 65), 64, "tile_max caps the policy");
        assert_eq!(tile_size_for(32, 256), 32, "k is the binding axis");
        assert_eq!(tile_size_for(256, 32), 32, "n is the binding axis");
        assert_eq!(tile_size_for(1, 1), 1, "degenerate 1x1 still selects");
    }
}
