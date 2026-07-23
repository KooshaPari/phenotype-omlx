//! Paged attention (vLLM / SGLang style).

use crate::attention::common::dense_attention_unchecked;
use crate::error::{KernelError, Result};

/// Paged attention.
///
/// `k_cache` and `v_cache` are flat `[num_blocks, block_size, kv_heads, head_dim]`.
/// `block_tables` enumerates `(block_id, intra_block_offset)` pairs the
/// query attends to. The number of gathered keys is taken from
/// `block_tables.len()`; the `seq_k` argument is accepted for symmetry
/// with [`crate::attention::dense_attention`] and must be zero or
/// equal to that length.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    block_tables: &[(usize, usize)],
    block_size: usize,
    kv_heads: usize,
    head_dim: usize,
    seq_q: usize,
    seq_k: usize,
    out: &mut [f32],
) -> Result<()> {
    if block_size == 0 {
        return Err(KernelError::ZeroDimension {
            what: "block_size",
            got: 0,
        });
    }
    if head_dim == 0 {
        return Err(KernelError::ZeroDimension {
            what: "head_dim",
            got: 0,
        });
    }
    if kv_heads == 0 {
        return Err(KernelError::ZeroDimension {
            what: "kv_heads",
            got: 0,
        });
    }
    if seq_q == 0 {
        return Err(KernelError::EmptySequence { what: "seq_q" });
    }
    // `block_tables.len()` is the authoritative number of gathered keys.
    // `seq_k` is accepted for symmetry with [`crate::attention::dense_attention`]
    // and is honoured only when it equals that length; passing `seq_k = 0`
    // also works and is interpreted as "unknown caller-side length".
    if seq_k != 0 && seq_k != block_tables.len() {
        return Err(KernelError::DimMismatch {
            what: "block_tables.len() vs seq_k",
            expected: seq_k,
            got: block_tables.len(),
        });
    }
    if block_tables.is_empty() {
        return Err(KernelError::EmptySequence {
            what: "block_tables",
        });
    }
    let per_block = block_size * kv_heads * head_dim;
    for &(bid, off) in block_tables {
        if off >= block_size {
            return Err(KernelError::DimMismatch {
                what: "intra_block_offset",
                expected: block_size,
                got: off,
            });
        }
        let base = bid
            .checked_mul(per_block)
            .and_then(|x| x.checked_add(off * kv_heads * head_dim))
            .ok_or(KernelError::BadBufferLength {
                what: "block_table index",
                expected: k_cache.len(),
                got: bid,
            })?;
        if base + kv_heads * head_dim > k_cache.len() {
            return Err(KernelError::BadBufferLength {
                what: "k_cache",
                expected: k_cache.len(),
                got: base + kv_heads * head_dim,
            });
        }
    }
    let q_len = seq_q * head_dim;
    if q.len() != q_len {
        return Err(KernelError::BadBufferLength {
            what: "q",
            expected: q_len,
            got: q.len(),
        });
    }
    if out.len() != q_len {
        return Err(KernelError::BadBufferLength {
            what: "out",
            expected: q_len,
            got: out.len(),
        });
    }
    let n = block_tables.len();
    let mut k_gathered: Vec<f32> = Vec::with_capacity(n * head_dim);
    let mut v_gathered: Vec<f32> = Vec::with_capacity(n * head_dim);
    for &(bid, off) in block_tables {
        let base = bid * block_size * kv_heads * head_dim + off * kv_heads * head_dim;
        k_gathered.extend_from_slice(&k_cache[base..base + head_dim]);
        v_gathered.extend_from_slice(&v_cache[base..base + head_dim]);
    }
    dense_attention_unchecked(q, &k_gathered, &v_gathered, head_dim, seq_q, n, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_mismatched_block_table() {
        // seq_k = 2 but block_tables has 1 entry.
        let q = [0.0f32; 1];
        let k = [0.0f32; 4];
        let v = [0.0f32; 4];
        let block_tables = vec![(0usize, 0usize)];
        let err =
            paged_attention(&q, &k, &v, &block_tables, 2, 1, 1, 1, 2, &mut [0.0; 1]).unwrap_err();
        assert!(matches!(err, KernelError::DimMismatch { .. }));
    }

    #[test]
    fn rejects_out_of_range_intra_offset() {
        let q = [0.0f32; 1];
        let k = [0.0f32; 4];
        let v = [0.0f32; 4];
        let block_tables = vec![(0usize, 5usize)]; // offset >= block_size
        let err =
            paged_attention(&q, &k, &v, &block_tables, 2, 1, 1, 1, 1, &mut [0.0; 1]).unwrap_err();
        assert!(matches!(err, KernelError::DimMismatch { .. }));
    }
}
