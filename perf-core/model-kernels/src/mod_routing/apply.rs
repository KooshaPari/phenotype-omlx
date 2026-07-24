use super::route::ModRoutePlan;
use crate::error::{KernelError, Result};

/// Materialize the surviving rows of `full_hidden_states` as a
/// contiguous `[k, dim]` buffer.
///
/// `full_hidden_states` is laid out `[num_tokens, dim]` row-major.
/// `dim` must be strictly positive; `full_hidden_states.len()` must
/// equal `plan.selected_tokens.len().max(num_tokens) * dim` modulo
/// rows that have not been read — we validate by ensuring the plan's
/// every selected index is in-range and that the buffer length is
/// consistent with the largest referenced index.
pub fn mod_apply(plan: &ModRoutePlan, full_hidden_states: &[f32], dim: usize) -> Result<Vec<f32>> {
    if dim == 0 {
        return Err(KernelError::ZeroDimension {
            what: "dim",
            got: 0,
        });
    }
    let k = plan.selected_tokens.len();
    let mut out = Vec::with_capacity(k.saturating_mul(dim));
    for &idx in &plan.selected_tokens {
        let row = idx as usize;
        let start = row.checked_mul(dim).ok_or(KernelError::BadBufferLength {
            what: "selected_token row index * dim",
            expected: 0,
            got: row * dim,
        })?;
        let end = start + dim;
        if end > full_hidden_states.len() {
            return Err(KernelError::BadBufferLength {
                what: "full_hidden_states",
                expected: end,
                got: full_hidden_states.len(),
            });
        }
        out.extend_from_slice(&full_hidden_states[start..end]);
    }
    Ok(out)
}

/// Inverse of [`mod_apply`]: scatter the processed rows back into a
/// full-size buffer, filling every position that was skipped with
/// `fill`.
///
/// Returns the full-size buffer; the caller does not need to allocate
/// it. Skipped positions are filled with `fill` (typically `0.0` so
/// the residual stream carries the previous value untouched, or a
/// residual scaling factor when the model uses weighted carries).
///
/// `full_len` is the *number of tokens* in the full buffer (so the
/// returned buffer has `full_len * dim` elements).
pub fn mod_scatter_back(
    selected: &[f32],
    plan: &ModRoutePlan,
    full_len: usize,
    dim: usize,
    fill: f32,
) -> Result<Vec<f32>> {
    if dim == 0 {
        return Err(KernelError::ZeroDimension {
            what: "dim",
            got: 0,
        });
    }
    if selected.len() != plan.selected_tokens.len() * dim {
        return Err(KernelError::BadBufferLength {
            what: "selected",
            expected: plan.selected_tokens.len() * dim,
            got: selected.len(),
        });
    }
    let mut out = vec![fill; full_len * dim];
    for (slot, &idx) in plan.selected_tokens.iter().enumerate() {
        let row = idx as usize;
        if row >= full_len {
            return Err(KernelError::BadBufferLength {
                what: "selected_token row index",
                expected: full_len,
                got: row,
            });
        }
        let src_start = slot * dim;
        let dst_start = row * dim;
        out[dst_start..dst_start + dim].copy_from_slice(&selected[src_start..src_start + dim]);
    }
    Ok(out)
}
