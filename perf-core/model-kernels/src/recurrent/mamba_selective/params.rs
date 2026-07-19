//! Parameter struct for the multi-channel selective state-space scan
//! (Mamba-style). See the [`crate::recurrent::mamba_selective`]
//! module docs for the recurrence.

/// Parameters for the selective state-space scan.
///
/// All slices must outlive the call to
/// [`crate::recurrent::mamba_selective::mamba_selective_scan`] /
/// [`crate::recurrent::mamba_selective::mamba_selective_scan_chunk`].
/// Lengths:
///
/// - `dt`, `b`, `c`, `d` are all `[n]` (per time-step),
/// - `a_log` is `[state_dim]` (per channel).
///
/// `state_dim` is the length of the per-channel state vector and is
/// derived from `a_log.len()` at call time — there is no separate
/// `state_dim` field.
pub struct MambaSelectiveParams<'a> {
    /// Per-time-step selective Δ (discretization step) `[n]`.
    pub dt: &'a [f32],
    /// Per-channel log-A `[state_dim]`. Convention: `A_c = -exp(a_log[c])`
    /// so the per-channel decay at time `t` is
    /// `exp(dt[t] * exp(a_log[c]))`.
    pub a_log: &'a [f32],
    /// Per-time-step B (input projection) `[n]`.
    pub b: &'a [f32],
    /// Per-time-step C (output projection) `[n]`.
    pub c: &'a [f32],
    /// Per-time-step D (residual / skip) `[n]`.
    pub d: &'a [f32],
}