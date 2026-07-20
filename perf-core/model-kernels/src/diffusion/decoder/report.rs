//! `DiffusionStepReport`: per-step summary returned by
//! [`super::step`].
//!
//! See the parent module docs for the LLaDA / Dream acceptance trace
//! algorithm.

/// Report of one step of a diffusion decoder trace.
///
/// Returned by [`super::DiffusionDecoder::step`]. The caller drives the outer
/// loop: the decoder does not run for a fixed number of internal steps,
/// it just exposes one step at a time so callers can inspect the
/// intermediate state.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionStepReport {
    /// 0-indexed step number (the order in which the caller invoked
    /// [`super::DiffusionDecoder::step`]).
    pub step: usize,
    /// Number of positions that are *not* masked after the step
    /// (i.e. net newly-accepted positions after remask).
    pub accepted_count: usize,
    /// Number of positions that were re-masked during the step
    /// (i.e. transitioned from `false -> true` between input mask and
    /// output mask).
    pub remasked_count: usize,
    /// `true` iff every position is unmasked (`mask[i] == false` for
    /// all `i`). The trace has finished when this is `true`.
    pub finished: bool,
}
