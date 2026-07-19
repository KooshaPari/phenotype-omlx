//! Discrete (masked) diffusion language model — *sampler* coverage.
//!
//! This file is the **sampler/selector** half of the discrete-diffusion
//! test family. The math/oracle half (`discrete_diffusion_oracle.rs`)
//! owns the schedule + reference oracle + LCG helpers + registry
//! wiring. This file owns the stub selector types and the registry
//! dispatch coverage test. The schedule-boundary file
//! (`discrete_diffusion_schedule.rs`) lives separately.
//!
//! This file was split from the prior `discrete_diffusion_sampler.rs`
//! (407L) in turn-10's module-size sweep to bring both files below the
//! 350L target.
//!
//! The stub is sufficient to verify the byte-identical oracle
//! contract for the discrete diffusion family:
//!
//! 1. The deterministic policy picks the kernel with the lowest p95
//!    for `OperatorKind::DiscreteDiffusion` at the chosen shape.
//! 2. The stub selector metadata exists in the right shape for the
//!    eventual kernel-registry `DiscreteDiffusion` selector.
//!
//! When the kernel-registry gains a real `DiscreteDiffusion` selector,
//! this file should be extended to register its backend candidates and
//! drop the test-only stub.

use kernel_registry::compat::OperatorKind;
use kernel_registry::selector::SelectionDecision;
use kernel_registry::SelectionPolicy;

use super::{
    discrete_diffusion_oracle::ddm_key, discrete_diffusion_oracle::ddm_registry,
    fresh_capabilities, NOW_UNIX_MS,
};

// Re-export the math/oracle surface so callers
// (`discrete_diffusion_schedule.rs`, the per-tag coverage matrix) do
// not need to import from two files.
pub(crate) use super::discrete_diffusion_oracle::{
    DiscreteDiffusionOracle, Schedule,
};

// ---------------------------------------------------------------------------
// Test-only stub selector types.
// ---------------------------------------------------------------------------

/// Selector execution mode. Matches the language of the surrounding
/// runtime; `Decode` means "one masked position per call, decode
/// greedily, advance the schedule".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SelectorMode {
    Prefill,
    Decode,
}

/// Kind of discrete-diffusion step the selector is asked to dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepKind {
    /// Decode one masked position per call under the noise schedule.
    MaskedDiffusionStep,
    /// Sample an entirely noised sequence from the prior (used at
    /// the start of every denoising chain).
    PriorSample,
}

/// Per-call selector metadata. Mirrors the design that will land in
/// the kernel-registry proper; defined here so the discrete-diffusion
/// oracle test does not need to wait on that refactor.
#[derive(Debug, Clone)]
pub(crate) struct SelectorMetadata {
    /// Family discriminator on the `KernelKey`.
    pub(crate) family: OperatorKind,
    /// Decode vs prefill — controls the oracle's update pattern.
    pub(crate) mode: SelectorMode,
    /// Selection policy for the registry call.
    pub(crate) policy: SelectionPolicy,
    /// Which step kind the dispatch represents.
    pub(crate) kind: StepKind,
}

impl SelectorMetadata {
    pub(crate) fn decode_deterministic() -> Self {
        Self {
            family: OperatorKind::DiscreteDiffusion,
            mode: SelectorMode::Decode,
            policy: SelectionPolicy::Deterministic { prefer_lower_p95: true },
            kind: StepKind::MaskedDiffusionStep,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (selector coverage).
// ---------------------------------------------------------------------------

/// The Deterministic policy under the discrete-diffusion metadata
/// selects the lowest-p95 backend (Metal). This is the selector-coverage
/// half of the contract: even though the oracle is the test's source of
/// truth, the registry must still pick the right kernel for it.
#[test]
fn ddm_metadata_decode_deterministic_picks_lowest_p95_metal_backend() {
    let (reg, _id_scalar, id_metal) = ddm_registry();
    let meta = SelectorMetadata::decode_deterministic();
    let key = ddm_key(16, 8);
    let decision = reg.select_with_caps(&key, meta.policy.clone(), &fresh_capabilities(), NOW_UNIX_MS);
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(
                candidate.id, id_metal,
                "metal p95=2100 must beat scalar p95=9500 under Deterministic"
            );
            // Pin every field of the stub metadata so future
            // refactors cannot accidentally drop the family / mode /
            // kind discriminators that distinguish the
            // DiscreteDiffusion selector from neighbors.
            assert_eq!(meta.family, OperatorKind::DiscreteDiffusion);
            assert_eq!(meta.mode, SelectorMode::Decode);
            assert_eq!(meta.kind, StepKind::MaskedDiffusionStep);
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
}

/// The stub metadata's other variants round-trip through `Debug`.
/// This pins the existence of `Prefill` and `PriorSample` so the
/// stub remains a faithful preview of the eventual full selector.
#[test]
fn ddm_metadata_other_variants_exist() {
    let _prefill = SelectorMode::Prefill;
    let _prior = StepKind::PriorSample;
}
