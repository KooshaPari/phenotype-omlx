//! Integration tests for `metal-runtime`.
//!
//! These tests exercise the public surface of the `metal-runtime` crate from
//! the outside, exactly as a downstream consumer would. They cover four
//! contracts:
//!
//! 1. Device fingerprinting is stable, distinct, hashable, and round-trips
//!    through serde regardless of platform.
//! 2. The bounded LRU/FIFO cache counts hits/misses/evictions and persists
//!    to disk.
//! 3. The bounded compiler respects the shader-byte and millisecond budget
//!    and emits useful errors when either (or both) are violated.
//! 4. The pipeline compiles + steps a `ModelPlan`, topologically orders its
//!    operators, caches by `(plan_id, plan_revision, fingerprint_hash)`, and
//!    is deterministic across compilations.
//!
//! 27 tests (the spec's checklist lists 27 explicit cases — we follow it
//! exactly).
//!
//! The `mut` keyword on every `let mut cache = ...` is required because
//! `Pipeline::compile(&mut cache)` needs an exclusive borrow; cache-only
//! tests get an `unused_mut` warning that we silence file-wide.
//!
//! This file is the entry point; the per-topic test groups live in sibling
//! modules:
//!
//! - [`fingerprint`] — §1, device fingerprinting contracts (5 tests)
//! - [`cache`]       — §2, bounded LRU/FIFO cache contracts (7 tests)
//! - [`compile`]     — §3, bounded compiler contracts (4 tests)
//! - [`pipeline`]    — §4, end-to-end pipeline contracts (11 tests)

#![allow(unused_mut)]

// The shared `common/` helpers live at `tests/common/mod.rs` (so they stay
// reachable from the other integration-test binaries in this crate:
// `moe.rs`, `property_fuzz.rs`, `soak.rs`). Because this entry point now
// lives one directory deeper (`tests/contracts/main.rs`), the default
// module resolver would look for `tests/contracts/common.rs` and miss it.
// `#[path = ...]` is the explicit, zero-copy fix.
#[path = "../common/mod.rs"]
mod common;

mod cache;
mod compile;
mod fingerprint;
mod pipeline;
