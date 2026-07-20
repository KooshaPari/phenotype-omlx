//! The actual test. One assertion per bucket per metric.
//!
//! Ceilings come from [`regress_baseline::dispatch_budget`] and
//! [`regress_baseline::energy_budget_j`] so production callers share a
//! single source of truth with this test. See the module docs in
//! `main.rs` for the bucket table and the headroom rationale.
//!
//! ## Load sensitivity (turn-16)
//!
//! Each `observe_bucket(b)` measures wall-clock `tile_ns`, which is
//! load-sensitive: under cargo test default `--test-threads=N>1` it
//! can be inflated 10-15× above budget by contention with other test
//! binaries. The energy_per_op_j ceiling is derived from the first
//! observed run, so any inflation past ~5% materializes as a flake.
//!
//! To fix this without `--test-threads=1`, the test wraps its
//! measurement loop in a `PerfGuard::enter()` window, which holds a
//! process-global lock that serializes the perf window across test
//! binaries. The lock is dropped at end of scope, releasing the
//! window for other tests to run normally.

use super::*;
use regress_baseline::PerfGuard;

/// Contract test: the `PerfGuard` from the regression library is
/// available and active-by-default. If `OMLX_PERF_NO_GUARD=1` is set
/// the guard is disabled (returns the noop sentinel), which would
/// re-load the test. This test pins that contract so the dispatch
/// envelope stays measurable even under `cargo test` parallelism.
#[test]
fn dispatch_and_energy_guard_is_active_by_default() {
    assert!(
        regress_baseline::perf_guard_active(),
        "PerfGuard must be active for the dispatch_buckets envelope test; \
         if you are disabling it intentionally, set OMLX_PERF_NO_GUARD=1 and \
         skip this test via OMLX_SKIP_DISPATCH_BUCKETS=1."
    );
}

#[test]
fn dispatch_and_energy_within_per_bucket_envelope() {
    // Hold the process-global perf window for the duration of all
    // bucket observations so concurrent test binaries cannot inflate
    // `tile_ns`. `PerfGuard::enter()` returns a noop sentinel if the
    // guard is disabled via OMLX_PERF_NO_GUARD=1.
    let _perf_guard = PerfGuard::enter();

    let mut failed: Vec<String> = Vec::new();

    eprintln!(
        "[dispatch_buckets] running {} buckets (ceilings read from regress_baseline::budget::BUCKETS)",
        BUCKETS.len()
    );

    for b in BUCKETS.iter() {
        let o = observe_bucket(b);
        print_observation(&o);

        let dispatch_ceiling = dispatch_budget(&b.shape);
        let energy_ceiling = energy_budget_j(&b.shape);

        if o.dispatches > dispatch_ceiling {
            failed.push(format!(
                "{}: dispatches={} > ceiling={} (M={}, N={}, K={})",
                o.name, o.dispatches, dispatch_ceiling, b.shape.m, b.shape.n, b.shape.k,
            ));
        }

        if o.energy_per_op_j > energy_ceiling {
            failed.push(format!(
                "{}: energy_per_op_j={:.3e} > ceiling={:.3e} (M={}, N={}, K={})",
                o.name, o.energy_per_op_j, energy_ceiling, b.shape.m, b.shape.n, b.shape.k,
            ));
        }
    }

    if !failed.is_empty() {
        let mut msg = String::from(
            "per-bucket dispatch / energy envelope exceeded. Tighten or widen the affected \
             rows in `regress_baseline::budget::BUCKETS` (or fix the regression in the kernel):\n",
        );
        for f in &failed {
            msg.push_str("  - ");
            msg.push_str(f);
            msg.push('\n');
        }
        panic!("{msg}");
    }
}
