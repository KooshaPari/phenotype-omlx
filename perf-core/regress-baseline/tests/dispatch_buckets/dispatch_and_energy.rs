//! The actual test. One assertion per bucket per metric. On the first
//! run at least one bucket will trip a ceiling, the test will fail, the
//! operator reads the `eprintln!` lines, and updates `DISPATCH_CEIL` /
//! `ENERGY_PER_OP_CEIL_J` (and, in a follow-up commit,
//! `dispatch_budget` / `energy_budget_j`).

use super::*;

#[test]
fn dispatch_and_energy_within_per_bucket_envelope() {
    assert_eq!(
        BUCKETS.len(),
        DISPATCH_CEIL.len(),
        "DISPATCH_CEIL must have one entry per bucket"
    );
    assert_eq!(
        BUCKETS.len(),
        ENERGY_PER_OP_CEIL_J.len(),
        "ENERGY_PER_OP_CEIL_J must have one entry per bucket"
    );

    let mut failed: Vec<String> = Vec::new();

    eprintln!(
        "[dispatch_buckets] running {} buckets (initial envelope; tighten in a follow-up commit once instrumented telemetry is wired in)",
        BUCKETS.len()
    );

    for (idx, b) in BUCKETS.iter().enumerate() {
        let o = observe_bucket(b);
        print_observation(&o);

        let dispatch_ceil = DISPATCH_CEIL[idx];
        let energy_ceil = ENERGY_PER_OP_CEIL_J[idx];

        if o.dispatches > dispatch_ceil {
            failed.push(format!(
                "{}: dispatches={} > ceil={} (M={}, N={}, K={})",
                o.name, o.dispatches, dispatch_ceil, b.shape.m, b.shape.n, b.shape.k,
            ));
        }

        if o.energy_per_op_j > energy_ceil {
            failed.push(format!(
                "{}: energy_per_op_j={:.3e} > ceil={:.3e} (M={}, N={}, K={})",
                o.name, o.energy_per_op_j, energy_ceil, b.shape.m, b.shape.n, b.shape.k,
            ));
        }

        // Sanity: the library-side stubs are wired up. They currently
        // return u64::MAX / f64::INFINITY; the follow-up commit will
        // tighten them. We only assert the structural property here:
        // the call must not panic, must return a finite or sentinel
        // value, and must be independent of the per-bucket ceilings.
        assert!(
            o.stub_dispatch_budget >= o.dispatches
                || o.stub_dispatch_budget == u64::MAX,
            "{}: stub dispatch_budget ({}) must be >= observed dispatches ({}) until tightened",
            o.name,
            o.stub_dispatch_budget,
            o.dispatches,
        );
        assert!(
            o.stub_energy_budget_j.is_infinite() || o.stub_energy_budget_j >= o.energy_per_op_j,
            "{}: stub energy_budget_j ({}) must be infinite or >= observed energy_per_op_j ({:.3e}) until tightened",
            o.name,
            o.stub_energy_budget_j,
            o.energy_per_op_j,
        );
    }

    if !failed.is_empty() {
        let mut msg = String::from(
            "per-bucket dispatch / energy envelope exceeded (initial envelope is intentionally \
             tight — copy observed numbers above into DISPATCH_CEIL / ENERGY_PER_OP_CEIL_J, or \
             tighten regress_baseline::dispatch_budget / energy_budget_j):\n",
        );
        for f in &failed {
            msg.push_str("  - ");
            msg.push_str(f);
            msg.push('\n');
        }
        panic!("{msg}");
    }
}
