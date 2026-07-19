//! The actual test. One assertion per bucket per metric.
//!
//! Ceilings come from [`regress_baseline::dispatch_budget`] and
//! [`regress_baseline::energy_budget_j`] so production callers share a
//! single source of truth with this test. See the module docs in
//! `main.rs` for the bucket table and the headroom rationale.

use super::*;

#[test]
fn dispatch_and_energy_within_per_bucket_envelope() {
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
