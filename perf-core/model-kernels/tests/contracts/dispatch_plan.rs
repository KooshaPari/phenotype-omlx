//! Section "DispatchPlan (sanity)" of the original contracts.rs.
//!
//! Split out of the original monolithic `model-kernels/tests/contracts.rs`
//! (1130 lines) so each topic stays under the 350-line target. Test bodies
//! are byte-identical to the source file; only the surrounding module
//! wrapper and `use super::*;` import differ.

use super::*;

#[test]
fn dispatch_plan_exposes_buckets_and_dropped() {
    let assignments: Vec<(usize, f32)> = vec![(0, 0.9), (0, 0.8), (1, 0.7)];
    let plan: DispatchPlan = moe_dispatch(
        &[0, 1, 2],
        &assignments,
        2,
        1.0, // capacity = ceil(1.0 * 3 / 2) = 2 per expert
    )
    .unwrap();
    assert_eq!(plan.expert_buckets.len(), 2);
    assert_eq!(plan.capacity_used.len(), 2);
    assert!(plan.dropped.is_empty());
}
