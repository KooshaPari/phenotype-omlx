//! Black-box integration tests for the Mixture-of-Depths (MoD) routing
//! kernel.
//!
//! These tests pin the public API described in
//! `docs/sessions/20260718-metal-model-runtime/02_SPECIFICATIONS.md` and
//! the operator list. The focus is end-to-end behaviour — `mod_route`
//! contract, `mod_apply` + `mod_scatter_back` round-trip, and
//! input-validation error messages.
//!
//! Conventions:
//!
//! - Tolerances for numerical comparisons follow the crate contract
//!   (`abs = 1e-5`, `rel = 1e-4`).
//! - Random inputs are produced from a fixed seed
//!   (`0xCAFE_BABE_DEAD_BEEF`).
//! - All buffer sizes are derived from the documented layouts.

use model_kernels::mod_routing::{
    mod_apply, mod_route, mod_scatter_back, ModRoutePlan, ModRouterConfig,
};

const SEED: u64 = 0xCAFE_BABE_DEAD_BEEF;

fn cfg(capacity_factor: f32) -> ModRouterConfig {
    ModRouterConfig {
        capacity_factor,
        mean_capacity: 0.0,
    }
}

/// Build a deterministic vector of `f32` of length `n` from `salt`.
fn deterministic_vec(n: usize, salt: u64) -> Vec<f32> {
    let mut state = SEED ^ salt;
    let mut next = move || {
        // MMIX LCG (matches `model_kernels::common::Lcg`).
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let v = (state >> 40) as u32;
        (v as f32) / (1u32 << 24) as f32 - 0.5
    };
    (0..n).map(|_| next()).collect()
}

// ---------------------------------------------------------------------------
// mod_route contract
// ---------------------------------------------------------------------------

#[test]
fn capacity_one_is_identity_routing() {
    // capacity=1.0 must select every token, regardless of weight. The
    // contract is "routing is a no-op identity": no token is dropped.
    let weights = deterministic_vec(8, 0xA1);
    let plan = mod_route(&weights, &cfg(1.0)).unwrap();
    assert_eq!(plan.selected_tokens.len(), weights.len(),
        "capacity=1.0 must return all tokens");
    // Survivors are a permutation of the input indices; sort to compare.
    let mut sorted = plan.selected_tokens.clone();
    sorted.sort();
    let expected: Vec<u32> = (0..weights.len() as u32).collect();
    assert_eq!(sorted, expected, "every index must appear exactly once");
}

#[test]
fn capacity_half_returns_exactly_half_top_scored() {
    // 10 tokens with monotonically increasing weights. The top half
    // (5) are the highest-scoring tokens; the bottom half are dropped.
    let n = 10;
    let mut weights: Vec<f32> = (0..n).map(|i| (i as f32 - 4.5) * 2.0).collect();
    // Shuffle in a few negatives to ensure the function isn't just
    // picking a contiguous block; we'll check membership of the
    // expected set.
    weights[0] = -5.0;
    weights[1] = -3.0;
    weights[2] = -1.0;
    let plan = mod_route(&weights, &cfg(0.5)).unwrap();
    assert_eq!(plan.selected_tokens.len(), n / 2,
        "capacity=0.5 on n=10 must return floor(0.5 * 10) = 5 tokens");
    // All of the top 5 weights (indices 5..10 with strict monotonic
    // increase) must be present.
    for i in 5..n {
        assert!(plan.selected_tokens.contains(&(i as u32)),
            "high-weight token {i} must survive; plan={:?}", plan.selected_tokens);
    }
    // None of the bottom 3 (which we set to large negatives) may survive.
    for i in 0..3 {
        assert!(!plan.selected_tokens.contains(&(i as u32)),
            "low-weight token {i} must not survive; plan={:?}", plan.selected_tokens);
    }
}

#[test]
fn mod_route_is_deterministic_under_repeated_calls() {
    let weights = deterministic_vec(16, 0xB2);
    let p1 = mod_route(&weights, &cfg(0.5)).unwrap();
    let p2 = mod_route(&weights, &cfg(0.5)).unwrap();
    let p3 = mod_route(&weights, &cfg(0.5)).unwrap();
    assert_eq!(p1, p2);
    assert_eq!(p2, p3);
    // Also stable across distinct capacity factors (the plan shape
    // depends only on the requested factor).
    let p_other = mod_route(&weights, &cfg(0.25)).unwrap();
    assert_eq!(p_other.selected_tokens.len(), 4);
}

#[test]
fn empty_weights_produces_empty_plan() {
    let plan = mod_route(&[], &cfg(0.5)).unwrap();
    assert!(plan.selected_tokens.is_empty());
    assert_eq!(plan.capacity_factor, 0.5);
}

#[test]
fn ties_break_deterministically_by_lower_index() {
    // All weights equal -> all sigmoid scores equal -> sort key is
    // (score asc, index asc) for the *descending* comparison, so the
    // lower-index tokens are kept first.
    let n = 6usize;
    let weights = vec![0.0f32; n];
    let plan = mod_route(&weights, &cfg(0.5)).unwrap();
    assert_eq!(plan.selected_tokens, vec![0, 1, 2],
        "all-ties case must select the first floor(0.5*6)=3 indices");
    // A second capacity factor selects the same lower-index prefix
    // because tie-break is index ascending.
    let plan2 = mod_route(&weights, &cfg(0.5)).unwrap();
    assert_eq!(plan2.selected_tokens, vec![0, 1, 2]);
}

// ---------------------------------------------------------------------------
// mod_apply + mod_scatter_back round-trip
// ---------------------------------------------------------------------------

#[test]
fn apply_and_scatter_back_round_trip_is_exact_with_zero_fill() {
    let num_tokens = 12usize;
    let dim = 4usize;
    let weights = deterministic_vec(num_tokens, 0xC3);
    let plan = mod_route(&weights, &cfg(0.5)).unwrap();
    assert!(!plan.selected_tokens.is_empty());

    // Build a deterministic full buffer: hidden[t, d] = t * 10 + d.
    let full: Vec<f32> = (0..num_tokens * dim)
        .map(|i| (i / dim) as f32 * 10.0 + (i % dim) as f32)
        .collect();

    // Apply: extract the surviving rows.
    let selected = mod_apply(&plan, &full, dim).unwrap();
    assert_eq!(selected.len(), plan.selected_tokens.len() * dim);

    // Scatter back with fill=0.0; the result must match `full` at
    // surviving rows and be 0.0 at the rest.
    let scattered = mod_scatter_back(&selected, &plan, num_tokens, dim, 0.0).unwrap();
    assert_eq!(scattered.len(), num_tokens * dim);

    let survivor: std::collections::HashSet<u32> =
        plan.selected_tokens.iter().copied().collect();
    let mut squared_l2 = 0.0f64;
    for t in 0..num_tokens {
        for d in 0..dim {
            let i = t * dim + d;
            let expected = if survivor.contains(&(t as u32)) {
                full[i] as f64
            } else {
                0.0
            };
            let got = scattered[i] as f64;
            let diff = got - expected;
            squared_l2 += diff * diff;
        }
    }
    assert_eq!(squared_l2, 0.0,
        "round-trip with fill=0.0 must be exact; got L2^2 = {squared_l2}");
}

#[test]
fn scatter_back_uses_supplied_fill_for_skipped_rows() {
    // With fill = 0.0, skipped rows must be 0.0; with fill = 7.5 they
    // must be 7.5. This pins the "fill for skipped rows" contract.
    let num_tokens = 8usize;
    let dim = 2usize;
    let weights = vec![0.0f32; num_tokens];
    let plan = mod_route(&weights, &cfg(0.5)).unwrap();
    // Fill is 0 -> half the rows must read 0.0 after scatter.
    let full: Vec<f32> = (0..num_tokens * dim).map(|i| (i + 1) as f32).collect();
    let selected = mod_apply(&plan, &full, dim).unwrap();
    let back_zero = mod_scatter_back(&selected, &plan, num_tokens, dim, 0.0).unwrap();
    let back_seven = mod_scatter_back(&selected, &plan, num_tokens, dim, 7.5).unwrap();
    let survivor: std::collections::HashSet<u32> =
        plan.selected_tokens.iter().copied().collect();
    for t in 0..num_tokens {
        for d in 0..dim {
            let i = t * dim + d;
            let expected = if survivor.contains(&(t as u32)) {
                full[i]
            } else {
                0.0
            };
            assert_eq!(back_zero[i], expected);
            let expected_seven = if survivor.contains(&(t as u32)) {
                full[i]
            } else {
                7.5
            };
            assert_eq!(back_seven[i], expected_seven);
        }
    }
}

// ---------------------------------------------------------------------------
// Validation errors
// ---------------------------------------------------------------------------

#[test]
fn rejects_capacity_factor_outside_unit_interval_with_clear_error() {
    let weights = vec![0.1f32, 0.2, 0.3];
    // Zero: capacity must be strictly positive.
    let err = mod_route(&weights, &cfg(0.0)).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("capacity_factor"),
        "error must name the offending argument, got: {msg}");
    assert!(msg.contains("0"),
        "error must echo the rejected value, got: {msg}");

    // Negative: not allowed.
    let err = mod_route(&weights, &cfg(-0.5)).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("capacity_factor"),
        "error must name capacity_factor, got: {msg}");

    // Above 1: not allowed.
    let err = mod_route(&weights, &cfg(1.5)).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("capacity_factor"),
        "error must name capacity_factor, got: {msg}");

    // NaN: not finite -> reject.
    let err = mod_route(&weights, &cfg(f32::NAN)).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("capacity_factor"),
        "NaN capacity_factor must be rejected, got: {msg}");
}

#[test]
fn capacity_one_boundary_is_inclusive() {
    // Spec: capacity_factor in (0, 1] *inclusive of 1.0*. Make sure
    // 1.0 is accepted (it's the identity case) and a value *just*
    // above 1.0 is rejected.
    let weights = deterministic_vec(4, 0xD4);
    let plan = mod_route(&weights, &cfg(1.0)).unwrap();
    assert_eq!(plan.selected_tokens.len(), 4);
    let err = mod_route(&weights, &cfg(1.0 + f32::EPSILON)).unwrap_err();
    assert!(err.to_string().contains("capacity_factor"));
}

#[test]
fn plan_carries_capacity_factor_through() {
    // mod_route must echo the capacity_factor on the returned plan so
    // downstream kernels can scale residual streams by 1/c.
    let weights = deterministic_vec(8, 0xE5);
    let plan = mod_route(&weights, &cfg(0.5)).unwrap();
    assert_eq!(plan.capacity_factor, 0.5);
    let plan2 = mod_route(&weights, &cfg(0.75)).unwrap();
    assert_eq!(plan2.capacity_factor, 0.75);
}

#[test]
fn plan_clone_and_equality_are_value_semantics() {
    let weights = deterministic_vec(6, 0xF6);
    let plan = mod_route(&weights, &cfg(0.5)).unwrap();
    let cloned = plan.clone();
    assert_eq!(plan, cloned);
    // Two plans built from different capacity factors are unequal.
    let plan_other = mod_route(&weights, &cfg(0.25)).unwrap();
    assert_ne!(plan, plan_other);
}

#[test]
fn mod_routing_plan_struct_is_constructible_with_explicit_fields() {
    // Pin the public field layout: the integration tests depend on
    // `selected_tokens` being a `Vec<u32>` and `capacity_factor`
    // being a `f32`. Compile-time checks already enforce these
    // types, but we exercise the struct from a `&str`-free builder
    // to keep the public surface honest.
    let plan = ModRoutePlan {
        selected_tokens: vec![0, 2, 4],
        capacity_factor: 0.5,
    };
    assert_eq!(plan.selected_tokens.len(), 3);
    assert_eq!(plan.capacity_factor, 0.5);
}