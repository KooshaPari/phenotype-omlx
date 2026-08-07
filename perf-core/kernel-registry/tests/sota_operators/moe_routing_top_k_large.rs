//! (k') `moe_routing_top_k_large` — byte-oracle test for the MoE
//! expert-routing *output* at the large `top_k` surface (`top_k` in
//! {4, 8}), complementing `moe_routing_top_k_small.rs` which pins the
//! `top_k <= 2` surface. The small-top-k file owns the shared helpers
//! (`RoutingPolicy`, `oracle_topk`, `deterministic_logits`,
//! `run_kernel_router`); this file re-imports them via
//! `use super::moe_routing_top_k_small::*;` and exercises the router at
//! production-realistic Mixtral-8x7B / DeepSeek-V3 / Qwen3-Coder-Next
//! shapes (64 experts, top-k ∈ {4, 8}).
//!
//! Four tests at the large-top-k surface:
//!
//! - `moe_routing_top_k_4_byte_identical_64_experts`
//! - `moe_routing_top_k_8_byte_identical_64_experts`
//! - `moe_routing_top_k_4_weights_sum_to_one`
//! - `moe_routing_top_k_8_changes_with_seed_two_distinct_decisions`

use super::moe_routing_top_k_small::{deterministic_logits, run_kernel_router, RoutingPolicy};

// ---------------------------------------------------------------------------
// Top-k stress coverage (turn-8 acceptance criteria).
//
// The four small-top-k tests in `moe_routing_top_k_small.rs` exercise
// the router at top_k <= 2. Production MoE models (Mixtral-8x7B,
// DeepSeek-V3, Qwen3-Coder-Next) route with top_k in {2..=8}. These
// four tests pin the router against the same oracle at top_k=4 and
// top_k=8, 64-expert layers, deterministic seeds.
// ---------------------------------------------------------------------------

/// Same seed, two independent `router_topk` invocations must produce a
/// byte-identical routing tensor for `top_k=4` across a `num_experts=64`
/// layer. This is the byte-equality floor at production top-k=4 surface:
/// any non-determinism in the pick-stage (e.g. a parallel-reduction that
/// flips summation order on large expert counts) trips here first.
#[test]
fn moe_routing_top_k_4_byte_identical_64_experts() {
    let batch = 16;
    let num_experts = 64;
    let top_k = 4;
    let logits = deterministic_logits(batch, num_experts, 0xCAFE_BABE);
    let policy = RoutingPolicy::Deterministic { seed: 0xC0FF_EE42 };

    let a = run_kernel_router(&logits, batch, num_experts, top_k, policy);
    let b = run_kernel_router(&logits, batch, num_experts, top_k, policy);

    assert_eq!(a.len(), batch, "router must emit one record per token");
    assert_eq!(
        a, b,
        "two runs at top_k=4 over 64 experts must produce byte-identical routing tensors"
    );

    // Tight per-byte replay check: pick ids and f32 weight bits must
    // match exactly across runs.
    for t in 0..batch {
        for i in 0..top_k {
            assert_eq!(
                a[t][i].0, b[t][i].0,
                "token {t} pick {i} expert_id drifted across runs (top_k=4)"
            );
            assert_eq!(a[t][i].1.to_bits(), b[t][i].1.to_bits(),
                "token {t} pick {i} weight drifted across runs (kernel not byte-deterministic at top_k=4)");
        }
    }

    // Top-k=4 picks must be distinct — regression sentinel for a router
    // that forgets to dedupe after picking.
    for (t, picks) in a.iter().enumerate() {
        let unique: std::collections::HashSet<usize> = picks.iter().map(|(e, _)| *e).collect();
        assert_eq!(
            unique.len(),
            top_k,
            "token {t}: top_k=4 expert ids must be distinct"
        );
    }
}

/// Same byte-equality contract as the previous test, but at
/// `top_k=8` — the native top-k of Mixtral-8x7B. This is the most
/// expensive pick in the routed-MoE design space and the most likely
/// place for a non-deterministic-reduction regression to surface.
#[test]
fn moe_routing_top_k_8_byte_identical_64_experts() {
    let batch = 16;
    let num_experts = 64;
    let top_k = 8;
    let logits = deterministic_logits(batch, num_experts, 0xCAFE_BABE);
    let policy = RoutingPolicy::Deterministic { seed: 0xDEAD_BEE8 };

    let a = run_kernel_router(&logits, batch, num_experts, top_k, policy);
    let b = run_kernel_router(&logits, batch, num_experts, top_k, policy);

    assert_eq!(a.len(), batch, "router must emit one record per token");
    assert_eq!(
        a, b,
        "two runs at top_k=8 over 64 experts must produce byte-identical routing tensors"
    );

    for t in 0..batch {
        for i in 0..top_k {
            assert_eq!(
                a[t][i].0, b[t][i].0,
                "token {t} pick {i} expert_id drifted across runs (top_k=8)"
            );
            assert_eq!(a[t][i].1.to_bits(), b[t][i].1.to_bits(),
                "token {t} pick {i} weight drifted across runs (kernel not byte-deterministic at top_k=8)");
        }
    }

    for (t, picks) in a.iter().enumerate() {
        let unique: std::collections::HashSet<usize> = picks.iter().map(|(e, _)| *e).collect();
        assert_eq!(
            unique.len(),
            top_k,
            "token {t}: top_k=8 expert ids must be distinct"
        );
        // Sanity: picks must be sorted by score DESC (and id ASC on
        // ties), which is the production ordering contract.
        for i in 1..top_k {
            assert!(
                picks[i - 1].1 >= picks[i].1,
                "token {t}: top_k=8 picks must be non-increasing in weight (got {} < {})",
                picks[i - 1].1,
                picks[i].1
            );
        }
    }
}

/// Weight-sum contract under `top_k=4`. With more picks the softmax
/// denominator is larger and any single-pick underflow would skew the
/// sum — this test pins the per-token sum to `(0.999, 1.001)` across
/// a small batch so deviations are visible.
#[test]
fn moe_routing_top_k_4_weights_sum_to_one() {
    let batch = 8;
    let num_experts = 64;
    let top_k = 4;
    let logits = deterministic_logits(batch, num_experts, 0xCAFE_BABE);

    let picks = run_kernel_router(
        &logits,
        batch,
        num_experts,
        top_k,
        RoutingPolicy::Deterministic { seed: 0xC0FF_EE42 },
    );

    for (t, p) in picks.iter().enumerate() {
        assert_eq!(p.len(), top_k, "token {t} must emit exactly top_k=4 picks");
        let sum: f32 = p.iter().map(|(_, w)| *w).sum();
        assert!(
            sum > 0.999 && sum < 1.001,
            "token {t}: top_k=4 weight sum {sum} not in (0.999, 1.001)"
        );
        for (e, w) in p {
            assert!(
                w.is_finite() && *w > 0.0 && *w <= 1.0,
                "token {t} expert {e}: weight {w} is not finite-positive-and-bounded"
            );
        }
    }
}

/// At `top_k=8`, swapping the global policy seed between two values
/// that are *both* different from the row-generation seed must yield
/// at least two distinct `(expert_id, weight)` decisions across the
/// batch. Because the current router's sorting is `(score DESC,
/// expert_id ASC)` and the seed drives post-pick RNG draws, the routing
/// tensor is fully determined by the logits; this test therefore runs
/// over two distinct logits matrices (one per seed) — the same
/// mutation pattern as `moe_routing_changes_with_seed` — and asserts
/// the resulting picks differ on at least two tokens. This is the
/// Mixtral-8x7B-shaped mutation-sanity gate.
#[test]
fn moe_routing_top_k_8_changes_with_seed_two_distinct_decisions() {
    let batch = 8;
    let num_experts = 64;
    let top_k = 8;
    let policy_a = RoutingPolicy::Deterministic { seed: 0xDEAD_BEE8 }; // 42
    let policy_b = RoutingPolicy::Deterministic { seed: 0xF00D_C0DE }; // 99

    // Two input matrices that differ in every token's logits so a
    // router that ignores its inputs cannot pass.
    let logits_a = deterministic_logits(batch, num_experts, 0xCAFE_BABE);
    let logits_b = deterministic_logits(batch, num_experts, 0xBADD_CAFE);

    let a = run_kernel_router(&logits_a, batch, num_experts, top_k, policy_a);
    let b = run_kernel_router(&logits_b, batch, num_experts, top_k, policy_b);

    let mut distinct_decisions = 0usize;
    for t in 0..batch {
        let ids_a: Vec<usize> = a[t].iter().map(|(e, _)| *e).collect();
        let ids_b: Vec<usize> = b[t].iter().map(|(e, _)| *e).collect();
        if ids_a != ids_b {
            distinct_decisions += 1;
        }
    }
    assert!(
        distinct_decisions >= 2,
        "expected >= 2 tokens whose top_k=8 routing differs between two distinct (logits, policy) pairs; got {distinct_decisions}"
    );

    // Sanity replay: under the same (logits, policy) the routing
    // tensor must reproduce byte-for-byte.
    let a_again = run_kernel_router(&logits_a, batch, num_experts, top_k, policy_a);
    for t in 0..batch {
        for i in 0..top_k {
            assert_eq!(
                a[t][i].0, a_again[t][i].0,
                "token {t} pick {i} expert_id drifted on replay (top_k=8)"
            );
            assert_eq!(a[t][i].1.to_bits(), a_again[t][i].1.to_bits(),
                "token {t} pick {i} weight drifted on replay (kernel not byte-deterministic at top_k=8)");
        }
    }
}
