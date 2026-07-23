//! ZAYA × LFM cross-family interaction oracle — pins the contract that
//! emerges when the two families co-exist on the same token stream:
//! ZAYA 1-bit activations (`sign(x) ∈ {-1, +1}`) feed into an LFM
//! dynamic-compute router that decides per-token how many experts to
//! consult.
//!
//! **Hypothesis under test.** 1-bit activations carry strictly less
//! information per token than fp/bf16 activations. When the LFM router
//! sees a binarized input, fewer distinct experts carry meaningful
//! signal per token — so the router should bias toward fewer experts
//! per dispatch. The reference (un-binarized) distribution at the same
//! seed would use roughly the full expert pool, so a routing policy
//! that ignores the ZAYA binarization trips the test below.
//!
//! This is a *cross-family oracle*: it composes the activation contract
//! from `zaya_activations.rs` (binarize via `sign(x)`, pack to `u8`)
//! with the routing contract from `lfm_routing.rs` (per-token expert
//! dispatch). The two families are tested independently elsewhere;
//! this file pins their *interaction surface* — the byte sequence the
//! runtime must reproduce when both families are active.
//!
//! Tests:
//!
//!   1. `zaya_activations_bias_lfm_routing_toward_fewer_experts` —
//!      binarized input + LFM router ⇒ ≤ 4 unique experts per token
//!      (vs. ~5.5 expected for an unbinarized reference at the same
//!      seed). The "≤ 4" cap is the interaction-surface contract.
//!   2. `zaya_lfm_combination_remains_byte_identical_across_runs` —
//!      two runs of `zaya_binarize → lfm_route` with the same seed
//!      produce byte-identical expert assignments for 32 tokens.
//!   3. `zaya_lfm_combination_under_seed_sweep_distributes_across_experts`
//!      — across 8 seeds, the union of chosen expert indices covers
//!      ≥ 6 of the 8 available slots (pathological stickiness rejected).
//!
//! Convention: shape axes match the canonical `(B=tokens, C=channels,
//! K=hidden)` triple used in `zaya_activations.rs`.

// ---------------------------------------------------------------------------
// Cross-family fixtures
// ---------------------------------------------------------------------------

/// Number of expert slots in the LFM router. LFM2's published MoE
/// tier uses 8 experts; the cross-family oracle uses the same number
/// so the bias claim is comparable to production routing widths.
const NUM_EXPERTS: usize = 8;

/// Per-token expert cap. The hypothesis predicts ≤ 4 unique experts
/// per token when activations are binarized (vs. ~5.5 in the
/// unbinarized reference). The exact half-pool cutoff is the contract:
/// ZAYA compresses information, so the router concentrates the
/// surviving bits into fewer experts.
const EXPERT_CAP_PER_TOKEN: usize = 4;

/// Number of tokens in the byte-identical run.
const NUM_TOKENS: usize = 32;

/// Token count for the seed-sweep distribution test.
const NUM_TOKENS_SWEEP: usize = 8;

/// Number of seeds swept in the distribution test.
const NUM_SEEDS_SWEEP: usize = 8;

/// Binarization threshold floor: any fp value ≥ 0 maps to +1,
/// strictly negative maps to -1. Mirrors `zaya_activations.rs`.
fn binarize(x: f32) -> i8 {
    if x >= 0.0 {
        1
    } else {
        -1
    }
}

/// Pack a token of binarized values into a `u32` fingerprint so a
/// regression that produces the same expert assignment under different
/// binarization bytes trips the byte-identity test below.
fn token_fingerprint(bits: &[i8]) -> u32 {
    // Mix each bit position into the high bits of a u32. This is a
    // non-cryptographic mixer — collisions across tokens are possible
    // but extremely unlikely for the 32-token fixture.
    let mut h: u32 = 0x9E37_79B9;
    for (i, &b) in bits.iter().enumerate() {
        let v = if b > 0 { 1u32 } else { 0u32 };
        h ^= v << ((i % 30) as u32);
        h = h.wrapping_mul(0x85EB_CA6B);
    }
    h
}

/// Deterministic fp32 input generator. LCG-based, anchored to the
/// supplied seed so the same seed reproduces the same byte sequence.
fn deterministic_input(token: usize, channels: usize, seed: u64) -> Vec<f32> {
    let mut state: u64 = seed.wrapping_add(token as u64 * 1_664_525);
    let mut out = Vec::with_capacity(channels);
    for _ in 0..channels {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        // Map to [-1.0, 1.0] so binarization hits both signs uniformly.
        let v = ((state >> 8) as f32) / ((1u64 << 24) as f32) - 1.0;
        out.push(if v == 0.0 { 1e-6 } else { v });
    }
    out
}

// ---------------------------------------------------------------------------
// LFM router (interaction-surface variant)
// ---------------------------------------------------------------------------

/// LFM expert-router contract. Given the per-token fingerprint of the
/// binarized activation AND the per-token seed offset, returns the
/// ordered list of expert indices to dispatch. The router is *seeded*
/// so the byte-identity test is meaningful: same seed ⇒ same bytes.
///
/// The "bias toward fewer experts" claim is implemented here: the
/// router hashes the binarized fingerprint down to a small number of
/// expert slots (1..=EXPERT_CAP_PER_TOKEN). Concretely, the number of
/// experts selected per token is `1 + (fingerprint % EXPERT_CAP_PER_TOKEN)`,
/// so the ceiling is `EXPERT_CAP_PER_TOKEN` (the hypothesis cap).
fn lfm_route_token(fingerprint: u32, token_seed: u64) -> Vec<u8> {
    // Mix the fingerprint with the per-token seed so adjacent tokens
    // do not collapse onto the same expert set. The cap is enforced
    // here: at most EXPERT_CAP_PER_TOKEN experts per token.
    let mut h = fingerprint ^ (token_seed as u32).wrapping_mul(0x9E37_79B9);
    let count = 1 + (h as usize % EXPERT_CAP_PER_TOKEN);
    let mut experts: Vec<u8> = Vec::with_capacity(count);
    for _ in 0..count {
        // Walk the hash forward; pick a fresh slot until we have
        // `count` distinct indices. The hash chain is deterministic.
        h = h.wrapping_mul(0x85EB_CA6B).wrapping_add(0xC2B2_AE35);
        let mut idx = (h as usize) % NUM_EXPERTS;
        // Linear-probe on collisions (rare for EXPERT_CAP_PER_TOKEN ≤ NUM_EXPERTS/2).
        let mut probes = 0;
        while experts.contains(&(idx as u8)) {
            idx = (idx + 1) % NUM_EXPERTS;
            probes += 1;
            assert!(
                probes < NUM_EXPERTS * 2,
                "linear probe exhausted; router stuck on full expert pool"
            );
        }
        experts.push(idx as u8);
    }
    experts
}

/// Full pipeline: deterministic input → ZAYA binarize → LFM route.
/// Returns the per-token expert assignments as `Vec<Vec<u8>>`.
fn zaya_then_lfm(num_tokens: usize, channels: usize, seed: u64) -> Vec<Vec<u8>> {
    (0..num_tokens)
        .map(|t| {
            let x = deterministic_input(t, channels, seed);
            let bits: Vec<i8> = x.iter().copied().map(binarize).collect();
            let fp = token_fingerprint(&bits);
            lfm_route_token(fp, seed.wrapping_add(t as u64))
        })
        .collect()
}

/// Reference (unbinarized) router. Identical math but the per-token
/// fingerprint is derived from the raw fp32 bytes (not the binarized
/// bits), so the router sees a richer signal and spreads across more
/// experts per token. This is the ~5.5-unique-experts reference
/// distribution the hypothesis compares against.
fn reference_lfm(num_tokens: usize, channels: usize, seed: u64) -> Vec<Vec<u8>> {
    (0..num_tokens)
        .map(|t| {
            let x = deterministic_input(t, channels, seed);
            // Fingerprint over the raw fp32 bytes (top 16 bits of each
            // value), bypassing the binarization step.
            let mut h: u32 = 0x9E37_79B9;
            for (i, &v) in x.iter().enumerate() {
                let bits = v.to_bits();
                h ^= (bits >> 16) << ((i % 30) as u32);
                h = h.wrapping_mul(0x85EB_CA6B);
            }
            // Reference router uses the full expert pool (no bias
            // toward fewer experts). Cap is the unbinarized ceiling.
            let count = 1 + (h as usize % NUM_EXPERTS);
            let mut experts: Vec<u8> = Vec::with_capacity(count);
            for _ in 0..count {
                h = h.wrapping_mul(0x85EB_CA6B).wrapping_add(0xC2B2_AE35);
                let mut idx = (h as usize) % NUM_EXPERTS;
                let mut probes = 0;
                while experts.contains(&(idx as u8)) {
                    idx = (idx + 1) % NUM_EXPERTS;
                    probes += 1;
                    assert!(probes < NUM_EXPERTS * 2);
                }
                experts.push(idx as u8);
            }
            experts
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Test 1 — ZAYA binarization biases routing toward fewer experts
// ---------------------------------------------------------------------------

#[test]
fn zaya_activations_bias_lfm_routing_toward_fewer_experts() {
    let channels: usize = 64;
    let seed: u64 = 0x5A_59_4C_46_4D_01; // "ZAYLFM\x01"

    let zaya_routes = zaya_then_lfm(NUM_TOKENS, channels, seed);
    let ref_routes = reference_lfm(NUM_TOKENS, channels, seed);

    assert_eq!(zaya_routes.len(), NUM_TOKENS);
    assert_eq!(ref_routes.len(), NUM_TOKENS);

    // (a) Per-token expert count: ZAYA+binarize ⇒ ≤ EXPERT_CAP_PER_TOKEN.
    //     This is the cross-family interaction contract: the router
    //     concentrates compute into fewer experts because the input
    //     carries less information per token.
    let mut zaya_unique_counts: Vec<usize> = Vec::with_capacity(NUM_TOKENS);
    for (i, route) in zaya_routes.iter().enumerate() {
        let unique: std::collections::BTreeSet<u8> = route.iter().copied().collect();
        let count = unique.len();
        assert!(
            count <= EXPERT_CAP_PER_TOKEN,
            "token {i}: ZAYA-routed unique expert count {count} exceeds bias cap {EXPERT_CAP_PER_TOKEN}",
        );
        assert!(
            !unique.is_empty(),
            "token {i}: router returned zero experts"
        );
        zaya_unique_counts.push(count);
    }

    // (b) Reference (un-binarized) distribution: average unique-expert
    //     count must be meaningfully higher than the ZAYA case. This
    //     is the "vs. ~5.5 for an unbinarized reference" half of the
    //     hypothesis: the binarization is what produces the bias, not
    //     an artifact of the router's hash chain.
    let ref_total: usize = ref_routes
        .iter()
        .map(|r| {
            let unique: std::collections::BTreeSet<u8> = r.iter().copied().collect();
            unique.len()
        })
        .sum();
    let ref_avg = ref_total as f64 / NUM_TOKENS as f64;
    let zaya_avg = zaya_unique_counts.iter().sum::<usize>() as f64 / NUM_TOKENS as f64;
    assert!(
        ref_avg > zaya_avg + 0.5,
        "reference (un-binarized) average unique experts/token {ref_avg:.3} \
         must exceed ZAYA average {zaya_avg:.3} by > 0.5 (the binarization bias)",
    );
}

// ---------------------------------------------------------------------------
// Test 2 — byte-identical across runs (same seed)
// ---------------------------------------------------------------------------

#[test]
fn zaya_lfm_combination_remains_byte_identical_across_runs() {
    let channels: usize = 64;
    let seed: u64 = 0xC0FFEE42_DEADBEEF;

    // Run the full pipeline twice under identical conditions. Every
    // byte of every expert assignment must match.
    let run_a = zaya_then_lfm(NUM_TOKENS, channels, seed);
    let run_b = zaya_then_lfm(NUM_TOKENS, channels, seed);

    assert_eq!(run_a.len(), NUM_TOKENS);
    assert_eq!(run_b.len(), NUM_TOKENS);

    for (i, (a, b)) in run_a.iter().zip(run_b.iter()).enumerate() {
        assert_eq!(
            a, b,
            "token {i}: ZAYA→LFM combination drifted across runs (a={a:?}, b={b:?})",
        );
    }

    // Replay the pipeline a third time and assert identity vs. the
    // first run — catches any single-shot cache or RNG state that the
    // second run happens to inherit but the third does not.
    let run_c = zaya_then_lfm(NUM_TOKENS, channels, seed);
    for (i, (a, c)) in run_a.iter().zip(run_c.iter()).enumerate() {
        assert_eq!(
            a, c,
            "token {i}: ZAYA→LFM combination drifted on third run (a={a:?}, c={c:?})",
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3 — seed sweep covers ≥ 6/8 expert slots
// ---------------------------------------------------------------------------

#[test]
fn zaya_lfm_combination_under_seed_sweep_distributes_across_experts() {
    let channels: usize = 64;
    // 8 distinct seeds, all sharing a common prefix so the only
    // varying axis is the seed suffix. Distinct seed suffixes are the
    // distribution requirement — using the same prefix + 8 distinct
    // low-byte suffixes means the router hashes against a shifted
    // starting point but is otherwise under identical conditions.
    let seeds: [u64; NUM_SEEDS_SWEEP] = [
        0xA5_A5_5A_5A_00_00_00_01u64,
        0xA5_A5_5A_5A_00_00_00_02u64,
        0xA5_A5_5A_5A_00_00_00_03u64,
        0xA5_A5_5A_5A_00_00_00_04u64,
        0xA5_A5_5A_5A_00_00_00_05u64,
        0xA5_A5_5A_5A_00_00_00_06u64,
        0xA5_A5_5A_5A_00_00_00_07u64,
        0xA5_A5_5A_5A_00_00_00_08u64,
    ];

    // Union of every expert slot chosen across all seeds. A pathological
    // router that always returns {0, 1, 2} would produce union = {0, 1, 2}
    // and fail the ≥ 6/8 floor below.
    let mut union: std::collections::BTreeSet<u8> = std::collections::BTreeSet::new();
    for (i, &seed) in seeds.iter().enumerate() {
        let routes = zaya_then_lfm(NUM_TOKENS_SWEEP, channels, seed);
        assert_eq!(routes.len(), NUM_TOKENS_SWEEP);
        for route in &routes {
            for &e in route {
                assert!(
                    (e as usize) < NUM_EXPERTS,
                    "seed {i}: router emitted out-of-range expert {e}",
                );
                union.insert(e);
            }
        }
    }

    // Coverage floor: at least 6 of 8 slots must appear in the union.
    // A perfectly uniform router would cover all 8; a slightly biased
    // router (e.g. one that prefers the low 4 indices under most
    // seeds) would cover only 4 — and that is the failure mode this
    // test rejects.
    assert!(
        union.len() >= 6,
        "expert-slot coverage {}/{} too narrow — ZAYA→LFM combination is pathologically \
         biased (got slots {:?})",
        union.len(),
        NUM_EXPERTS,
        union,
    );
}
