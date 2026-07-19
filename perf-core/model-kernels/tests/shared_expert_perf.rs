//! Regression test for the [`shared_expert`] scalar matmul inner loop.
//!
//! Pins the perf-invariant declared at the top of
//! `model-kernels/src/moe/shared.rs`: the helper must finish a
//! `512×512×4096` dense matmul (≈1.07 GFLOP) on a single thread in well
//! under 5 seconds on Apple Silicon in debug mode. The same invariant is
//! what keeps `regress-baseline/tests/dispatch_buckets.rs` (which calls
//! `shared_expert` on a 64-wide inner tile for six shape buckets) under
//! 60 seconds end-to-end.
//!
//! The test deliberately runs in **debug** mode (no `--release`): the
//! regression-bucket test that motivated the cap is itself a debug-mode
//! `cargo test`, so a release-only regression test would let a slow
//! debug-mode inner loop slip past CI.
//!
//! Run with:
//!
//! ```text
//! cargo test -p model-kernels --test shared_expert_perf -- --nocapture
//! ```
//!
//! The ceiling is intentionally generous (5 s) to absorb variance across
//! CI machines and Apple-silicon SKUs; the real ceiling on this machine
//! (M-series) is around 1 s in debug. If the test starts failing on a
//! CI runner that is materially slower than dev hardware, raise the
//! ceiling **and** open a follow-up — do not just bump the number.

use std::hint::black_box;
use std::time::Instant;

use model_kernels::moe_facade::shared_expert;

/// Wall-clock ceiling (seconds) for a single 512×512×4096 invocation
/// of [`shared_expert`] in debug mode on Apple Silicon.
const CEIL_SECS: f64 = 5.0;

#[test]
fn shared_expert_512x512x4096_finishes_under_5s_in_debug() {
    let m: usize = 512;
    let n: usize = 512;
    let k: usize = 4096;

    // Deterministic, non-zero contents so the optimizer cannot fold
    // the matmul away.
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let w: Vec<f32> = (0..k * n)
        .map(|i| ((i % 97) as f32) * 0.25 + 0.5)
        .collect();
    let mut out: Vec<f32> = vec![0.0; m * n];

    // Warmup so the first-call cache / page-fault cost is paid before
    // the timed call.
    shared_expert(&x, &w, &mut out).expect("shared_expert must accept well-formed buffers");

    let start = Instant::now();
    shared_expert(&x, &w, &mut out).expect("shared_expert must accept well-formed buffers");
    let elapsed = start.elapsed();

    // Sanity: the result is non-trivial (not all zeros, not folded away).
    let acc: f32 = out.iter().copied().sum();
    assert!(
        acc.is_finite() && acc.abs() > 1.0,
        "shared_expert produced a degenerate result (acc={acc}); the optimizer may have folded the matmul away"
    );

    // Black-box the buffers so the optimizer cannot elide the call.
    black_box(&x);
    black_box(&w);
    black_box(&out);

    let elapsed_secs = elapsed.as_secs_f64();
    eprintln!(
        "[shared_expert_perf] m={m} n={n} k={k} elapsed={elapsed_secs:.3}s (ceil={CEIL_SECS:.1}s)"
    );

    assert!(
        elapsed_secs <= CEIL_SECS,
        "shared_expert took {elapsed_secs:.3}s for {m}x{n}x{k}; \
         must finish within {CEIL_SECS:.1}s. The divisor-loop O(total) \
         regression has returned — see the Perf invariants note at the \
         top of model-kernels/src/moe/shared.rs."
    );
}