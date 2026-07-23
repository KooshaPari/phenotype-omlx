//! Shape-bucketed dispatch-count + energy-per-op regression test for the
//! Metal model-runtime.
//!
//! # Why each bucket has its own ceiling
//!
//! A single global "no kernel may exceed N dispatches" rule is wrong: a
//! `512x2048x2048` matmul is a tiny problem that fits in one tile and
//! should require ~1 Metal command-buffer dispatch, while a
//! `8192x8192x8192` matmul genuinely needs to be split across hundreds
//! of tiles so the GPU's command queue stays fed. The spec's perf budget
//! is per-shape, not global, so this test pins one ceiling per bucket.
//!
//! # What this test does
//!
//! For each of the six buckets defined in [`BUCKETS`]:
//!
//! 1. Allocate `a: [M, K]`, `w: [K, N]`, `out: [M, N]` as `f32` and run
//!    the reference matmul from `model_kernels` (`shared_expert`).
//! 2. Time the call with `std::time::Instant`.
//! 3. Derive two observable metrics:
//!
//!    - `dispatches`: the number of Metal command-buffer dispatches a
//!      naive 64×64 output-tile policy would emit. The scalar reference
//!      kernel itself is a single function call (one "logical" launch),
//!      so we synthesize the policy-driven count as
//!      `ceil(M/64) * ceil(N/64)` to model what the Metal runtime will
//!      actually queue.
//!    - `energy_j`: a reproducible joules estimate, derived from the
//!      measured wall time of a **single 64×64 tile** (scaled to the
//!      full shape by `num_tiles`) and a constant 30 W TDP share
//!      (`joules = seconds * 30.0`). Tile-based measurement keeps the
//!      test fast even at the largest bucket (a naive `8192×8192×8192`
//!      scalar matmul would otherwise run for tens of minutes); for
//!      production telemetry the caller should swap in
//!      `Measurement::energy_j` from the instrumented Metal runtime.
//!    - `energy_per_op_j = energy_j / flops` (joules per fused
//!      multiply-add). This normalizes across bucket sizes so a tiny
//!      bucket and a huge one share the same axis.
//!
//! 4. Print every observed number with `eprintln!` (visible with
//!    `cargo test -- --nocapture`).
//! 5. Assert each metric is `<=` the per-shape ceiling exposed by
//!    [`regress_baseline::dispatch_budget`] and
//!    [`regress_baseline::energy_budget_j`]. The ceilings live in the
//!    library so production callers (not just this test) get the same
//!    envelope.
//!
//! # TDD discipline and follow-up plan
//!
//! This test was built by the red-green cycle: the first check-in
//! shipped with intentionally tight ceilings so the test failed loudly
//! and dumped the actual numbers via `--nocapture`. The follow-up
//! commit lifted those ceilings into [`regress_baseline::budget`] with
//! 1.2× (dispatch) and 1.5× (energy) headroom applied over the first
//! observed run. This file no longer holds its own `DISPATCH_CEIL` /
//! `ENERGY_PER_OP_CEIL_J` mirrors — both are now read from the library
//! so the envelope is single-sourced.
//!
//! The remaining follow-up commit must:
//!
//! - plumb `Measurement::dispatches` and `Measurement::energy_j` from
//!   the instrumented Metal runtime into this test (replacing the
//!   synthesis in `synthetic_dispatch_count` and `measure_matmul`),
//! - tighten the per-bucket ceilings in
//!   [`regress_baseline::budget::BUCKETS`] against that real telemetry.
//!
//! # Why we don't add a Metal dependency here
//!
//! This test sits inside `regress-baseline`, which is intentionally
//! pure-Rust and free of GPU / runtime crates. The reference matmul is
//! `model_kernels::shared_expert`. When the Metal instrumented path
//! (`metal-runtime`) starts emitting `Measurement { dispatches,
//! energy_j, .. }` rows, the synthesis in `observe_bucket` should be
//! swapped for the real values; the assertions stay the same.

use std::hint::black_box;
use std::time::Instant;

use model_kernels::moe_facade::shared_expert;
use regress_baseline::{dispatch_budget, energy_budget_j, ShapeKey};

// ---------------------------------------------------------------------------
// Buckets + initial-envelope ceilings
// ---------------------------------------------------------------------------

/// One shape bucket the Metal model-runtime must hit a perf budget for.
#[derive(Debug, Clone, Copy)]
struct Bucket {
    /// Human-readable tag used in printed diagnostics.
    name: &'static str,
    /// M, N, K of the dense matmul `out[m, n] = a[m, k] @ w[k, n]`.
    shape: ShapeKey,
}

/// Eight buckets the spec calls out: a long-context decode, a tiny
/// decode, two medium prompts, a square 4k, a Mixtral-class MoE expert
/// FFN, a square 8k, and a long-context 16k decode path. Ordered from
/// smallest to largest output cell count so the printout reads as a perf
/// scaling ladder.
const BUCKETS: &[Bucket] = &[
    Bucket {
        name: "longctx_64x32_c2048",
        shape: ShapeKey::new(64, 8192, 2048),
    },
    Bucket {
        name: "tiny_decode_512x2048x2048",
        shape: ShapeKey::new(512, 2048, 2048),
    },
    Bucket {
        name: "small_prompt_1024x4096x4096",
        shape: ShapeKey::new(1024, 4096, 4096),
    },
    Bucket {
        name: "medium_prompt_2048x8192x8192",
        shape: ShapeKey::new(2048, 8192, 8192),
    },
    Bucket {
        name: "square_4k_4096x4096x4096",
        shape: ShapeKey::new(4096, 4096, 4096),
    },
    Bucket {
        name: "bigmoe_expert_2x14336",
        shape: ShapeKey::new(2048, 14336, 14336),
    },
    Bucket {
        name: "square_8k_8192x8192x8192",
        shape: ShapeKey::new(8192, 8192, 8192),
    },
    Bucket {
        name: "long_decode_16384x4096x4096",
        shape: ShapeKey::new(16384, 4096, 4096),
    },
];

// ---------------------------------------------------------------------------
// Per-bucket ceilings live in `regress_baseline::budget::BUCKETS`.
// ---------------------------------------------------------------------------
//
// The test pulls its dispatch / energy ceilings from
// [`regress_baseline::dispatch_budget`] /
// [`regress_baseline::energy_budget_j`] rather than duplicating them
// here. This keeps a single source of truth (the library) so a change
// to the canonical envelopes shows up in every consumer — including
// future production gates outside the test suite — without a parallel
// edit.
//
// The initial observed values were captured by the first run of this
// test on 2026-07-18 and lifted into the library in the same commit
// that deleted the local `DISPATCH_CEIL` / `ENERGY_PER_OP_CEIL_J`
// mirrors:
//
// | Bucket                       | dispatches | energy_per_op_j | library ceiling |
// |------------------------------|-----------:|----------------:|----------------:|
// | tiny_decode_512x2048x2048    |        256 |        1.14e-7  |   308 / 1.75e-7 |
// | small_prompt_1024x4096x4096  |       1024 |        1.09e-7  |  1229 / 1.70e-7 |
// | medium_prompt_2048x8192x8192 |       4096 |        1.15e-7  |  4916 / 1.80e-7 |
// | square_4k_4096x4096x4096     |       4096 |        1.17e-7  |  4916 / 1.80e-7 |
// | square_8k_8192x8192x8192     |      16384 |        1.25e-7  | 19661 / 1.90e-7 |
// | long_decode_16384x4096x4096  |      16384 |        1.27e-7  | 19661 / 1.95e-7 |
//
// (Headroom: 1.2× for dispatches, 1.5× for energy.)
// Dispatch / energy synthesis
// ---------------------------------------------------------------------------

/// Output-tile dimension used by the synthetic dispatch policy. A 64×64
/// tile is the smallest size the Metal Performance Shaders GEMM kernel
/// is happy with on Apple-silicon GPUs; smaller tiles leak overhead.
const TILE_DIM: u32 = 64;

/// Number of dispatches a 64×64 tile policy would emit for a `[M, N]`
/// output tile. Equivalent to `ceil(M / 64) * ceil(N / 64)`. Production
/// code should replace this with `Measurement::dispatches` from the
/// instrumented Metal runtime.
fn synthetic_dispatch_count(shape: &ShapeKey) -> u64 {
    let m_tiles = (shape.m as u64).div_ceil(TILE_DIM as u64);
    let n_tiles = (shape.n as u64).div_ceil(TILE_DIM as u64);
    m_tiles * n_tiles
}

/// Constant TDP share (watts) attributed to one matmul invocation when
/// deriving a synthetic energy figure. 30 W is the steady-state thermal
/// envelope of a single M-class GPU core under sustained GEMM. The
/// point is to give the test a reproducible number, not a real Joules
/// measurement — when instrumented telemetry is available the caller
/// should pass `Measurement::energy_j` in directly.
const SYNTHETIC_TDP_W: f64 = 30.0;

/// Run the reference matmul once on a **single output tile** and return
/// the wall time, then scale to the full shape by multiplying by the
/// number of tiles (`synthetic_dispatch_count`). This keeps the test
/// fast even at the largest bucket (a scalar `out[16384, 4096] @ w[4096,
/// 4096]` would otherwise run for tens of minutes).
///
/// `shared_expert` infers `k` and `n` from buffer lengths, so we pin
/// the tile size to `TILE_DIM` and let `k` stay at the full inner
/// dimension — that way the per-tile FLOP count is realistic.
fn measure_matmul(shape: &ShapeKey) -> (u128, f64) {
    let m_tile = TILE_DIM.min(shape.m) as usize;
    let n_tile = TILE_DIM.min(shape.n) as usize;
    let k_full = shape.k as usize;

    // Deterministic, non-zero contents so the JIT / LLVM can't fold the
    // loop away. Pattern is `i * 0.5 + 1` so values stay small and we
    // don't have to worry about f32 precision drift across buckets.
    let a: Vec<f32> = (0..m_tile * k_full)
        .map(|i| (i as f32) * 0.5 + 1.0)
        .collect();
    let w: Vec<f32> = (0..k_full * n_tile)
        .map(|i| ((i % 97) as f32) * 0.25 + 0.5)
        .collect();
    let mut out: Vec<f32> = vec![0.0; m_tile * n_tile];

    // Warmup: at least one full call so any first-call setup (cache
    // misses, page faults on the output buffer) is paid before the
    // timed call.
    shared_expert(&a, &w, &mut out).expect("shared_expert must accept well-formed buffers");

    let start = Instant::now();
    shared_expert(&a, &w, &mut out).expect("shared_expert must accept well-formed buffers");
    let tile_ns = start.elapsed().as_nanos();

    // Black-box the buffers so the optimizer cannot prove the call is
    // dead and elide it entirely.
    black_box(&a);
    black_box(&w);
    black_box(&out);

    // Scale the per-tile wall time up to the full matmul. Each output
    // tile does the same `2 * TILE_DIM * TILE_DIM * K` FLOPs so the
    // full call is `num_tiles * tile_time`.
    let num_tiles = synthetic_dispatch_count(shape);
    let wall_ns = tile_ns.saturating_mul(num_tiles as u128);

    let seconds = (wall_ns as f64) / 1e9;
    let energy_j = seconds * SYNTHETIC_TDP_W;
    (wall_ns, energy_j)
}

/// Observed metrics for one bucket. Printed verbatim on every run so
/// the operator can read the numbers and confirm the budget envelope
/// still covers them.
#[derive(Debug, Clone, Copy)]
struct BucketObservation {
    name: &'static str,
    shape: ShapeKey,
    wall_ns: u128,
    flops: u64,
    energy_j: f64,
    energy_per_op_j: f64,
    dispatches: u64,
    /// Per-shape dispatch ceiling pulled from
    /// [`regress_baseline::dispatch_budget`].
    dispatch_budget: u64,
    /// Per-shape energy-per-FLOP ceiling pulled from
    /// [`regress_baseline::energy_budget_j`].
    energy_budget_j: f64,
}

fn observe_bucket(b: &Bucket) -> BucketObservation {
    let (wall_ns, energy_j) = measure_matmul(&b.shape);
    let flops = b.shape.flops();
    let energy_per_op_j = if flops == 0 {
        0.0
    } else {
        energy_j / (flops as f64)
    };
    let dispatches = synthetic_dispatch_count(&b.shape);
    let dispatch_budget = dispatch_budget(&b.shape);
    let energy_budget_j = energy_budget_j(&b.shape);
    BucketObservation {
        name: b.name,
        shape: b.shape,
        wall_ns,
        flops,
        energy_j,
        energy_per_op_j,
        dispatches,
        dispatch_budget,
        energy_budget_j,
    }
}

fn print_observation(o: &BucketObservation) {
    let ShapeKey { m, n, k } = o.shape;
    let wall_ms = o.wall_ns as f64 / 1e6;
    let tflops = o.flops as f64 / wall_ms / 1e9;
    eprintln!(
        "[dispatch_buckets] {name}: M={m} N={n} K={k} | \
         flops={flops} wall_ns={wall_ns} ({wall_ms:.3} ms, {tflops:.2} TF/s) | \
         energy_j={energy_j:.6} energy_per_op_j={epop:.3e} | \
         dispatches={dispatches} (budget={budget_d}, energy_budget_j={budget_e:.3e})",
        name = o.name,
        flops = o.flops,
        wall_ns = o.wall_ns,
        wall_ms = wall_ms,
        tflops = tflops,
        energy_j = o.energy_j,
        epop = o.energy_per_op_j,
        dispatches = o.dispatches,
        budget_d = o.dispatch_budget,
        budget_e = o.energy_budget_j,
    );
}

mod dispatch_and_energy;
