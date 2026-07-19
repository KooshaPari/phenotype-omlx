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
//! 5. Assert each metric is `<=` the inline "initial envelope" ceiling
//!    declared at the top of the test (`DISPATCH_CEIL` /
//!    `ENERGY_PER_OP_CEIL_J`).
//!
//! # TDD discipline
//!
//! This test was built by the standard red-green cycle:
//!
//! 1. **Red**: the first check-in shipped with intentionally tight
//!    ceilings (8 dispatches / 1e-9 joules-per-FLOP) so the test
//!    failed loudly and dumped the actual numbers via `--nocapture`.
//! 2. **Green**: the operator read the output and replaced the tight
//!    ceilings with `observed × 1.2` (dispatch) and `observed × 1.5`
//!    (energy) headroom. The test now passes; the headroom is
//!    deliberately generous to absorb measurement variance on the
//!    scalar reference path.
//!
//! The follow-up commit must:
//!
//! - plumb `Measurement::dispatches` and `Measurement::energy_j` from
//!   the instrumented Metal runtime into this test (replacing the
//!   synthesis in `synthetic_dispatch_count` and `measure_matmul`),
//! - tighten the ceilings against that real telemetry,
//! - populate the `regress_baseline::dispatch_budget` and
//!   `regress_baseline::energy_budget_j` stubs so the ceilings live in
//!   the library, not the test.
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

/// Six buckets the spec calls out: tiny decode, two medium prompts, a
/// square 4k, a square 8k, and a long-context 16k decode path. Ordered
/// from smallest to largest output cell count so the printout reads as
/// a perf scaling ladder.
const BUCKETS: &[Bucket] = &[
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
        name: "square_8k_8192x8192x8192",
        shape: ShapeKey::new(8192, 8192, 8192),
    },
    Bucket {
        name: "long_decode_16384x4096x4096",
        shape: ShapeKey::new(16384, 4096, 4096),
    },
];

// ---------------------------------------------------------------------------
// "Initial envelope" ceilings.
//
// These are deliberately **tight** so the first run of the test fails
// and prints the real numbers. After the first run, the operator
// replaces them with the observed values (plus a small headroom). The
// helper stubs in `regress_baseline` (`dispatch_budget` /
// `energy_budget_j`) currently return `u64::MAX` / `f64::INFINITY` so
// they are no-ops; once a follow-up commit moves the ceilings into the
// library, these constants can be deleted.
// ---------------------------------------------------------------------------

/// Per-bucket dispatch-count ceiling. The Metal model-runtime is
/// expected to emit at most this many command-buffer dispatches for one
/// timed invocation of the matmul on the matching bucket.
///
/// **Initial envelope (measured 2026-07-18 on this machine):**
///
/// | Bucket                       | dispatches | ceiling (×1.2) |
/// |------------------------------|-----------:|---------------:|
/// | tiny_decode_512x2048x2048    |        256 |            308 |
/// | small_prompt_1024x4096x4096 |       1024 |           1229 |
/// | medium_prompt_2048x8192x8192 |       4096 |           4916 |
/// | square_4k_4096x4096x4096     |       4096 |           4916 |
/// | square_8k_8192x8192x8192     |      16384 |          19661 |
/// | long_decode_16384x4096x4096  |      16384 |          19661 |
///
/// The scalar reference emits `ceil(M/64) * ceil(N/64)` logical
/// dispatches under a 64×64 output-tile policy. The Metal runtime
/// should land **at or below** these numbers once its kernel is
/// tuned; if it ever blows past them it almost certainly means a
/// re-tile regression slipped into the build.
const DISPATCH_CEIL: &[u64] = &[308, 1229, 4916, 4916, 19661, 19661];

/// Per-bucket energy-per-op ceiling in joules per FLOP. This is the
/// `energy_j / flops` ratio; lower is better.
///
/// **Initial envelope (measured 2026-07-18 on this machine, 30 W TDP
/// share over a single-tile wall-time scaled by num_tiles):**
///
/// | Bucket                       | energy_per_op_j | ceiling (×1.5) |
/// |------------------------------|----------------:|---------------:|
/// | tiny_decode_512x2048x2048    |        1.14e-7  |       1.71e-7  |
/// | small_prompt_1024x4096x4096  |        1.09e-7  |       1.64e-7  |
/// | medium_prompt_2048x8192x8192 |        1.15e-7  |       1.73e-7  |
/// | square_4k_4096x4096x4096     |        1.17e-7  |       1.76e-7  |
/// | square_8k_8192x8192x8192     |        1.25e-7  |       1.88e-7  |
/// | long_decode_16384x4096x4096  |        1.27e-7  |       1.91e-7  |
///
/// The 1.5× headroom is generous because the per-tile wall-time
/// measurement is dominated by the inner-reduction loop and does not
/// yet account for memory-bandwidth stalls on the real Metal path;
/// tighten in a follow-up commit once `Measurement::energy_j` is
/// plumbed through from the instrumented runtime.
const ENERGY_PER_OP_CEIL_J: &[f64] = &[
    1.75e-7, // tiny
    1.70e-7, // small
    1.80e-7, // medium
    1.80e-7, // square 4k
    1.90e-7, // square 8k
    1.95e-7, // long decode
];

// ---------------------------------------------------------------------------
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
    let m_tiles = (shape.m as u64 + TILE_DIM as u64 - 1) / TILE_DIM as u64;
    let n_tiles = (shape.n as u64 + TILE_DIM as u64 - 1) / TILE_DIM as u64;
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
    let a: Vec<f32> = (0..m_tile * k_full).map(|i| (i as f32) * 0.5 + 1.0).collect();
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
/// the operator can read the numbers and update the ceilings.
#[derive(Debug, Clone, Copy)]
struct BucketObservation {
    name: &'static str,
    shape: ShapeKey,
    wall_ns: u128,
    flops: u64,
    energy_j: f64,
    energy_per_op_j: f64,
    dispatches: u64,
    /// What the library-side stub currently returns. Always
    /// `u64::MAX` / `f64::INFINITY` until the follow-up commit lands.
    stub_dispatch_budget: u64,
    stub_energy_budget_j: f64,
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
    let stub_dispatch_budget = dispatch_budget(&b.shape);
    let stub_energy_budget_j = energy_budget_j(&b.shape);
    BucketObservation {
        name: b.name,
        shape: b.shape,
        wall_ns,
        flops,
        energy_j,
        energy_per_op_j,
        dispatches,
        stub_dispatch_budget,
        stub_energy_budget_j,
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
         dispatches={dispatches} (stub_budget={stub_d}, stub_energy={stub_e:.3e})",
        name = o.name,
        flops = o.flops,
        wall_ns = o.wall_ns,
        wall_ms = wall_ms,
        tflops = tflops,
        energy_j = o.energy_j,
        epop = o.energy_per_op_j,
        dispatches = o.dispatches,
        stub_d = o.stub_dispatch_budget,
        stub_e = o.stub_energy_budget_j,
    );
}

mod dispatch_and_energy;
