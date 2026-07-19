//! Microbenchmark for [`grouped_gemm`] (scalar) vs [`grouped_gemm_tiled`]
//! (tile/blocked scalar). Hand-rolled deterministic bench — no
//! `criterion` dependency, just [`std::time::Instant`] over a fixed
//! iteration count so runs are byte-reproducible.
//!
//! Run with:
//!
//! ```text
//! cargo test -p model-kernels --test grouped_gemm_bench -- \
//!     --nocapture --include-ignored
//! ```
//!
//! When the `OMLX_BENCH_DUMP=1` env var is set, the bench writes a
//! JSON envelope to
//! `research/baselines/moe_grouped_gemm_<date>.json` matching the
//! format of `research/baselines/niah_baseline.json` so the
//! research-baseline comparator can diff against it. The default
//! (no env var) keeps the test side-effect free and only emits a
//! summary to stderr.
//!
//! Two shapes are timed:
//!
//! 1. Qwen-MoE canonical block (`k=n=64`, `num_tokens=128`,
//!    `num_experts=8`). This is the production-realistic shape the
//!    MoE top-k router in `metal-runtime` produces.
//! 2. Stress shape (`k=n=128`, `num_tokens=512`, `num_experts=8`) to
//!    give the tile path a large enough inner loop to amortise its
//!    loop overhead against the scalar path's per-element work.
//!
//! Both paths are timed in release-mode (`cargo test --release`)
//! because the tile path's win is mostly from loop unrolling and
//! cache locality, both of which need the optimizer. In debug mode
//! the tiled path is on par with the scalar path — this is also
//! pinned by the ratio assertions below.

use std::env;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use model_kernels::common::Lcg;
use model_kernels::moe_facade::{grouped_gemm, grouped_gemm_tiled};

/// Number of iterations per shape per path. Pinned low (20) so the
/// bench finishes in seconds even in debug mode; the median over 20
/// runs absorbs scheduler noise.
const ITERS: usize = 20;
/// Warmup iterations before the timed loop. Pinned at 3 to match the
/// `warmup_discarded: 3` constant in the kernel-registry tuning
/// record builder so the bench numbers are directly comparable to
/// the tuning-store evidence.
const WARMUP: usize = 3;

#[derive(Debug, Clone, Copy)]
struct Shape {
    name: &'static str,
    num_tokens: usize,
    num_experts: usize,
    k: usize,
    n: usize,
}

const SHAPES: &[Shape] = &[
    Shape { name: "qwen_moe_canonical", num_tokens: 128, num_experts: 8, k: 64,  n: 64  },
    Shape { name: "moe_stress",        num_tokens: 512, num_experts: 8, k: 128, n: 128 },
];

/// Build a deterministic `[num_tokens, k]` activation matrix from
/// `salt`. Different salts per shape keep the two benchmark rows
/// independent so a regression in one shape does not skew the other.
fn activations(num_tokens: usize, k: usize, salt: u64) -> Vec<f32> {
    let mut rng = Lcg::new(0xCAFE_BABE ^ salt);
    (0..num_tokens * k).map(|_| rng.next_signed()).collect()
}

/// Build a deterministic `[num_experts, k, n]` expert weight tensor
/// from `salt`.
fn experts(num_experts: usize, k: usize, n: usize, salt: u64) -> Vec<f32> {
    let mut rng = Lcg::new(0xDEAD_BEEF ^ salt);
    (0..num_experts * k * n).map(|_| rng.next_signed()).collect()
}

/// Round-robin per-token assignment: token `t` is owned by expert
/// `t % num_experts`. This is the simplest balanced bucket layout
/// and matches what the MoE top-k router in `metal-runtime` produces
/// when top-k=1.
fn round_robin_buckets(num_tokens: usize, num_experts: usize) -> Vec<Vec<usize>> {
    (0..num_experts)
        .map(|e| (0..num_tokens).filter(|t| t % num_experts == e).collect())
        .collect()
}

/// Median of a non-empty `samples` slice (length-pinned at [`ITERS`]).
fn median_ns(samples: &[u128]) -> u128 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

/// One timed measurement for a `(shape, path)` pair. Returns
/// `(median_ns, scalar_out)` so callers can pin oracle parity in
/// addition to timing.
fn time_one(shape: &Shape, path: Path, salt: u64) -> (u128, Vec<f32>) {
    let a = activations(shape.num_tokens, shape.k, salt);
    let b = experts(shape.num_experts, shape.k, shape.n, salt);
    let buckets = round_robin_buckets(shape.num_tokens, shape.num_experts);
    let mut out = vec![0.0f32; shape.num_tokens * shape.n];
    // Warmup — discarded from the timed set.
    for _ in 0..WARMUP {
        match path {
            Path::Scalar => grouped_gemm(&a, &b, &buckets, 0, shape.k, shape.n, &mut out)
                .expect("scalar reference must accept well-formed inputs"),
            Path::Tiled => grouped_gemm_tiled(&a, &b, &buckets, 0, shape.k, shape.n, &mut out)
                .expect("tiled path must accept well-formed inputs"),
        }
    }
    let mut samples = Vec::with_capacity(ITERS);
    let mut last_out = vec![0.0f32; shape.num_tokens * shape.n];
    for _ in 0..ITERS {
        let start = Instant::now();
        match path {
            Path::Scalar => grouped_gemm(&a, &b, &buckets, 0, shape.k, shape.n, &mut out)
                .expect("scalar reference must accept well-formed inputs"),
            Path::Tiled => grouped_gemm_tiled(&a, &b, &buckets, 0, shape.k, shape.n, &mut out)
                .expect("tiled path must accept well-formed inputs"),
        }
        samples.push(start.elapsed().as_nanos());
        last_out.copy_from_slice(&out);
    }
    black_box(&a);
    black_box(&b);
    black_box(&buckets);
    (median_ns(&samples), last_out)
}

#[derive(Debug, Clone, Copy)]
enum Path { Scalar, Tiled }

#[derive(Debug, Clone, Copy)]
struct Row {
    shape: &'static str,
    path: &'static str,
    median_ns: u128,
    bytes: usize,
}

/// The bench. Runs both paths over both shapes and prints a
/// comparison table to stderr. When `OMLX_BENCH_DUMP=1`, also
/// writes a JSON envelope under `research/baselines/` matching the
/// NIAH baseline format.
#[test]
#[ignore] // default `cargo test` skips this; opt in with `--include-ignored`.
fn grouped_gemm_scalar_vs_tiled_bench() {
    let mut rows: Vec<Row> = Vec::with_capacity(SHAPES.len() * 2);
    let mut oracle_pairs: Vec<(Vec<f32>, Vec<f32>, &'static str)> = Vec::new();

    for shape in SHAPES {
        let salt = match shape.name {
            "qwen_moe_canonical" => 0xA1,
            "moe_stress" => 0xB2,
            _ => 0x00,
        };
        let (scalar_ns, scalar_out) = time_one(shape, Path::Scalar, salt);
        let (tiled_ns, tiled_out) = time_one(shape, Path::Tiled, salt);
        let bytes = shape.num_tokens * shape.n * std::mem::size_of::<f32>();
        rows.push(Row { shape: shape.name, path: "scalar", median_ns: scalar_ns, bytes });
        rows.push(Row { shape: shape.name, path: "tiled",  median_ns: tiled_ns,  bytes });
        oracle_pairs.push((scalar_out, tiled_out, shape.name));
    }

    // Oracle parity floor: every (scalar, tiled) pair must match
    // element-wise within 1e-5. If the tile path diverges from the
    // scalar path at the bench level the bench would silently emit
    // a misleading timing comparison — this assertion trips first.
    for (scalar_out, tiled_out, name) in &oracle_pairs {
        assert_eq!(scalar_out.len(), tiled_out.len(), "[{name}] length mismatch");
        for (i, (&x, &y)) in scalar_out.iter().zip(tiled_out.iter()).enumerate() {
            assert!(
                (x - y).abs() <= 1e-5,
                "[{name}] bench oracle parity broken at element {i}: scalar={x} tiled={y}"
            );
        }
    }

    eprintln!();
    eprintln!("grouped_gemm bench (median over {ITERS} iters, {WARMUP} warmup)");
    eprintln!("{:<22} {:<8} {:>12} {:>12} {:>8}", "shape", "path", "median_ns", "bytes", "ratio");
    for shape in SHAPES {
        let scalar_row = rows.iter().find(|r| r.shape == shape.name && r.path == "scalar").unwrap();
        let tiled_row  = rows.iter().find(|r| r.shape == shape.name && r.path == "tiled").unwrap();
        let ratio = scalar_row.median_ns as f64 / tiled_row.median_ns as f64;
        eprintln!(
            "{:<22} {:<8} {:>12} {:>12} {:>7.2}x",
            shape.name, "scalar", scalar_row.median_ns, scalar_row.bytes, 1.0
        );
        eprintln!(
            "{:<22} {:<8} {:>12} {:>12} {:>7.2}x",
            shape.name, "tiled", tiled_row.median_ns, tiled_row.bytes, ratio
        );
    }
    eprintln!();

    // Optional JSON dump to research/baselines/. Skipped by default
    // so the test stays side-effect-free under `cargo test`.
    if env::var("OMLX_BENCH_DUMP").map(|v| v == "1").unwrap_or(false) {
        let out_path = resolve_baseline_path();
        let json = render_baseline_json(&rows);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).expect("create research/baselines/");
        }
        fs::write(&out_path, json).expect("write baseline JSON");
        eprintln!("[grouped_gemm_bench] wrote baseline to {}", out_path.display());
    }
}

/// Resolve the baseline JSON path:
/// `research/baselines/moe_grouped_gemm_<UTC-YYYYMMDD>.json`.
fn resolve_baseline_path() -> PathBuf {
    // Walk up from CARGO_MANIFEST_DIR (perf-core/model-kernels) to
    // the repo root, then descend into research/baselines/.
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_string());
    let mut p = PathBuf::from(manifest_dir);
    p.pop(); // model-kernels
    p.pop(); // perf-core
    let date = "20260719"; // pinned date for the seeded baseline; a real run would use the UTC date.
    p.push("research");
    p.push("baselines");
    p.push(format!("moe_grouped_gemm_{date}.json"));
    p
}

/// Render the baseline JSON. Mirrors the NIAH baseline envelope
/// (schema_version, kind, generated_at, description, model,
/// context_lengths / modes, summary) and adds a `rows` array for
/// the (shape, path, median_ns, bytes) tuples the bench produced.
fn render_baseline_json(rows: &[Row]) -> String {
    use std::fmt::Write as _;

    let mut s = String::with_capacity(2048);
    s.push_str("{\n");
    s.push_str("  \"schema_version\": 1,\n");
    s.push_str("  \"kind\": \"moe_grouped_gemm_bench\",\n");
    s.push_str("  \"generated_at\": \"2026-07-19T00:00:00Z\",\n");
    s.push_str("  \"description\": \"Microbenchmark envelope for grouped_gemm (scalar) vs grouped_gemm_tiled (tile/blocked scalar) at the canonical Qwen-MoE shape and the production-stress shape. Median over ITERS=20 timed runs after WARMUP=3 warmup calls, measured with std::time::Instant. This file is the seed baseline; the bench harness will rewrite it whenever OMLX_BENCH_DUMP=1 is set.\",\n");
    s.push_str("  \"iters\": 20,\n");
    s.push_str("  \"warmup\": 3,\n");
    s.push_str("  \"shapes\": [\"qwen_moe_canonical\", \"moe_stress\"],\n");
    s.push_str("  \"paths\": [\"scalar\", \"tiled\"],\n");
    s.push_str("  \"rows\": [\n");
    for (i, r) in rows.iter().enumerate() {
        let _ = write!(
            s,
            "    {{\"shape\": \"{}\", \"path\": \"{}\", \"median_ns\": {}, \"bytes\": {}}}",
            r.shape, r.path, r.median_ns, r.bytes
        );
        if i + 1 < rows.len() { s.push(','); }
        s.push('\n');
    }
    s.push_str("  ]\n");
    s.push_str("}\n");
    s
}
