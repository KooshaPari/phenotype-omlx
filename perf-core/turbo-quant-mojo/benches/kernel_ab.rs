// kernel_ab.rs — A/B benchmark: naive Rust vs SIMD-chunked Rust vs Mojo path.
//
// Measures GEMV decode throughput on Qwen 0.8B weight dimensions.
// Reports tokens/second assuming one GEMV per token decode step.

use std::time::Instant;
use turbo_quant_mojo::{gemv_decode, gemv_decode_rust_simd};

// ─── Qwen 0.8B dimensions ─────────────────────────────────────────────

const QWEN_0_8B_HIDDEN: usize = 2048;
const QWEN_0_8B_INTERMEDIATE: usize = 5120;
const QWEN_0_8B_NUM_HEADS: usize = 16;
const QWEN_0_8B_HEAD_DIM: usize = 128;

// ─── Benchmark parameters ─────────────────────────────────────────────

const WARMUP_ITERS: usize = 20;
const MAX_BENCH_TIME_MS: u64 = 8000;

// ─── Kernel trait for dispatch ─────────────────────────────────────────

trait GemvKernel {
    fn name(&self) -> &'static str;
    fn run(&self, weights: &[f32], input: &[f32], output: &mut [f32], rows: usize, cols: usize);
}

struct NaiveKernel;
impl GemvKernel for NaiveKernel {
    fn name(&self) -> &'static str {
        "gemv_decode (naive)"
    }
    fn run(&self, w: &[f32], i: &[f32], o: &mut [f32], r: usize, c: usize) {
        gemv_decode(w, i, o, r, c);
    }
}

struct SimdKernel;
impl GemvKernel for SimdKernel {
    fn name(&self) -> &'static str {
        "gemv_decode_rust_simd"
    }
    fn run(&self, w: &[f32], i: &[f32], o: &mut [f32], r: usize, c: usize) {
        gemv_decode_rust_simd(w, i, o, r, c);
    }
}

struct MojoPlaceholderKernel;
impl GemvKernel for MojoPlaceholderKernel {
    fn name(&self) -> &'static str {
        "gemv_decode_mojo (stub)"
    }
    fn run(&self, w: &[f32], i: &[f32], o: &mut [f32], r: usize, c: usize) {
        // Placeholder: delegates to naive until Mojo FFI gemv bridge lands.
        gemv_decode(w, i, o, r, c);
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────

fn make_weights(rows: usize, cols: usize) -> Vec<f32> {
    // Deterministic pseudo-random weights (seeded pattern).
    let mut w = Vec::with_capacity(rows * cols);
    for i in 0..rows * cols {
        let val = ((i as f64 * 0.618033988749895) % 1.0) as f32 - 0.5;
        w.push(val);
    }
    w
}

fn make_input(cols: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(cols);
    for i in 0..cols {
        let val = ((i as f64 * 0.381966011250105) % 1.0) as f32 - 0.5;
        v.push(val);
    }
    v
}

struct BenchResult {
    kernel: &'static str,
    rows: usize,
    cols: usize,
    us_per_call: f64,
    tokens_per_sec: f64,
    iters: usize,
}

fn bench_single(
    kernel: &dyn GemvKernel,
    weights: &[f32],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) -> BenchResult {
    // Warm up with a few iterations to settle caches
    for _ in 0..WARMUP_ITERS {
        kernel.run(weights, input, output, rows, cols);
    }

    // Adaptive: run for MAX_BENCH_TIME_MS, count iterations
    let start = Instant::now();
    let mut iters: usize = 0;
    while start.elapsed().as_millis() < MAX_BENCH_TIME_MS as u128 {
        kernel.run(weights, input, output, rows, cols);
        iters += 1;
    }
    let elapsed = start.elapsed();

    let us_per_call = elapsed.as_micros() as f64 / iters as f64;
    let tokens_per_sec = 1.0 / (us_per_call / 1_000_000.0);

    BenchResult {
        kernel: kernel.name(),
        rows,
        cols,
        us_per_call,
        tokens_per_sec,
        iters,
    }
}

fn print_table(results: &[BenchResult]) {
    println!();
    println!(
        "{:<30} {:>6}x{:<6} {:>12} {:>14}",
        "Kernel", "Rows", "Cols", "us/call", "tokens/sec"
    );
    println!("{}", "-".repeat(70));
    for r in results {
        println!(
            "{:<30} {:>6}x{:<6} {:>12.1} {:>14.0}",
            r.kernel, r.rows, r.cols, r.us_per_call, r.tokens_per_sec
        );
    }
    println!();
}

// ─── Main benchmark ────────────────────────────────────────────────────

fn main() {
    println!("=== Qwen 0.8B GEMV Decode Kernel A/B Benchmark ===");
    println!();
    println!("Model: Qwen 0.8B");
    println!(
        "  hidden={}, intermediate={}, num_heads={}, head_dim={}",
        QWEN_0_8B_HIDDEN, QWEN_0_8B_INTERMEDIATE, QWEN_0_8B_NUM_HEADS, QWEN_0_8B_HEAD_DIM
    );
    println!(
        "  warmup={}, max_time={}ms",
        WARMUP_ITERS, MAX_BENCH_TIME_MS
    );
    println!();

    // Test dimensions representative of Qwen 0.8B layer operations:
    //   - Q/K/V projection:    hidden → hidden     (2048 × 2048)
    //   - Gate/Up projection:  hidden → intermediate (2048 × 5120)
    //   - Down projection:     intermediate → hidden (5120 × 2048)
    //   - Head slice:          head_dim × hidden     (128 × 2048)
    let dimensions: &[(usize, usize)] = &[
        (QWEN_0_8B_HIDDEN, QWEN_0_8B_HIDDEN),
        (QWEN_0_8B_HIDDEN, QWEN_0_8B_INTERMEDIATE),
        (QWEN_0_8B_INTERMEDIATE, QWEN_0_8B_HIDDEN),
        (QWEN_0_8B_HEAD_DIM, QWEN_0_8B_HIDDEN),
    ];

    let kernels: Vec<Box<dyn GemvKernel>> = vec![
        Box::new(NaiveKernel),
        Box::new(SimdKernel),
        Box::new(MojoPlaceholderKernel),
    ];

    let mut all_results: Vec<BenchResult> = Vec::new();

    for &(rows, cols) in dimensions {
        let weights = make_weights(rows, cols);
        let input = make_input(cols);
        let mut output = vec![0.0f32; rows];

        println!("--- {rows}x{cols} ---");

        for kernel in &kernels {
            let result = bench_single(kernel.as_ref(), &weights, &input, &mut output, rows, cols);
            println!(
                "  {:<28} {:>10.1} us/call  {:>10.0} tok/s  ({:>5} iters)",
                result.kernel, result.us_per_call, result.tokens_per_sec, result.iters
            );
            all_results.push(result);
        }
        println!();
    }

    println!();
    println!("=== Summary Table ===");
    print_table(&all_results);

    // Compute speedup: SIMD vs naive, per dimension
    println!("=== Speedup: SIMD / Naive ===");
    for chunk in all_results.chunks(3) {
        // Each dimension produces 3 results: naive, simd, mojo
        if chunk.len() >= 2 {
            let naive = &chunk[0];
            let simd = &chunk[1];
            let speedup = naive.us_per_call / simd.us_per_call;
            println!("  {}x{}: {:.2}x", naive.rows, naive.cols, speedup);
        }
    }
    println!();
    println!("=== Mojo FFI path: stub (delegates to naive until Mojo gemv bridge lands) ===");
    println!();
}
