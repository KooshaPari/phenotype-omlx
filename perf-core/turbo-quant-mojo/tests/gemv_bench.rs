// Integration tests for GEMV decode kernels: correctness + directional timing.

use std::time::Instant;
use turbo_quant_mojo::{gemv_decode, gemv_decode_rust_simd};

/// Assert two f32 values are close within a relative tolerance.
fn assert_close(a: f32, b: f32, label: &str) {
    let diff = (a - b).abs();
    let denom = a.abs().max(b.abs()).max(1.0);
    let rel = diff / denom;
    assert!(rel < 1e-5, "{label}: {a} vs {b} (rel={rel:.2e})");
}

/// Naive reference for correctness checks.
fn gemv_reference(weights: &[f32], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += weights[r * cols + c] * input[c];
        }
        out[r] = sum;
    }
    out
}

fn make_data(rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
    let weights: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001 - 1.0).collect();
    let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01).collect();
    (weights, input)
}

// ─── Correctness: gemv_decode ──────────────────────────────────────────

#[test]
fn gemv_decode_matches_reference_64x128() {
    let (w, v) = make_data(64, 128);
    let expected = gemv_reference(&w, &v, 64, 128);
    let mut out = vec![0.0f32; 64];
    gemv_decode(&w, &v, &mut out, 64, 128);
    for (a, b) in expected.iter().zip(out.iter()) {
        assert!((a - b).abs() < 1e-5, "mismatch: {a} vs {b}");
    }
}

#[test]
fn gemv_decode_matches_reference_128x256() {
    let (w, v) = make_data(128, 256);
    let expected = gemv_reference(&w, &v, 128, 256);
    let mut out = vec![0.0f32; 128];
    gemv_decode(&w, &v, &mut out, 128, 256);
    for (a, b) in expected.iter().zip(out.iter()) {
        assert!((a - b).abs() < 1e-4, "mismatch: {a} vs {b}");
    }
}

#[test]
fn gemv_decode_matches_reference_256x512() {
    let (w, v) = make_data(256, 512);
    let expected = gemv_reference(&w, &v, 256, 512);
    let mut out = vec![0.0f32; 256];
    gemv_decode(&w, &v, &mut out, 256, 512);
    for (a, b) in expected.iter().zip(out.iter()) {
        assert!((a - b).abs() < 1e-3, "mismatch: {a} vs {b}");
    }
}

// ─── Correctness: gemv_decode_rust_simd ────────────────────────────────

#[test]
fn gemv_simd_matches_reference_64x128() {
    let (w, v) = make_data(64, 128);
    let expected = gemv_reference(&w, &v, 64, 128);
    let mut out = vec![0.0f32; 64];
    gemv_decode_rust_simd(&w, &v, &mut out, 64, 128);
    for (a, b) in expected.iter().zip(out.iter()) {
        assert_close(*a, *b, "64x128");
    }
}

#[test]
fn gemv_simd_matches_reference_128x256() {
    let (w, v) = make_data(128, 256);
    let expected = gemv_reference(&w, &v, 128, 256);
    let mut out = vec![0.0f32; 128];
    gemv_decode_rust_simd(&w, &v, &mut out, 128, 256);
    for (a, b) in expected.iter().zip(out.iter()) {
        assert_close(*a, *b, "128x256");
    }
}

#[test]
fn gemv_simd_matches_reference_256x512() {
    let (w, v) = make_data(256, 512);
    let expected = gemv_reference(&w, &v, 256, 512);
    let mut out = vec![0.0f32; 256];
    gemv_decode_rust_simd(&w, &v, &mut out, 256, 512);
    for (a, b) in expected.iter().zip(out.iter()) {
        assert_close(*a, *b, "256x512");
    }
}

// ─── Edge cases ─────────────────────────────────────────────────────────

#[test]
fn gemv_decode_single_row() {
    let (w, v) = make_data(1, 64);
    let expected = gemv_reference(&w, &v, 1, 64);
    let mut out = vec![0.0f32; 1];
    gemv_decode(&w, &v, &mut out, 1, 64);
    assert!((expected[0] - out[0]).abs() < 1e-5);
}

#[test]
fn gemv_decode_single_column() {
    let weights: Vec<f32> = vec![2.0, 3.0, 5.0];
    let input: Vec<f32> = vec![7.0];
    let expected = gemv_reference(&weights, &input, 3, 1);
    let mut out = vec![0.0f32; 3];
    gemv_decode(&weights, &input, &mut out, 3, 1);
    assert_eq!(out, expected);
}

#[test]
fn gemv_decode_single_element() {
    let weights: Vec<f32> = vec![4.0];
    let input: Vec<f32> = vec![3.0];
    let mut out = vec![0.0f32; 1];
    gemv_decode(&weights, &input, &mut out, 1, 1);
    assert!((out[0] - 12.0).abs() < 1e-6);
}

#[test]
fn gemv_simd_single_row() {
    let (w, v) = make_data(1, 64);
    let expected = gemv_reference(&w, &v, 1, 64);
    let mut out = vec![0.0f32; 1];
    gemv_decode_rust_simd(&w, &v, &mut out, 1, 64);
    assert!((expected[0] - out[0]).abs() < 1e-5);
}

#[test]
fn gemv_simd_single_column() {
    let weights: Vec<f32> = vec![2.0, 3.0, 5.0];
    let input: Vec<f32> = vec![7.0];
    let expected = gemv_reference(&weights, &input, 3, 1);
    let mut out = vec![0.0f32; 3];
    gemv_decode_rust_simd(&weights, &input, &mut out, 3, 1);
    assert_eq!(out, expected);
}

#[test]
fn gemv_simd_single_element() {
    let weights: Vec<f32> = vec![4.0];
    let input: Vec<f32> = vec![3.0];
    let mut out = vec![0.0f32; 1];
    gemv_decode_rust_simd(&weights, &input, &mut out, 1, 1);
    assert!((out[0] - 12.0).abs() < 1e-6);
}

// ─── Directional timing comparison ─────────────────────────────────────

fn bench_kernel(
    name: &str,
    weights: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
    iters: usize,
    kernel: fn(&[f32], &[f32], &mut [f32], usize, usize),
) {
    let mut out = vec![0.0f32; rows];
    // Warm up
    for _ in 0..3 {
        kernel(weights, input, &mut out, rows, cols);
    }
    let start = Instant::now();
    for _ in 0..iters {
        kernel(weights, input, &mut out, rows, cols);
    }
    let elapsed = start.elapsed();
    let us_per_call = elapsed.as_micros() as f64 / iters as f64;
    println!("  {name}: {us_per_call:.1} us/call ({iters} iters, {rows}x{cols})");
}

#[test]
fn gemv_bench_comparison_print() {
    let sizes: &[(usize, usize)] = &[(64, 128), (128, 256), (256, 512)];
    let iters = 500;
    println!("\n=== GEMV decode kernel timing (directional, not statistical) ===");
    for &(rows, cols) in sizes {
        let (w, v) = make_data(rows, cols);
        println!("\nMatrix {rows}x{cols}:");
        bench_kernel("gemv_decode       ", &w, &v, rows, cols, iters, gemv_decode);
        bench_kernel(
            "gemv_decode_simd  ",
            &w,
            &v,
            rows,
            cols,
            iters,
            gemv_decode_rust_simd,
        );
    }
    println!();
}
