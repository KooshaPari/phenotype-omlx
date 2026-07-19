//! FR-2: f32 min/max reduction for TurboQuant group scaling.
//!
//! On `aarch64`, uses explicit NEON intrinsics (`vminq_f32` / `vmaxq_f32`) with a
//! scalar tail for lengths not divisible by four. All other targets dispatch to
//! [`scalar_min_max`], which is also the parity oracle in unit tests.

#[cfg(any(test, not(target_arch = "aarch64")))]
pub(crate) fn scalar_min_max(data: &[f32]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &value in data {
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
    }
    (min, max)
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn min_max(data: &[f32]) -> (f32, f32) {
    use core::arch::aarch64::{
        vdupq_n_f32, vld1q_f32, vmaxq_f32, vmaxvq_f32, vminq_f32, vminvq_f32,
    };

    unsafe {
        let mut min_vector = vdupq_n_f32(f32::INFINITY);
        let mut max_vector = vdupq_n_f32(f32::NEG_INFINITY);
        let vector_len = data.len() / 4 * 4;
        let mut index = 0;

        while index < vector_len {
            let values = vld1q_f32(data.as_ptr().add(index));
            min_vector = vminq_f32(min_vector, values);
            max_vector = vmaxq_f32(max_vector, values);
            index += 4;
        }

        let mut min = vminvq_f32(min_vector);
        let mut max = vmaxvq_f32(max_vector);
        for &value in &data[vector_len..] {
            if value < min {
                min = value;
            }
            if value > max {
                max = value;
            }
        }
        (min, max)
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn min_max(data: &[f32]) -> (f32, f32) {
    scalar_min_max(data)
}

#[cfg(test)]
mod tests {
    use super::{min_max, scalar_min_max};
    use std::hint::black_box;
    use std::time::Instant;

    fn data(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32) * 0.013).sin() + ((i as f32) * 0.007).cos())
            .collect()
    }

    #[test]
    fn scalar_and_dispatched_results_match() {
        for len in [1, 3, 4, 5, 17, 64, 129, 1024] {
            let values = data(len);
            assert_eq!(scalar_min_max(&values), min_max(&values));
        }
    }

    /// arm64 CI gate (FR-2): NEON path must match the scalar oracle on tails and
    /// unaligned sub-slices. Skipped automatically on non-aarch64 hosts.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_handles_unaligned_slice_and_scalar_tail() {
        let values = data(131);
        for len in [1, 3, 5, 7, 9, 17, 33, 65, 129] {
            let slice = &values[1..=len];
            assert_eq!(scalar_min_max(slice), min_max(slice));
        }
    }

    fn measure(values: &[f32], iterations: usize, neon: bool) -> Vec<f64> {
        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            let result = if neon {
                min_max(black_box(values))
            } else {
                scalar_min_max(black_box(values))
            };
            black_box(result);
            samples.push(start.elapsed().as_secs_f64());
        }
        samples.sort_by(|a, b| a.total_cmp(b));
        samples
    }

    #[test]
    #[ignore]
    fn microbench_scalar_vs_neon_min_max() {
        let values = data(1 << 20);
        for _ in 0..4 {
            black_box(scalar_min_max(black_box(&values)));
            black_box(min_max(black_box(&values)));
        }

        let mut scalar = Vec::new();
        let mut dispatched = Vec::new();
        for iteration in 0..12 {
            if iteration % 2 == 0 {
                scalar.extend(measure(&values, 1, false));
                dispatched.extend(measure(&values, 1, true));
            } else {
                dispatched.extend(measure(&values, 1, true));
                scalar.extend(measure(&values, 1, false));
            }
        }
        scalar.sort_by(|a, b| a.total_cmp(b));
        dispatched.sort_by(|a, b| a.total_cmp(b));

        let scalar_median = scalar[scalar.len() / 2];
        let dispatched_median = dispatched[dispatched.len() / 2];
        let bytes = values.len() as f64 * std::mem::size_of::<f32>() as f64;
        eprintln!(
            "min_max: scalar={:.3}ms ({:.2} GB/s), dispatched={:.3}ms ({:.2} GB/s), speedup={:.2}x",
            scalar_median * 1e3,
            bytes / scalar_median / 1e9,
            dispatched_median * 1e3,
            bytes / dispatched_median / 1e9,
            scalar_median / dispatched_median,
        );
    }
}
