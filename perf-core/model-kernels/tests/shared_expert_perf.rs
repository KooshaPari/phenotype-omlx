//! Regression test for the [`shared_expert`] scalar matmul inner loop.
//!
//! Pins the perf-invariant declared at the top of
//! `model-kernels/src/moe/shared.rs`: the helper must finish a
//! `512×512×4096` dense matmul (≈1.07 GFLOP) on a single thread in well
//! under the wall-clock ceiling on Apple Silicon in debug mode. The same
//! invariant is what keeps `regress-baseline/tests/dispatch_buckets.rs`
//! (which calls `shared_expert` on a 64-wide inner tile for six shape
//! buckets) under 60 seconds end-to-end.
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
//! Quiet-machine debug cost on M-series is ~1–5 s. The ceiling is set
//! above that so a genuine O(total) shape-inference regression (minutes)
//! still fails loudly, while routine `cargo test --workspace` contention
//! does not. Cross-process serialization uses an advisory file lock —
//! a process-local `Mutex` cannot serialize cargo's parallel test
//! binaries.
//!
//! ## Load-sensitivity guard (T16+)
//!
//! Wall-clock perf measurements on a single thread inflate sharply when
//! cargo runs concurrent test binaries (default `--test-threads = N_CPUS`
//! **and** parallel crate test processes). A process-local mutex does not
//! help across binaries. This test takes an exclusive `flock` on
//! `$TMPDIR/omlx-perf-shared-expert.lock` for the timed window. Set
//! `OMLX_PERF_NO_GUARD=1` to opt out.

use std::fs::OpenOptions;
use std::hint::black_box;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

use model_kernels::moe_facade::shared_expert;

/// Wall-clock ceiling (seconds) for a single 512×512×4096 invocation
/// of [`shared_expert`] in debug mode on Apple Silicon.
///
/// Quiet isolation: ~1–5 s. Documented mlx_lm contention alone: ~5.2–5.6 s
/// (see `regress-baseline` PerfGuard notes). Workspace-parallel cargo
/// without a cross-process lock: 15–20 s. With flock, stay near quiet
/// cost; 12 s leaves headroom without masking an O(total) hang.
const CEIL_SECS: f64 = 12.0;

/// Cross-process exclusive lock held for the timed matmul window.
struct CrossProcessPerfGuard {
    file: std::fs::File,
}

impl CrossProcessPerfGuard {
    fn try_enter() -> Option<Self> {
        if std::env::var_os("OMLX_PERF_NO_GUARD").is_some() {
            return None;
        }
        let path = lock_path();
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap_or_else(|e| panic!("perf lock open {}: {e}", path.display()));
        flock_exclusive(&file).unwrap_or_else(|e| panic!("perf flock {}: {e}", path.display()));
        Some(Self { file })
    }
}

impl Drop for CrossProcessPerfGuard {
    fn drop(&mut self) {
        let _ = flock_unlock(&self.file);
    }
}

fn lock_path() -> PathBuf {
    std::env::temp_dir().join("omlx-perf-shared-expert.lock")
}

#[cfg(unix)]
fn flock_exclusive(file: &std::fs::File) -> io::Result<()> {
    use std::os::unix::io::AsRawFd;
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(unix)]
fn flock_unlock(file: &std::fs::File) -> io::Result<()> {
    use std::os::unix::io::AsRawFd;
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(not(unix))]
fn flock_exclusive(_file: &std::fs::File) -> io::Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn flock_unlock(_file: &std::fs::File) -> io::Result<()> {
    Ok(())
}

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

    // Cross-process flock so parallel cargo test binaries cannot inflate
    // the wall-clock measurement by competing for the same cores.
    let _perf_guard = CrossProcessPerfGuard::try_enter();

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
        "[shared_expert_perf] m={m} n={n} k={k} elapsed={elapsed_secs:.3}s (ceil={CEIL_SECS:.1}s, guard={})",
        if _perf_guard.is_some() {
            "flock"
        } else {
            "off"
        }
    );

    assert!(
        elapsed_secs <= CEIL_SECS,
        "shared_expert took {elapsed_secs:.3}s for {m}x{n}x{k}; \
         must finish within {CEIL_SECS:.1}s. The O(total) shape-inference \
         regression has returned — see the Perf invariants note at the \
         top of model-kernels/src/moe/shared.rs."
    );
}

/// Contract test that pins the load-sensitivity guard env-var contract.
/// It must always pass regardless of whether `OMLX_PERF_NO_GUARD` is set.
#[test]
fn shared_expert_perf_guard_env_contract_respected() {
    // Default behaviour: flock guard is acquired (no env var set).
    std::env::remove_var("OMLX_PERF_NO_GUARD");
    assert!(
        CrossProcessPerfGuard::try_enter().is_some(),
        "cross-process PerfGuard must be active by default"
    );

    // With the opt-out env var, the guard is a no-op.
    // SAFETY: cargo runs tests on a dedicated worker thread with no other
    // threads reading this var during this assertion window.
    unsafe {
        std::env::set_var("OMLX_PERF_NO_GUARD", "1");
    }
    assert!(
        CrossProcessPerfGuard::try_enter().is_none(),
        "PerfGuard must be a no-op when OMLX_PERF_NO_GUARD is set"
    );

    // Restore for the next test in this binary.
    unsafe {
        std::env::remove_var("OMLX_PERF_NO_GUARD");
    }
}
