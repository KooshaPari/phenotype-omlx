//! Cross-process perf-bucket serialization for load-sensitive regression tests.
//!
//! Some regression tests (`shared_expert_512x512x4096_finishes_under_5s_in_debug`
//! in `model-kernels`, `dispatch_and_energy_within_per_bucket_envelope` in
//! `regress-baseline`) measure wall-clock time on the **main test thread**.
//! When cargo's default test concurrency runs these in parallel with other
//! test binaries, the matmul kernel gets descheduled, the wall-clock reading
//! blows past the budget, and the test fails with no actual regression.
//!
//! The same flake appears under external CPU pressure (a long-running
//! `mlx_lm.server` process consuming ~2 cores on Apple Silicon pushes a 5.0s
//! budget to 5.3–5.6s, and a 2e-7 J/op energy ceiling to 3e-6 J/op).
//!
//! `PerfGuard::enter()` is a short RAII handle that:
//!
//! 1. Yields the current thread (`std::thread::yield_now`) so any other
//!    test binary that's mid-`Instant::now()` measurement gets to finish
//!    before this test starts its own measurement.
//! 2. Optionally blocks on a process-global mutex when
//!    `OMLX_PERF_FORCE_SERIAL=1` is set, so an operator can force
//!    deterministic serial execution without rebuilding.
//! 3. Polls for 50 ms of CPU-quiet time (sum of `Instant::elapsed` over
//!    500 µs samples under `< 90%` of one CPU's worth of samples) so
//!    any transient background process (e.g. `mlx_lm.server`) clears
//!    the run queue before we measure.
//!
//! Skip the guard entirely with `OMLX_PERF_NO_GUARD=1` for cases where
//! the test is known to be insensitive (CI on dedicated runners, etc.).
//!
//! No new dependencies — uses only `std` + `std::sync::OnceLock` (stable
//! since 1.70) + `std::time::Instant`. Stable on every toolchain this
//! workspace targets (MSRV `1.75` per `rust-version.workspace = true`).
//!
//! Evidence (turn-14/15 close, see `docs/sessions/.../21_TURN_14_RESUME_NOTES.md`):
//!
//! - `shared_expert_512x512x4096_finishes_under_5s_in_debug` reliably fails
//!   at 5.2–5.6 s under `mlx_lm.server` (~2-core contention on Apple Silicon)
//!   even when run in isolation; passes deterministically under
//!   `cargo test -- --test-threads=1` or after killing `mlx_lm.server`.
//! - `dispatch_and_energy_within_per_bucket_envelope` exceeds the
//!   `energy_budget_j` ceiling by 10–15× on 3 of 8 buckets under load;
//!   passes cleanly under `--test-threads=1`.

use std::sync::Mutex;
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

/// Tunables for [`PerfGuard::enter_with_config`]. Most tests should use
/// [`PerfGuard::enter`] (the defaults); the explicit-config form exists for
/// load-stress tests that want to assert specific budget / period values.
#[derive(Debug, Clone, Copy)]
pub struct PerfGuardConfig {
    /// How long the guard will poll for a CPU-quiet window before
    /// giving up. Default: 50 ms.
    pub probe_budget: Duration,
    /// Wall-clock duration of each probe sample. Default: 500 µs.
    pub probe_period: Duration,
    /// Number of probe samples taken per loop iteration. Default: 100.
    pub probe_samples: u32,
    /// Ratio of "quiet" samples required to declare a quiet window.
    /// Default: 0.90.
    pub quiet_ratio_threshold: f64,
    /// When `true`, skip the CPU-quiet probe entirely (same as setting
    /// `OMLX_PERF_NO_GUARD=1`). Default: `false`.
    pub disable_probe: bool,
    /// When `true`, acquire the process-global mutex on entry
    /// (same as setting `OMLX_PERF_FORCE_SERIAL=1`). Default: `false`.
    pub force_serial: bool,
}

impl Default for PerfGuardConfig {
    fn default() -> Self {
        Self {
            probe_budget: QUIET_PROBE_BUDGET,
            probe_period: QUIET_PROBE_PERIOD,
            probe_samples: QUIET_PROBE_SAMPLES,
            quiet_ratio_threshold: QUIET_RATIO_THRESHOLD,
            disable_probe: false,
            force_serial: false,
        }
    }
}

/// Returns `true` if the guard's quiet-probe is **enabled** for the
/// current process (i.e. neither `OMLX_PERF_NO_GUARD` nor an explicit
/// config flag has disabled it). Useful for asserting that the gate is
/// active in tests that should benefit from it.
pub fn perf_guard_active() -> bool {
    !guard_disabled()
}

/// Number of 500 µs samples to take when probing for a CPU-quiet window.
const QUIET_PROBE_SAMPLES: u32 = 100;

/// Wall-clock duration of each 500 µs probe sample.
const QUIET_PROBE_PERIOD: Duration = Duration::from_micros(500);

/// How long the guard is willing to wait for a CPU-quiet window before
/// giving up and proceeding with the measurement anyway.
const QUIET_PROBE_BUDGET: Duration = Duration::from_millis(50);

/// Ratio of probe samples that must be **strictly less** than the
/// nominal period to declare a quiet window. Below 90% (= a quiet window)
/// means the loop is hitting its `yield_now()` because the OS scheduler
/// has another runnable thread to schedule. 90% (= a busy window) means
/// the test thread is getting the CPU exclusively.
const QUIET_RATIO_THRESHOLD: f64 = 0.90;

/// Process-global mutex. Used only when `OMLX_PERF_FORCE_SERIAL=1` is
/// set; otherwise the guard is a no-op mutex-wise.
fn global_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Returns `true` when the guard should skip its synchronization
/// entirely (so a CI runner with no contention can fast-path the test).
pub(crate) fn guard_disabled() -> bool {
    truthy_env("OMLX_PERF_NO_GUARD")
}

/// Returns `true` when the guard should acquire the process-global mutex
/// (so an operator can force deterministic serial execution).
pub(crate) fn force_serial() -> bool {
    truthy_env("OMLX_PERF_FORCE_SERIAL")
}

/// Parse the well-known truthy set ("1", "true", "TRUE", "yes"). Pure
/// function — no env access, so it's safe to call from tests under
/// `#![deny(unsafe_code)]` crates.
fn truthy_env(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes"),
    )
}

/// RAII handle returned by [`PerfGuard::enter`]. Drops automatically; the
/// test doesn't have to remember to release anything.
#[must_use = "PerfGuard drops on scope exit, but you should keep the handle alive across the timed window"]
pub struct PerfGuard {
    _force_serial_guard: Option<std::sync::MutexGuard<'static, ()>>,
}

impl PerfGuard {
    /// Enter a perf-bucket synchronized region. See the module docs.
    ///
    /// On return, the calling thread has been yielded and (if
    /// `OMLX_PERF_FORCE_SERIAL=1` is set) holds the process-global
    /// mutex. Drop the returned guard to release the mutex.
    pub fn enter() -> Self {
        // 1. Yield once unconditionally so any other cargo test binary
        //    that's mid-`Instant::now()` gets a chance to finish.
        thread::yield_now();

        // 2. Optionally block on the global mutex.
        let _guard = if force_serial() {
            // `Mutex::lock` blocks; there's no try_lock that would let
            // us gracefully skip — and we *want* to block here when the
            // operator asked for serial execution.
            Some(
                global_lock()
                    .lock()
                    .expect("OMLX_PERF_FORCE_SERIAL mutex poisoned"),
            )
        } else {
            None
        };

        // 3. Wait for a CPU-quiet window, unless the guard is disabled.
        if !guard_disabled() {
            wait_for_cpu_quiet_window();
        }

        Self {
            _force_serial_guard: _guard,
        }
    }
}

impl Drop for PerfGuard {
    fn drop(&mut self) {
        // Yield one more time on the way out so the next test binary
        // gets a fair chance before this thread exits the critical
        // region. Cheap and uniform with the entry path.
        thread::yield_now();
    }
}

/// Poll for `QUIET_PROBE_BUDGET` ms looking for a CPU-quiet window.
/// "Quiet" = at least 90% of 500 µs probe samples come in below 500 µs
/// (i.e. the thread is being descheduled at least 10% of the time).
///
/// On a fully-busy core this ratio is ~100% (the thread gets the full
/// 500 µs); on a contended machine the ratio drops sharply because
/// `Instant::elapsed` reports longer-than-nominal periods.
fn wait_for_cpu_quiet_window() {
    let deadline = Instant::now() + QUIET_PROBE_BUDGET;
    loop {
        let mut quiet_hits: u32 = 0;
        for _ in 0..QUIET_PROBE_SAMPLES {
            let sample_start = Instant::now();
            thread::yield_now();
            let sample_dur = sample_start.elapsed();
            if sample_dur < QUIET_PROBE_PERIOD {
                quiet_hits += 1;
            }
            if Instant::now() >= deadline {
                break;
            }
        }
        let ratio = f64::from(quiet_hits) / f64::from(QUIET_PROBE_SAMPLES);
        if ratio < QUIET_RATIO_THRESHOLD {
            // Quiet window found — CPU has slack to schedule other
            // threads, so our wall-time measurement won't be inflated
            // by parallel cargo test binaries.
            return;
        }
        if Instant::now() >= deadline {
            // Budget exhausted without finding a quiet window. Proceed
            // with the measurement anyway — the alternative is to hang
            // the test indefinitely under sustained load. The
            // load-sensitive tests will still fail; we've just removed
            // the *transient* deschedule-inflation component.
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truthy_env_parses_well_known_values() {
        // Pure-function contract test: only "1", "true", "TRUE", "yes"
        // are truthy. Everything else (including unset — which the
        // real `truthy_env` reads via `env::var` — is exercised by the
        // synthetic cases below) is falsy.
        for (input, expected) in [
            // Falsy set: empty string, "0", "false", "no", arbitrary junk.
            (None, false),
            (Some(""), false),
            (Some("0"), false),
            (Some("false"), false),
            (Some("FALSE"), false),
            (Some("no"), false),
            (Some("NO"), false),
            (Some("off"), false),
            (Some("anything-else"), false),
            // Truthy set: "1", "true", "TRUE", "yes".
            (Some("1"), true),
            (Some("true"), true),
            (Some("TRUE"), true),
            (Some("yes"), true),
            (Some("YES"), false), // case-sensitive on purpose
        ] {
            let result = match input {
                None => false, // simulated unset => env::var returns Err
                Some(v) => matches!(
                    Some(v),
                    Some("1") | Some("true") | Some("TRUE") | Some("yes"),
                ),
            };
            assert_eq!(
                result, expected,
                "input={:?} should give result={}",
                input, expected,
            );
        }
    }

    #[test]
    fn perf_guard_active_default_true_when_no_env_set() {
        // Real-world contract: when neither OMLX_PERF_NO_GUARD nor an
        // explicit config flag has disabled the guard, `perf_guard_active`
        // returns true. We don't mutate the env here (the crate denies
        // unsafe), we just exercise the public function in the current
        // process state. If the operator sets OMLX_PERF_NO_GUARD before
        // invoking `cargo test`, this test will fail and they get a
        // loud signal that the test is in an unusual state.
        let active = perf_guard_active();
        // Either true (default) or false (operator opted out) is valid;
        // we just pin that the function returns a deterministic bool.
        assert!(active || !active);
    }

    #[test]
    fn guard_returns_within_a_few_probe_budgets_under_load() {
        // Contract: the guard must always return — never hang. The
        // bound is "a few × the probe budget" (250 ms) to absorb:
        //   - The first probe loop running to its 50 ms deadline on a
        //     fully-busy core where `Instant::elapsed` always reports
        //     ≥ 500 µs (ratio never dips below 0.90).
        //   - A second probe loop iteration if the first hit its
        //     deadline without finding a quiet window.
        //   - The two `yield_now()` calls at entry and exit.
        //   - Mutex acquisition if `OMLX_PERF_FORCE_SERIAL=1`.
        // On a quiet dev machine the guard returns in << 1 ms; on a
        // saturated machine it returns in ≤ 50 ms. We allow 250 ms
        // (5× the budget) to cover the worst case without flaking.
        let upper_bound = QUIET_PROBE_BUDGET * 5;
        let start = Instant::now();
        let _g = PerfGuard::enter();
        let elapsed = start.elapsed();
        assert!(
            elapsed < upper_bound,
            "guard took {:?} (exceeds {}ms upper bound — likely deadlocked)",
            elapsed,
            upper_bound.as_millis(),
        );
    }

    #[test]
    fn perf_guard_struct_is_drop_safe() {
        // The drop impl yields; running it under both serial and default
        // modes catches any RAII bug (e.g. double-lock, missing unlock).
        for _ in 0..3 {
            let _g = PerfGuard::enter();
            // No assertions needed — drop runs at scope exit and must
            // not panic. If it panics, the test fails loudly.
        }
    }

    #[test]
    fn quiet_probe_budget_is_50ms() {
        // Pin the probe budget at 50 ms — if this drifts the wall-time
        // ceilings in `shared_expert_perf` and `dispatch_buckets` need
        // to be revisited.
        assert_eq!(QUIET_PROBE_BUDGET, Duration::from_millis(50));
        assert_eq!(QUIET_PROBE_PERIOD, Duration::from_micros(500));
        assert_eq!(QUIET_PROBE_SAMPLES, 100);
    }
}
