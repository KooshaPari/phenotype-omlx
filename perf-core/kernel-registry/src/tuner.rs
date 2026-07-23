//! Bounded tuning with warmup, sample budget, and time-budget enforcement.
//!
//! [`BoundedTuner`] is the only sanctioned path for producing a
//! [`TuningRecord`]. It enforces:
//!
//! - `warmup`: measurements discarded before sample collection starts;
//! - `samples`: the minimum number of recorded measurements;
//! - `max_time_ms`: a wall-clock budget that is checked *between* samples.
//!
//! Budget overflow returns [`TunerError::BudgetExceeded`] rather than
//! panicking; the caller is expected to record a partial trace and fall
//! back to the reference kernel.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::candidate::Candidate;
use crate::key::KernelKey;
use crate::record::{Measurement, TuningRecord};

/// Tuner configuration: warmup + sample counts plus a wall-clock budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundedTuner {
    pub warmup: usize,
    pub samples: usize,
    pub max_time_ms: u64,
}

/// Errors raised by [`BoundedTuner::run`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
pub enum TunerError {
    #[error("tuning budget exceeded: {used_ms}ms > {max_ms}ms")]
    BudgetExceeded { used_ms: u64, max_ms: u64 },
}

impl BoundedTuner {
    /// Build a tuner. `warmup` and `samples` are the per-record counts;
    /// `max_time_ms` is the upper bound on total wall-clock spent in
    /// `measure` calls.
    pub fn new(warmup: usize, samples: usize, max_time_ms: u64) -> Self {
        Self {
            warmup,
            samples,
            max_time_ms,
        }
    }

    /// Run the tuner. `measure` is called `warmup + samples` times; each
    /// invocation receives the current `key` and returns a [`Measurement`].
    ///
    /// The total wall-clock time (summed across `measure` calls) is
    /// tracked. As soon as it exceeds `max_time_ms` the tuner returns
    /// [`TunerError::BudgetExceeded`] with `used_ms` and `max_ms` set so
    /// callers can decide whether to retry with a larger budget.
    pub fn run<F>(
        &self,
        key: KernelKey,
        candidate: Candidate,
        mut measure: F,
    ) -> Result<TuningRecord, TunerError>
    where
        F: FnMut(&KernelKey) -> Measurement,
    {
        assert!(self.samples > 0, "BoundedTuner requires samples >= 1");
        let started = Instant::now();
        let mut all: Vec<Measurement> = Vec::with_capacity(self.warmup + self.samples);

        // Warmup phase.
        for _ in 0..self.warmup {
            if elapsed_ms(started) > self.max_time_ms {
                return Err(TunerError::BudgetExceeded {
                    used_ms: elapsed_ms(started),
                    max_ms: self.max_time_ms,
                });
            }
            let m = measure(&key);
            all.push(m);
        }

        // Sample phase.
        for i in 0..self.samples {
            if elapsed_ms(started) > self.max_time_ms {
                return Err(TunerError::BudgetExceeded {
                    used_ms: elapsed_ms(started),
                    max_ms: self.max_time_ms,
                });
            }
            let m = measure(&key);
            // Re-stamp the sample_idx so warmup samples stay 0..warmup and
            // sample samples take the next slot. We *don't* overwrite the
            // caller's sample_idx because callers may want provenance; the
            // selector only consults stats, not indices.
            let _ = i;
            all.push(m);
        }

        // Build the record. Stats are computed over the *post-warmup*
        // samples so the warmup phase cannot poison median / p95.
        let samples_vec: Vec<u64> = all.iter().skip(self.warmup).map(|m| m.latency_ns).collect();
        let record = TuningRecord::from_samples(
            candidate.id,
            key,
            &samples_vec,
            self.warmup,
            /* compiler */ "unknown",
            /* compiler_version */ "0.0.0",
            /* captured_at_unix_ms */ 0,
            /* source_revision */ "",
            /* expires_at_unix_ms */ None,
        );
        Ok(record_with_measurements(record, all))
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis() as u64
}

/// Re-attach the full measurement vector (warmup + samples) to the
/// summary record. We can't do this inside `TuningRecord::from_samples`
/// because that builder only accepts latencies.
fn record_with_measurements(
    mut record: TuningRecord,
    measurements: Vec<Measurement>,
) -> TuningRecord {
    record.measurements = measurements;
    record
}
