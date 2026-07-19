//! Tuning records: the immutable evidence that a candidate beats the
//! reference for a given [`crate::KernelKey`].
//!
//! A [`TuningRecord`] is written once by [`crate::BoundedTuner`] and never
//! mutated. The selector consults `expires_at_unix_ms` to drop stale
//! evidence; a record without `expires_at_unix_ms` is treated as
//! long-lived but always revisitable via fresh tuning.

use serde::{Deserialize, Serialize};

use crate::candidate::CandidateId;
use crate::key::KernelKey;

/// A single timed invocation of a candidate for a given key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub sample_idx: u32,
    pub latency_ns: u64,
    /// Optional joules consumed (only meaningful on instrumented devices).
    pub energy_j: Option<f64>,
    pub bytes_written: u64,
}

impl Measurement {
    /// Convenience constructor for tests and synthetic measurements.
    pub fn new(sample_idx: u32, latency_ns: u64) -> Self {
        Self {
            sample_idx,
            latency_ns,
            energy_j: None,
            bytes_written: 0,
        }
    }
}

/// Immutable evidence artifact produced by [`crate::BoundedTuner`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TuningRecord {
    pub candidate_id: CandidateId,
    pub key: KernelKey,
    pub measurements: Vec<Measurement>,
    pub median_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub variance_ns2: u64,
    pub samples: usize,
    pub warmup_discarded: usize,
    pub compiler: String,
    pub compiler_version: String,
    pub captured_at_unix_ms: u64,
    pub source_revision: String,
    pub expires_at_unix_ms: Option<u64>,
}

impl TuningRecord {
    /// `true` when `now_unix_ms >= expires_at_unix_ms`. Records without an
    /// expiry are considered fresh forever.
    pub fn is_stale(&self, now_unix_ms: u64) -> bool {
        match self.expires_at_unix_ms {
            None => false,
            Some(expires) => now_unix_ms >= expires,
        }
    }

    /// Build a synthetic record directly from samples (used by tests and
    /// trusted capture paths). Stats are computed deterministically:
    /// median is the high-middle sample; p95/p99 are `ceil(0.95*N)` /
    /// `ceil(0.99*N)` indices into the sorted sample array.
    pub fn from_samples(
        candidate_id: CandidateId,
        key: KernelKey,
        samples: &[u64],
        warmup_discarded: usize,
        compiler: impl Into<String>,
        compiler_version: impl Into<String>,
        captured_at_unix_ms: u64,
        source_revision: impl Into<String>,
        expires_at_unix_ms: Option<u64>,
    ) -> Self {
        assert!(
            !samples.is_empty(),
            "from_samples requires at least one measurement"
        );
        let mut sorted: Vec<u64> = samples.to_vec();
        sorted.sort_unstable();
        let n = sorted.len();
        let median_idx = n / 2;
        let p95_idx = ((n as f64 * 0.95).ceil() as usize).saturating_sub(1).min(n - 1);
        let p99_idx = ((n as f64 * 0.99).ceil() as usize).saturating_sub(1).min(n - 1);
        let median_ns = sorted[median_idx];
        let p95_ns = sorted[p95_idx];
        let p99_ns = sorted[p99_idx];
        let mean = sorted.iter().sum::<u64>() as f64 / n as f64;
        let variance = sorted
            .iter()
            .map(|&x| ((x as f64) - mean).powi(2))
            .sum::<f64>() / n as f64;
        let measurements: Vec<Measurement> = samples
            .iter()
            .enumerate()
            .map(|(i, &latency)| Measurement::new(i as u32, latency))
            .collect();
        Self {
            candidate_id,
            key,
            measurements,
            median_ns,
            p95_ns,
            p99_ns,
            variance_ns2: variance as u64,
            samples: n,
            warmup_discarded,
            compiler: compiler.into(),
            compiler_version: compiler_version.into(),
            captured_at_unix_ms,
            source_revision: source_revision.into(),
            expires_at_unix_ms,
        }
    }
}