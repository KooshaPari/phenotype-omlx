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
use crate::quality::QualityAttachment;

/// A single timed invocation of a candidate for a given key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub sample_idx: u32,
    pub latency_ns: u64,
    /// Optional joules consumed (only meaningful on instrumented devices).
    pub energy_j: Option<f64>,
    /// Optional dispatch count for this sample (cumulative kernel launches
    /// attributed to a single timed invocation). Populated by instrumentation
    /// when the runtime can count it; `None` means "not measured".
    pub dispatches: Option<u32>,
    pub bytes_written: u64,
}

impl Measurement {
    /// Convenience constructor for tests and synthetic measurements.
    pub fn new(sample_idx: u32, latency_ns: u64) -> Self {
        Self {
            sample_idx,
            latency_ns,
            energy_j: None,
            dispatches: None,
            bytes_written: 0,
        }
    }

    /// Constructor with energy and dispatch metadata, used by instrumented
    /// tuners. `energy_j == None` or `dispatches == None` is treated as
    /// "not measured" — selectors requiring those metrics will treat the
    /// candidate as not-yet-tunable on that axis.
    pub fn with_metadata(
        sample_idx: u32,
        latency_ns: u64,
        energy_j: Option<f64>,
        dispatches: Option<u32>,
        bytes_written: u64,
    ) -> Self {
        Self {
            sample_idx,
            latency_ns,
            energy_j,
            dispatches,
            bytes_written,
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
    /// Median joules per invocation across the post-warmup samples. `None`
    /// when no sample carried an `energy_j` measurement.
    pub median_energy_j: Option<f64>,
    /// Median dispatches per invocation across the post-warmup samples.
    /// `None` when no sample carried a `dispatches` measurement.
    pub median_dispatches: Option<u32>,
    pub samples: usize,
    pub warmup_discarded: usize,
    pub compiler: String,
    pub compiler_version: String,
    pub captured_at_unix_ms: u64,
    pub source_revision: String,
    pub expires_at_unix_ms: Option<u64>,
    /// Optional [`QualityAttachment`] describing the gate(s) the
    /// production policy must hold and the evidence rows that satisfy
    /// them. `None` means "no quality requirement"; gating policy will
    /// reject unattached candidates with `MissingQualityEvidence`.
    #[serde(default)]
    pub quality: Option<QualityAttachment>,
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
    /// `ceil(0.99*N)` indices into the sorted sample array. `median_energy_j`
    /// and `median_dispatches` are derived from the post-warmup samples
    /// and are `None` when no sample reported that metric.
    #[allow(clippy::too_many_arguments)]
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
            .sum::<f64>()
            / n as f64;
        let measurements: Vec<Measurement> = samples
            .iter()
            .enumerate()
            .map(|(i, &latency)| Measurement::new(i as u32, latency))
            .collect();
        let (median_energy_j, median_dispatches) = median_metadata(&measurements);
        Self {
            candidate_id,
            key,
            measurements,
            median_ns,
            p95_ns,
            p99_ns,
            variance_ns2: variance as u64,
            median_energy_j,
            median_dispatches,
            samples: n,
            warmup_discarded,
            compiler: compiler.into(),
            compiler_version: compiler_version.into(),
            captured_at_unix_ms,
            source_revision: source_revision.into(),
            expires_at_unix_ms,
            quality: None,
        }
    }

    /// Builder for instrumented records that carry per-sample energy and
    /// dispatch counts. The same percentile/median rules as
    /// [`TuningRecord::from_samples`] apply; per-sample metadata is reduced
    /// to median values so the selector never has to walk the sample array.
    pub fn from_measurements(
        candidate_id: CandidateId,
        key: KernelKey,
        measurements: Vec<Measurement>,
        compiler: impl Into<String>,
        compiler_version: impl Into<String>,
        captured_at_unix_ms: u64,
        source_revision: impl Into<String>,
        expires_at_unix_ms: Option<u64>,
    ) -> Self {
        assert!(
            !measurements.is_empty(),
            "from_measurements requires at least one measurement"
        );
        let mut latencies: Vec<u64> = measurements.iter().map(|m| m.latency_ns).collect();
        latencies.sort_unstable();
        let n = latencies.len();
        let median_idx = n / 2;
        let p95_idx = ((n as f64 * 0.95).ceil() as usize).saturating_sub(1).min(n - 1);
        let p99_idx = ((n as f64 * 0.99).ceil() as usize).saturating_sub(1).min(n - 1);
        let median_ns = latencies[median_idx];
        let p95_ns = latencies[p95_idx];
        let p99_ns = latencies[p99_idx];
        let mean = latencies.iter().sum::<u64>() as f64 / n as f64;
        let variance = latencies
            .iter()
            .map(|&x| ((x as f64) - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let (median_energy_j, median_dispatches) = median_metadata(&measurements);
        Self {
            candidate_id,
            key,
            measurements,
            median_ns,
            p95_ns,
            p99_ns,
            variance_ns2: variance as u64,
            median_energy_j,
            median_dispatches,
            samples: n,
            warmup_discarded: 0,
            compiler: compiler.into(),
            compiler_version: compiler_version.into(),
            captured_at_unix_ms,
            source_revision: source_revision.into(),
            expires_at_unix_ms,
            quality: None,
        }
    }
}
/// `(None, None)` when no sample carried the metric. Energy is averaged in
/// floating-point before the high-middle sample is taken; dispatches are
/// rounded up to the nearest integer because a partial dispatch is not a
/// valid measurement.
fn median_metadata(measurements: &[Measurement]) -> (Option<f64>, Option<u32>) {
    let mut energies: Vec<f64> = measurements.iter().filter_map(|m| m.energy_j).collect();
    let mut dispatches: Vec<u32> = measurements.iter().filter_map(|m| m.dispatches).collect();

    let median_energy = if energies.is_empty() {
        None
    } else {
        energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Some(energies[energies.len() / 2])
    };
    let median_dispatches = if dispatches.is_empty() {
        None
    } else {
        dispatches.sort_unstable();
        Some(dispatches[dispatches.len() / 2])
    };
    (median_energy, median_dispatches)
}