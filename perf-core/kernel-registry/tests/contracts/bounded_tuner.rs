//! `BoundedTuner` contract tests: warmup, samples, budget exceeded.

use kernel_registry::compat::OperatorKind;
use kernel_registry::tuner::TunerError;
use kernel_registry::{BackendKind, BoundedTuner};

use super::{candidate_from, key_with, measurement, TEST_DEVICE_FINGERPRINT};

#[test]
fn bounded_tuner_warmup_samples_and_emits_record() {
    let key = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);
    let cand = candidate_from("warm-tester", BackendKind::Cpu, vec![]);
    let tuner = BoundedTuner::new(
        /*warmup*/ 3, /*samples*/ 10, /*max_time_ms*/ 10_000,
    );

    let mut invocation_count: u32 = 0;
    let record = tuner
        .run(key.clone(), cand.clone(), |_k| {
            invocation_count += 1;
            // Latency rises with sample index to exercise ordering.
            let lat = (invocation_count as u64) * 100;
            measurement(invocation_count - 1, lat)
        })
        .expect("tuner should succeed within budget");

    assert_eq!(record.candidate_id, cand.id);
    assert_eq!(record.warmup_discarded, 3);
    // The closure is called warmup + samples times.
    assert_eq!(invocation_count, (3 + 10) as u32);
    assert_eq!(record.samples, 10);
    assert_eq!(
        record.measurements.len(),
        13,
        "all invocations (warmup + samples) are recorded for provenance"
    );
    assert!(record.median_ns > 0);
    assert!(record.p95_ns >= record.median_ns);
    assert!(record.p99_ns >= record.p95_ns);
}

#[test]
fn bounded_tuner_returns_budget_exceeded_when_measurements_exceed_limit() {
    let key = key_with(OperatorKind::DenseMatmul, TEST_DEVICE_FINGERPRINT, 1);
    let cand = candidate_from("budget-breaker", BackendKind::Cpu, vec![]);
    // max_time_ms = 0 forces any measurement work to overflow the budget.
    let tuner = BoundedTuner::new(
        /*warmup*/ 1, /*samples*/ 2, /*max_time_ms*/ 0,
    );

    let mut count = 0u32;
    let result = tuner.run(key, cand, |_k| {
        count += 1;
        // Sleep a measurable amount so budget accounting can flag it.
        std::thread::sleep(std::time::Duration::from_millis(2));
        measurement(count - 1, 2_000_000)
    });

    match result {
        Err(TunerError::BudgetExceeded { used_ms, max_ms }) => {
            assert_eq!(max_ms, 0);
            // We may have stopped after warmup or after first sample; in either
            // case the tuner reports the budget violation.
            assert!(
                used_ms >= max_ms,
                "used_ms {used_ms} should be >= max_ms {max_ms}"
            );
        }
        Ok(rec) => panic!(
            "expected BudgetExceeded, got record with {} samples",
            rec.samples
        ),
    }
}
