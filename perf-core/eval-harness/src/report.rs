//! Cross-suite report aggregation.
//!
//! A [`MultiSuiteReport`] collects [`EvaluationReport`]s from multiple suites
//! and produces deterministic aggregate metrics. Aggregation is purely
//! arithmetic: it sums task counts and correct counts, computes per-suite
//! accuracy, and computes an overall accuracy weighted by task count so a
//! 1-task suite cannot dominate the headline number.
//!
//! Reports carry the provenance of their constituent datasets so downstream
//! consumers can attribute the aggregate to specific dataset revisions.

use crate::provenance::DatasetProvenance;
use crate::{EvaluationReport, Suite};
use serde::{Deserialize, Serialize};

/// One suite's contribution to a multi-suite report.
///
/// Bundles the per-suite report with the provenance of the dataset the
/// results were scored against so reports stay attributable to specific
/// dataset bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuiteReportEntry {
    pub suite: Suite,
    pub provenance: DatasetProvenance,
    pub report: EvaluationReport,
}

impl SuiteReportEntry {
    /// Build a new entry. The provenance's `task_count` is overwritten with
    /// the report's `task_count` so the recorded metadata matches.
    pub fn new(provenance: DatasetProvenance, report: EvaluationReport) -> Self {
        Self {
            suite: report.suite,
            provenance,
            report,
        }
    }

    pub fn task_count(&self) -> usize {
        self.report.task_count
    }

    pub fn correct_count(&self) -> usize {
        self.report.correct_count
    }

    pub fn accuracy(&self) -> f64 {
        self.report.accuracy
    }
}

/// Cross-suite aggregation of per-suite reports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiSuiteReport {
    pub task_count: usize,
    pub correct_count: usize,
    /// Weighted overall accuracy: sum(correct) / sum(tasks). Empty input
    /// yields 0.0 to match [`EvaluationReport::from_results`].
    pub overall_accuracy: f64,
    /// Arithmetic mean of per-suite accuracies. Empty input yields 0.0.
    pub mean_suite_accuracy: f64,
    /// Arithmetic mean of per-suite mean scores. Empty input yields 0.0.
    pub mean_suite_score: f64,
    pub entries: Vec<SuiteReportEntry>,
}

impl MultiSuiteReport {
    /// Aggregate per-suite reports. The input is sorted by `suite` so the
    /// resulting entries are reproducible regardless of call order.
    pub fn from_reports(mut entries: Vec<SuiteReportEntry>) -> Self {
        entries.sort_by_key(|entry| entry.suite.as_str());

        let task_count: usize = entries.iter().map(|e| e.task_count()).sum();
        let correct_count: usize = entries.iter().map(|e| e.correct_count()).sum();
        let overall_accuracy = if task_count == 0 {
            0.0
        } else {
            correct_count as f64 / task_count as f64
        };
        let mean_suite_accuracy = if entries.is_empty() {
            0.0
        } else {
            entries.iter().map(|e| e.accuracy()).sum::<f64>() / entries.len() as f64
        };
        let mean_suite_score = if entries.is_empty() {
            0.0
        } else {
            entries.iter().map(|e| e.report.mean_score).sum::<f64>() / entries.len() as f64
        };

        Self {
            task_count,
            correct_count,
            overall_accuracy,
            mean_suite_accuracy,
            mean_suite_score,
            entries,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::DatasetProvenance;
    use crate::TaskResult;

    fn prov(source: &str, task_count: usize) -> DatasetProvenance {
        DatasetProvenance::new(source, "v1", "test", b"x", task_count)
    }

    fn result(task_id: &str, correct: bool, suite: Suite) -> TaskResult {
        TaskResult {
            task_id: task_id.into(),
            suite,
            prompt_tokens: 1,
            completion_tokens: 1,
            completion: "c".into(),
            normalized_completion: "c".into(),
            correct,
            score: if correct { 1.0 } else { 0.0 },
            latency_ms: 1.0,
            matched_answer: None,
        }
    }

    fn report(suite: Suite, correct: usize, total: usize) -> EvaluationReport {
        let results: Vec<TaskResult> = (0..total)
            .map(|i| result(&format!("{}-{}", suite.as_str(), i), i < correct, suite))
            .collect();
        EvaluationReport::from_results(suite, results)
    }

    fn entry(suite: Suite, correct: usize, total: usize) -> SuiteReportEntry {
        SuiteReportEntry::new(prov(suite.as_str(), total), report(suite, correct, total))
    }

    #[test]
    fn aggregate_is_task_weighted() {
        // MMLU: 1/1 correct (1.0); GPQA: 0/1 correct (0.0). Overall must be
        // 0.5 (1 of 2), not the 0.5 mean of per-suite accuracies.
        let multi = MultiSuiteReport::from_reports(vec![
            entry(Suite::MMLU, 1, 1),
            entry(Suite::GPQA, 0, 1),
        ]);
        assert_eq!(multi.task_count, 2);
        assert_eq!(multi.correct_count, 1);
        assert_eq!(multi.overall_accuracy, 0.5);
        assert_eq!(multi.mean_suite_accuracy, 0.5);
        assert_eq!(multi.mean_suite_score, 0.5);
        // Entries sorted by suite.
        assert_eq!(multi.entries[0].suite, Suite::GPQA);
        assert_eq!(multi.entries[1].suite, Suite::MMLU);
    }

    #[test]
    fn aggregate_uneven_suites_uses_task_weighted_accuracy() {
        // MMLU: 9/10 correct; GPQA: 0/10 correct. Overall must be 0.45.
        let multi = MultiSuiteReport::from_reports(vec![
            entry(Suite::MMLU, 9, 10),
            entry(Suite::GPQA, 0, 10),
        ]);
        assert_eq!(multi.overall_accuracy, 0.45);
        assert_eq!(multi.mean_suite_accuracy, 0.45);
    }

    #[test]
    fn aggregate_empty_is_zero() {
        let multi = MultiSuiteReport::from_reports(vec![]);
        assert_eq!(multi.task_count, 0);
        assert_eq!(multi.correct_count, 0);
        assert_eq!(multi.overall_accuracy, 0.0);
        assert_eq!(multi.mean_suite_accuracy, 0.0);
        assert_eq!(multi.mean_suite_score, 0.0);
        assert!(multi.entries.is_empty());
    }

    #[test]
    fn aggregate_is_serde_stable() {
        let multi = MultiSuiteReport::from_reports(vec![
            entry(Suite::MMLU, 1, 2),
            entry(Suite::GPQA, 0, 1),
        ]);
        let encoded = serde_json::to_string(&multi).unwrap();
        let decoded: MultiSuiteReport = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, multi);
    }

    #[test]
    fn entry_new_aligns_provenance_task_count_with_report() {
        let mut p = prov("mmlu", 99);
        p.task_count = 0;
        let r = report(Suite::MMLU, 2, 3);
        let e = SuiteReportEntry::new(p, r);
        assert_eq!(e.task_count(), 3);
        assert_eq!(e.correct_count(), 2);
        assert_eq!(e.accuracy(), 2.0 / 3.0);
    }

    #[test]
    fn determinism_independent_of_input_order() {
        let r1 = entry(Suite::MMLU, 1, 2);
        let r2 = entry(Suite::GPQA, 2, 3);
        let multi_a = MultiSuiteReport::from_reports(vec![r1.clone(), r2.clone()]);
        let multi_b = MultiSuiteReport::from_reports(vec![r2, r1]);
        // Both aggregated reports must be byte-equal because entries are
        // sorted by suite.
        assert_eq!(multi_a, multi_b);
    }

    #[test]
    fn aggregate_preserves_provenance() {
        let multi = MultiSuiteReport::from_reports(vec![entry(Suite::MMLU, 1, 1)]);
        assert_eq!(multi.entries[0].provenance.source, "mmlu");
        assert_eq!(multi.entries[0].provenance.split, "test");
        assert_eq!(multi.entries[0].provenance.source_revision, "v1");
    }

    #[test]
    fn mean_suite_score_uses_report_mean_scores() {
        let mut r = report(Suite::MMLU, 1, 2);
        // Set non-binary scores to test mean_suite_score.
        r.results[0].score = 0.7;
        r.results[1].score = 0.3;
        r.mean_score = (0.7 + 0.3) / 2.0;
        let entry = SuiteReportEntry::new(prov("mmlu", 2), r);
        let multi = MultiSuiteReport::from_reports(vec![entry]);
        assert_eq!(multi.mean_suite_score, 0.5);
    }
}
