//! PyO3 type wrappers for report and provenance data structures.

use pyo3::prelude::*;

use crate::provenance::DatasetProvenance;
use crate::report::{MultiSuiteReport, SuiteReportEntry};
use crate::EvaluationReport;

use super::types::{PySuite, PyTaskResult};

// ── EvaluationReport ───────────────────────────────────────────────────────

/// Per-suite aggregation of task results.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub(super) struct PyEvaluationReport {
    #[pyo3(get)]
    pub(super) suite: PySuite,
    #[pyo3(get)]
    pub(super) task_count: usize,
    #[pyo3(get)]
    pub(super) correct_count: usize,
    #[pyo3(get)]
    pub(super) accuracy: f64,
    #[pyo3(get)]
    pub(super) mean_score: f64,
    #[pyo3(get)]
    pub(super) results: Vec<PyTaskResult>,
}

#[pymethods]
impl PyEvaluationReport {
    #[staticmethod]
    fn from_results(suite: PySuite, results: Vec<PyTaskResult>) -> Self {
        Self::build_from_results(suite, results)
    }

    fn to_json(&self) -> PyResult<String> {
        self.to_json_str()
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        Self::parse_json(json)
    }

    fn __repr__(&self) -> String {
        format!(
            "EvaluationReport(suite={}, accuracy={:.4}, mean_score={:.4}, tasks={})",
            self.suite.as_str(),
            self.accuracy,
            self.mean_score,
            self.task_count
        )
    }
}

impl PyEvaluationReport {
    pub(super) fn build_from_results(suite: PySuite, results: Vec<PyTaskResult>) -> Self {
        let rust_results: Vec<crate::TaskResult> =
            results.into_iter().map(|r| r.into_rust()).collect();
        let report = EvaluationReport::from_results(suite.into(), rust_results);
        Self::from_rust(report)
    }

    pub(super) fn to_json_str(&self) -> PyResult<String> {
        let report = self.to_rust();
        serde_json::to_string_pretty(&report)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    pub(super) fn parse_json(json: &str) -> PyResult<Self> {
        let report: EvaluationReport = serde_json::from_str(json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self::from_rust(report))
    }

    pub(super) fn from_rust(r: EvaluationReport) -> Self {
        Self {
            suite: r.suite.into(),
            task_count: r.task_count,
            correct_count: r.correct_count,
            accuracy: r.accuracy,
            mean_score: r.mean_score,
            results: r.results.into_iter().map(PyTaskResult::from).collect(),
        }
    }

    pub(super) fn to_rust(&self) -> EvaluationReport {
        EvaluationReport {
            suite: self.suite.into(),
            task_count: self.task_count,
            correct_count: self.correct_count,
            accuracy: self.accuracy,
            mean_score: self.mean_score,
            results: self
                .results
                .iter()
                .cloned()
                .map(|r| r.into_rust())
                .collect(),
        }
    }
}

// ── DatasetProvenance ──────────────────────────────────────────────────────

/// Provenance metadata for an evaluation dataset.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub(super) struct PyDatasetProvenance {
    #[pyo3(get)]
    source: String,
    #[pyo3(get)]
    source_revision: String,
    #[pyo3(get)]
    split: String,
    #[pyo3(get)]
    sha256: String,
    #[pyo3(get)]
    task_count: usize,
}

#[pymethods]
impl PyDatasetProvenance {
    #[new]
    #[pyo3(signature = (source, source_revision, split, data_bytes, task_count))]
    fn new(
        source: String,
        source_revision: String,
        split: String,
        data_bytes: &[u8],
        task_count: usize,
    ) -> Self {
        let prov = DatasetProvenance::new(
            source.clone(),
            &source_revision,
            &split,
            data_bytes,
            task_count,
        );
        Self {
            source,
            source_revision,
            split,
            sha256: prov.sha256,
            task_count,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetProvenance(source={}, rev={}, split={}, tasks={})",
            self.source, self.source_revision, self.split, self.task_count
        )
    }
}

impl From<DatasetProvenance> for PyDatasetProvenance {
    fn from(p: DatasetProvenance) -> Self {
        Self {
            source: p.source,
            source_revision: p.source_revision,
            split: p.split,
            sha256: p.sha256,
            task_count: p.task_count,
        }
    }
}

// ── SuiteReportEntry ───────────────────────────────────────────────────────

/// One suite's contribution to a multi-suite report.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub(super) struct PySuiteReportEntry {
    #[pyo3(get)]
    suite: PySuite,
    #[pyo3(get)]
    provenance: PyDatasetProvenance,
    #[pyo3(get)]
    report: PyEvaluationReport,
}

#[pymethods]
impl PySuiteReportEntry {
    #[new]
    fn new(provenance: PyDatasetProvenance, report: PyEvaluationReport) -> Self {
        let suite = report.suite;
        Self {
            suite,
            provenance,
            report,
        }
    }

    fn task_count(&self) -> usize {
        self.report.task_count
    }

    fn correct_count(&self) -> usize {
        self.report.correct_count
    }

    fn accuracy(&self) -> f64 {
        self.report.accuracy
    }
}

// ── MultiSuiteReport ───────────────────────────────────────────────────────

/// Cross-suite aggregation of per-suite reports.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub(super) struct PyMultiSuiteReport {
    #[pyo3(get)]
    task_count: usize,
    #[pyo3(get)]
    correct_count: usize,
    #[pyo3(get)]
    overall_accuracy: f64,
    #[pyo3(get)]
    mean_suite_accuracy: f64,
    #[pyo3(get)]
    mean_suite_score: f64,
    #[pyo3(get)]
    entries: Vec<PySuiteReportEntry>,
}

#[pymethods]
impl PyMultiSuiteReport {
    #[staticmethod]
    fn from_reports(entries: Vec<PySuiteReportEntry>) -> Self {
        let rust_entries: Vec<SuiteReportEntry> = entries
            .into_iter()
            .map(|e| {
                let report = e.report.to_rust();
                let prov = DatasetProvenance::new(
                    &e.provenance.source,
                    &e.provenance.source_revision,
                    &e.provenance.split,
                    b"",
                    e.provenance.task_count,
                );
                SuiteReportEntry::new(prov, report)
            })
            .collect();
        let multi = MultiSuiteReport::from_reports(rust_entries);
        Self::from_rust(multi)
    }

    fn to_json(&self) -> PyResult<String> {
        let multi = self.to_rust();
        serde_json::to_string_pretty(&multi)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let multi: MultiSuiteReport = serde_json::from_str(json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self::from_rust(multi))
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiSuiteReport(overall_accuracy={:.4}, tasks={})",
            self.overall_accuracy, self.task_count
        )
    }
}

impl PyMultiSuiteReport {
    fn from_rust(m: MultiSuiteReport) -> Self {
        Self {
            task_count: m.task_count,
            correct_count: m.correct_count,
            overall_accuracy: m.overall_accuracy,
            mean_suite_accuracy: m.mean_suite_accuracy,
            mean_suite_score: m.mean_suite_score,
            entries: m
                .entries
                .into_iter()
                .map(|e| PySuiteReportEntry {
                    suite: e.suite.into(),
                    provenance: e.provenance.into(),
                    report: PyEvaluationReport::from_rust(e.report),
                })
                .collect(),
        }
    }

    fn to_rust(&self) -> MultiSuiteReport {
        let entries: Vec<SuiteReportEntry> = self
            .entries
            .iter()
            .map(|e| {
                let report = e.report.to_rust();
                let prov = DatasetProvenance::new(
                    &e.provenance.source,
                    &e.provenance.source_revision,
                    &e.provenance.split,
                    b"",
                    e.provenance.task_count,
                );
                SuiteReportEntry::new(prov, report)
            })
            .collect();
        MultiSuiteReport::from_reports(entries)
    }
}
