use crate::report::{MultiSuiteReport, SuiteReportEntry};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{PyDatasetProvenance, PyTaskResult};

/// Evaluation report containing aggregated per-suite task results.
#[pyclass(name = "EvaluationReport", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyEvaluationReport {
    pub inner: crate::EvaluationReport,
}

#[pymethods]
impl PyEvaluationReport {
    #[staticmethod]
    pub fn from_json(json_str: &str) -> PyResult<Self> {
        let report: crate::EvaluationReport = serde_json::from_str(json_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid EvaluationReport JSON: {e}")))?;
        Ok(Self { inner: report })
    }

    #[getter]
    pub fn suite(&self) -> String {
        self.inner.suite.as_str().to_string()
    }

    #[getter]
    pub fn task_count(&self) -> usize {
        self.inner.task_count
    }

    #[getter]
    pub fn correct_count(&self) -> usize {
        self.inner.correct_count
    }

    #[getter]
    pub fn accuracy(&self) -> f64 {
        self.inner.accuracy
    }

    #[getter]
    pub fn mean_score(&self) -> f64 {
        self.inner.mean_score
    }

    #[getter]
    pub fn results(&self) -> Vec<PyTaskResult> {
        self.inner
            .results
            .iter()
            .cloned()
            .map(|r| PyTaskResult { inner: r })
            .collect()
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {e}")))
    }

    pub fn to_json_pretty(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {e}")))
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let json_str = self.to_json()?;
        let json_mod = py.import("json")?;
        let dict = json_mod.call_method1("loads", (json_str,))?;
        dict.cast_into::<PyDict>()
            .map_err(|e| PyValueError::new_err(format!("Dict conversion failed: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "EvaluationReport(suite={}, tasks={}, accuracy={:.4})",
            self.inner.suite.as_str(),
            self.inner.task_count,
            self.inner.accuracy
        )
    }

    fn __str__(&self) -> PyResult<String> {
        self.to_json()
    }
}

/// Cross-suite aggregate evaluation report.
#[pyclass(name = "MultiSuiteReport", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMultiSuiteReport {
    pub inner: MultiSuiteReport,
}

#[pymethods]
impl PyMultiSuiteReport {
    #[staticmethod]
    pub fn from_reports(
        reports: Vec<PyEvaluationReport>,
        provenances: Vec<PyDatasetProvenance>,
    ) -> PyResult<Self> {
        if reports.len() != provenances.len() {
            return Err(PyValueError::new_err(
                "reports and provenances must have matching lengths",
            ));
        }
        let entries: Vec<SuiteReportEntry> = reports
            .into_iter()
            .zip(provenances)
            .map(|(r, p)| SuiteReportEntry::new(p.inner, r.inner))
            .collect();
        Ok(Self {
            inner: MultiSuiteReport::from_reports(entries),
        })
    }

    #[staticmethod]
    pub fn from_json(json_str: &str) -> PyResult<Self> {
        let multi: MultiSuiteReport = serde_json::from_str(json_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid MultiSuiteReport JSON: {e}")))?;
        Ok(Self { inner: multi })
    }

    #[getter]
    pub fn task_count(&self) -> usize {
        self.inner.task_count
    }

    #[getter]
    pub fn correct_count(&self) -> usize {
        self.inner.correct_count
    }

    #[getter]
    pub fn overall_accuracy(&self) -> f64 {
        self.inner.overall_accuracy
    }

    #[getter]
    pub fn mean_suite_accuracy(&self) -> f64 {
        self.inner.mean_suite_accuracy
    }

    #[getter]
    pub fn mean_suite_score(&self) -> f64 {
        self.inner.mean_suite_score
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {e}")))
    }

    pub fn to_json_pretty(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {e}")))
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let json_str = self.to_json()?;
        let json_mod = py.import("json")?;
        let dict = json_mod.call_method1("loads", (json_str,))?;
        dict.cast_into::<PyDict>()
            .map_err(|e| PyValueError::new_err(format!("Dict conversion failed: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiSuiteReport(tasks={}, overall_accuracy={:.4})",
            self.inner.task_count, self.inner.overall_accuracy
        )
    }

    fn __str__(&self) -> PyResult<String> {
        self.to_json()
    }
}
