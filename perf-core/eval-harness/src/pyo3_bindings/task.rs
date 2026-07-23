use crate::dataset::Dataset;
use crate::provenance::DatasetProvenance;
use crate::{gpqa, mmlu, terminal_bench, Suite};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::parse_suite;

/// Provenance metadata for a loaded evaluation dataset.
#[pyclass(name = "DatasetProvenance", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyDatasetProvenance {
    pub inner: DatasetProvenance,
}

#[pymethods]
impl PyDatasetProvenance {
    #[new]
    #[pyo3(signature = (source, source_revision, split, bytes, task_count))]
    pub fn new(
        source: String,
        source_revision: String,
        split: String,
        bytes: &[u8],
        task_count: usize,
    ) -> Self {
        Self {
            inner: DatasetProvenance::new(source, source_revision, split, bytes, task_count),
        }
    }

    #[getter]
    pub fn source(&self) -> String {
        self.inner.source.clone()
    }

    #[getter]
    pub fn source_revision(&self) -> String {
        self.inner.source_revision.clone()
    }

    #[getter]
    pub fn split(&self) -> String {
        self.inner.split.clone()
    }

    #[getter]
    pub fn sha256(&self) -> String {
        self.inner.sha256.clone()
    }

    #[getter]
    pub fn task_count(&self) -> usize {
        self.inner.task_count
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {e}")))
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("source", &self.inner.source)?;
        dict.set_item("source_revision", &self.inner.source_revision)?;
        dict.set_item("split", &self.inner.split)?;
        dict.set_item("sha256", &self.inner.sha256)?;
        dict.set_item("task_count", self.inner.task_count)?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// A single evaluation task specification.
#[pyclass(name = "TaskSpec", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyTaskSpec {
    pub inner: crate::TaskSpec,
}

#[pymethods]
impl PyTaskSpec {
    #[getter]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    pub fn suite(&self) -> String {
        self.inner.suite.as_str().to_string()
    }

    #[getter]
    pub fn prompt(&self) -> String {
        self.inner.prompt.clone()
    }

    #[getter]
    pub fn expected(&self) -> Option<String> {
        self.inner.expected.clone()
    }

    #[getter]
    pub fn choices(&self) -> Vec<String> {
        self.inner.choices.clone()
    }

    pub fn is_multiple_choice(&self) -> bool {
        self.inner.is_multiple_choice()
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
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
        format!("{:?}", self.inner)
    }
}

/// Evaluation result for a single task.
#[pyclass(name = "TaskResult", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyTaskResult {
    pub inner: crate::TaskResult,
}

#[pymethods]
impl PyTaskResult {
    #[getter]
    pub fn task_id(&self) -> String {
        self.inner.task_id.clone()
    }

    #[getter]
    pub fn suite(&self) -> String {
        self.inner.suite.as_str().to_string()
    }

    #[getter]
    pub fn prompt_tokens(&self) -> usize {
        self.inner.prompt_tokens
    }

    #[getter]
    pub fn completion_tokens(&self) -> usize {
        self.inner.completion_tokens
    }

    #[getter]
    pub fn completion(&self) -> String {
        self.inner.completion.clone()
    }

    #[getter]
    pub fn normalized_completion(&self) -> String {
        self.inner.normalized_completion.clone()
    }

    #[getter]
    pub fn correct(&self) -> bool {
        self.inner.correct
    }

    #[getter]
    pub fn score(&self) -> f64 {
        self.inner.score
    }

    #[getter]
    pub fn latency_ms(&self) -> f64 {
        self.inner.latency_ms
    }

    #[getter]
    pub fn matched_answer(&self) -> Option<String> {
        self.inner.matched_answer.clone()
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
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
        format!("{:?}", self.inner)
    }
}

/// A loaded dataset with associated provenance and tasks.
#[pyclass(name = "Dataset", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyDataset {
    pub inner: Dataset,
}

#[pymethods]
impl PyDataset {
    #[staticmethod]
    #[pyo3(signature = (suite, path, source_revision=None, split=None))]
    pub fn load(
        suite: &str,
        path: &str,
        source_revision: Option<&str>,
        split: Option<&str>,
    ) -> PyResult<Self> {
        let suite_enum = parse_suite(suite)?;
        let rev = source_revision.unwrap_or("unspecified");
        let sp = split.unwrap_or("unspecified");
        let ds = match suite_enum {
            Suite::Mmlu => mmlu::load_csv_with_provenance(path, rev, sp)?,
            Suite::Gpqa => gpqa::load_jsonl_with_provenance(path, rev, sp)?,
            Suite::TerminalBench => terminal_bench::load_yaml_with_provenance(path, rev, sp)?,
            Suite::Perplexity => {
                return Err(PyValueError::new_err(
                    "Perplexity suite does not use file-based dataset loading",
                ));
            }
        };
        Ok(Self { inner: ds })
    }

    #[staticmethod]
    #[pyo3(signature = (suite, bytes, source="<bytes>", source_revision=None, split=None))]
    pub fn load_bytes(
        suite: &str,
        bytes: &[u8],
        source: &str,
        source_revision: Option<&str>,
        split: Option<&str>,
    ) -> PyResult<Self> {
        let suite_enum = parse_suite(suite)?;
        let rev = source_revision.unwrap_or("unspecified");
        let sp = split.unwrap_or("unspecified");
        let ds = match suite_enum {
            Suite::Mmlu => mmlu::load_csv_bytes(bytes, source, rev, sp)?,
            Suite::Gpqa => gpqa::load_jsonl_bytes(bytes, source, rev, sp)?,
            Suite::TerminalBench => terminal_bench::load_yaml_bytes(bytes, source, rev, sp)?,
            Suite::Perplexity => {
                return Err(PyValueError::new_err(
                    "Perplexity suite does not use file-based dataset loading",
                ));
            }
        };
        Ok(Self { inner: ds })
    }

    #[staticmethod]
    pub fn from_json(json_str: &str) -> PyResult<Self> {
        let ds: Dataset = serde_json::from_str(json_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid dataset JSON: {e}")))?;
        Ok(Self { inner: ds })
    }

    #[getter]
    pub fn suite(&self) -> String {
        self.inner.suite().as_str().to_string()
    }

    #[getter]
    pub fn provenance(&self) -> PyDatasetProvenance {
        PyDatasetProvenance {
            inner: self.inner.provenance().clone(),
        }
    }

    #[getter]
    pub fn tasks(&self) -> Vec<PyTaskSpec> {
        self.inner
            .as_tasks()
            .iter()
            .cloned()
            .map(|t| PyTaskSpec { inner: t })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Serialization error: {e}")))
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let json_str = self.to_json()?;
        let json_mod = py.import("json")?;
        let dict = json_mod.call_method1("loads", (json_str,))?;
        dict.cast_into::<PyDict>()
            .map_err(|e| PyValueError::new_err(format!("Dict conversion failed: {e}")))
    }

    #[pyo3(signature = (backend=None))]
    pub fn run_suite(
        &self,
        py: Python<'_>,
        backend: Option<Py<PyAny>>,
    ) -> PyResult<super::evaluation::PyEvaluationReport> {
        super::py_run_suite(py, self, backend)
    }

    #[pyo3(signature = (backend=None))]
    pub fn run_suite_json(&self, py: Python<'_>, backend: Option<Py<PyAny>>) -> PyResult<String> {
        super::py_run_suite_json(py, self, backend)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Dataset(suite={}, tasks={})",
            self.inner.suite().as_str(),
            self.inner.len()
        )
    }
}

/// Fixture oracle backend for deterministic testing.
#[pyclass(name = "OracleBackend", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyOracleBackend {
    pub tasks: std::sync::Arc<Vec<crate::TaskSpec>>,
}

#[pymethods]
impl PyOracleBackend {
    #[new]
    pub fn new(dataset: &PyDataset) -> Self {
        Self {
            tasks: std::sync::Arc::new(dataset.inner.as_tasks().to_vec()),
        }
    }
}
