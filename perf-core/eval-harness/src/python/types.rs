//! PyO3 type wrappers for eval-harness data structures.

use pyo3::prelude::*;

use crate::Suite;
use crate::{TaskResult, TaskSpec};

// ── Suite ──────────────────────────────────────────────────────────────────

/// Evaluation suite discriminator.
#[pyclass(eq, eq_int, frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum PySuite {
    Mmlu,
    Gpqa,
    TerminalBench,
    Perplexity,
}

#[pymethods]
impl PySuite {
    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        match s {
            "mmlu" => Ok(Self::Mmlu),
            "gpqa" => Ok(Self::Gpqa),
            "terminal-bench" => Ok(Self::TerminalBench),
            "perplexity" => Ok(Self::Perplexity),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown suite: {s}"
            ))),
        }
    }

    pub(super) fn as_str(&self) -> &'static str {
        Suite::from(*self).as_str()
    }

    fn __repr__(&self) -> String {
        format!("Suite.{}", self.as_str())
    }
}

impl From<PySuite> for Suite {
    fn from(s: PySuite) -> Self {
        match s {
            PySuite::Mmlu => Suite::Mmlu,
            PySuite::Gpqa => Suite::Gpqa,
            PySuite::TerminalBench => Suite::TerminalBench,
            PySuite::Perplexity => Suite::Perplexity,
        }
    }
}

impl From<Suite> for PySuite {
    fn from(s: Suite) -> Self {
        match s {
            Suite::Mmlu => PySuite::Mmlu,
            Suite::Gpqa => PySuite::Gpqa,
            Suite::TerminalBench => PySuite::TerminalBench,
            Suite::Perplexity => PySuite::Perplexity,
        }
    }
}

// ── Criteria ───────────────────────────────────────────────────────────────

/// Substring-gating criteria for terminal-bench style tasks.
#[pyclass(from_py_object)]
#[derive(Clone, Default)]
pub(super) struct PyCriteria {
    #[pyo3(get)]
    expected_commands: Vec<String>,
    #[pyo3(get)]
    required_output: Vec<String>,
    #[pyo3(get)]
    forbidden_output: Vec<String>,
}

#[pymethods]
impl PyCriteria {
    #[new]
    #[pyo3(signature = (expected_commands=None, required_output=None, forbidden_output=None))]
    fn new(
        expected_commands: Option<Vec<String>>,
        required_output: Option<Vec<String>>,
        forbidden_output: Option<Vec<String>>,
    ) -> Self {
        Self {
            expected_commands: expected_commands.unwrap_or_default(),
            required_output: required_output.unwrap_or_default(),
            forbidden_output: forbidden_output.unwrap_or_default(),
        }
    }
}

impl PyCriteria {
    pub(super) fn from_rust(c: crate::Criteria) -> Self {
        Self {
            expected_commands: c.expected_commands,
            required_output: c.required_output,
            forbidden_output: c.forbidden_output,
        }
    }

    pub(super) fn into_rust(self) -> crate::Criteria {
        crate::Criteria {
            expected_commands: self.expected_commands,
            required_output: self.required_output,
            forbidden_output: self.forbidden_output,
        }
    }
}

// ── TaskSpec ───────────────────────────────────────────────────────────────

/// A single evaluation task specification.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub(super) struct PyTaskSpec {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    suite: PySuite,
    #[pyo3(get)]
    prompt: String,
    #[pyo3(get)]
    expected: Option<String>,
    #[pyo3(get)]
    choices: Vec<String>,
    #[pyo3(get)]
    criteria: Option<PyCriteria>,
}

#[pymethods]
impl PyTaskSpec {
    /// Create a multiple-choice task.
    #[staticmethod]
    #[pyo3(signature = (id, suite, prompt, choices, expected))]
    fn multiple_choice(
        id: String,
        suite: PySuite,
        prompt: String,
        choices: Vec<String>,
        expected: String,
    ) -> Self {
        let rust = TaskSpec::multiple_choice(id, suite.into(), prompt, choices, &expected);
        Self::from_rust(rust)
    }

    /// Create an open-ended task with an expected answer.
    #[staticmethod]
    #[pyo3(signature = (id, suite, prompt, expected=None, criteria=None))]
    fn open_ended(
        id: String,
        suite: PySuite,
        prompt: String,
        expected: Option<String>,
        criteria: Option<PyCriteria>,
    ) -> Self {
        Self {
            id,
            suite,
            prompt,
            expected,
            choices: Vec::new(),
            criteria,
        }
    }

    fn is_multiple_choice(&self) -> bool {
        !self.choices.is_empty()
    }

    fn __repr__(&self) -> String {
        format!(
            "TaskSpec(id={}, suite={}, prompt_len={})",
            self.id,
            self.suite.as_str(),
            self.prompt.len()
        )
    }
}

impl PyTaskSpec {
    fn from_rust(t: TaskSpec) -> Self {
        Self {
            id: t.id,
            suite: t.suite.into(),
            prompt: t.prompt,
            expected: t.expected,
            choices: t.choices,
            criteria: t.criteria.map(PyCriteria::from_rust),
        }
    }

    pub(super) fn into_rust(self) -> TaskSpec {
        TaskSpec {
            id: self.id,
            suite: self.suite.into(),
            prompt: self.prompt,
            expected: self.expected,
            choices: self.choices,
            criteria: self.criteria.map(|c| c.into_rust()),
        }
    }
}

// ── TaskResult ─────────────────────────────────────────────────────────────

/// Result of scoring a single task.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub(super) struct PyTaskResult {
    #[pyo3(get)]
    pub(super) task_id: String,
    #[pyo3(get)]
    pub(super) suite: PySuite,
    #[pyo3(get)]
    pub(super) prompt_tokens: usize,
    #[pyo3(get)]
    pub(super) completion_tokens: usize,
    #[pyo3(get)]
    pub(super) completion: String,
    #[pyo3(get)]
    pub(super) normalized_completion: String,
    #[pyo3(get)]
    pub(super) correct: bool,
    #[pyo3(get)]
    pub(super) score: f64,
    #[pyo3(get)]
    pub(super) latency_ms: f64,
    #[pyo3(get)]
    pub(super) matched_answer: Option<String>,
}

#[pymethods]
impl PyTaskResult {
    fn __repr__(&self) -> String {
        format!(
            "TaskResult(id={}, correct={}, score={:.4})",
            self.task_id, self.correct, self.score
        )
    }
}

impl From<TaskResult> for PyTaskResult {
    fn from(r: TaskResult) -> Self {
        Self {
            task_id: r.task_id,
            suite: r.suite.into(),
            prompt_tokens: r.prompt_tokens,
            completion_tokens: r.completion_tokens,
            completion: r.completion,
            normalized_completion: r.normalized_completion,
            correct: r.correct,
            score: r.score,
            latency_ms: r.latency_ms,
            matched_answer: r.matched_answer,
        }
    }
}

impl PyTaskResult {
    pub(super) fn into_rust(self) -> TaskResult {
        TaskResult {
            task_id: self.task_id,
            suite: self.suite.into(),
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            completion: self.completion,
            normalized_completion: self.normalized_completion,
            correct: self.correct,
            score: self.score,
            latency_ms: self.latency_ms,
            matched_answer: self.matched_answer,
        }
    }
}

// ── EvaluationReport ───────────────────────────────────────────────────────
