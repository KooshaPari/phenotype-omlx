//! Evaluation harness for the Metal model runtime.
//!
//! The harness scores deterministic completions against task specifications for
//! MMLU, GPQA, terminal-bench, and perplexity suites. Scoring is pure: it
//! never executes commands or shells out. All loaded datasets carry explicit
//! [`provenance::DatasetProvenance`] so results are attributable to the exact
//! bytes that were evaluated.
//!
//! Public modules:
//! - [`dataset`]: the [`dataset::Dataset`] wrapper that bundles tasks with
//!   provenance.
//! - [`provenance`]: source, revision, split, and SHA-256 content hash.
//! - [`mmlu`], [`gpqa`], [`terminal_bench`], [`perplexity`]: per-suite
//!   loaders and helpers.
//! - [`report`]: cross-suite report aggregation.
//!
//! Example:
//! ```no_run
//! use eval_harness::mmlu;
//! let dataset = mmlu::load_csv_with_provenance("path/to/mmlu.csv", "v1.0", "test").unwrap();
//! assert_eq!(dataset.suite(), eval_harness::Suite::MMLU);
//! assert_eq!(dataset.provenance().task_count, dataset.len());
//! ```

pub mod dataset;
pub mod gpqa;
pub mod mmlu;
pub mod perplexity;
pub mod provenance;
pub mod report;
pub mod terminal_bench;
pub mod verifier;

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Suite {
    MMLU,
    GPQA,
    TerminalBench,
    Perplexity,
}

impl Suite {
    /// Stable lowercase string identifier for the suite. Used in serialized
    /// reports and dataset records.
    pub fn as_str(&self) -> &'static str {
        match self {
            Suite::MMLU => "mmlu",
            Suite::GPQA => "gpqa",
            Suite::TerminalBench => "terminal-bench",
            Suite::Perplexity => "perplexity",
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Default, PartialEq, Eq)]
pub struct Criteria {
    #[serde(default)]
    pub expected_commands: Vec<String>,
    #[serde(default)]
    pub required_output: Vec<String>,
    #[serde(default)]
    pub forbidden_output: Vec<String>,
}

/// A single evaluation task.
///
/// `choices` is empty (not `None`) for tasks without multiple choice; this
/// keeps index-based access ergonomic for callers and aligns with the
/// `score_choice` contract that requires `num_choices >= 1`. `expected` is the
/// canonical answer (or the canonical answer letter for choice tasks).
/// `criteria` is used for terminal-bench style substring gating and takes
/// precedence over `expected`/`choices` when present.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct TaskSpec {
    pub id: String,
    pub suite: Suite,
    pub prompt: String,
    #[serde(default)]
    pub expected: Option<String>,
    #[serde(default)]
    pub choices: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<Criteria>,
}

impl TaskSpec {
    /// Construct a multiple-choice task. The choices are stored in order and
    /// labeled `A`, `B`, `C`, ... at scoring time. The `choices` argument
    /// accepts any iterator of string-like values so callers can pass `vec!["x"]`
    /// without manually converting each element.
    pub fn multiple_choice<I, S>(
        id: impl Into<String>,
        suite: Suite,
        prompt: impl Into<String>,
        choices: I,
        expected: impl Into<String>,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let choices: Vec<String> = choices.into_iter().map(Into::into).collect();
        Self {
            id: id.into(),
            suite,
            prompt: prompt.into(),
            expected: Some(expected.into()),
            choices,
            criteria: None,
        }
    }

    /// True when this task has a non-empty `choices` list.
    pub fn is_multiple_choice(&self) -> bool {
        !self.choices.is_empty()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct TaskResult {
    pub task_id: String,
    pub suite: Suite,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub completion: String,
    pub normalized_completion: String,
    pub correct: bool,
    pub score: f64,
    pub latency_ms: f64,
    pub matched_answer: Option<String>,
}

/// Per-suite aggregation of task results.
///
/// `accuracy` is the fraction of tasks with `correct == true`. `mean_score`
/// is the arithmetic mean of per-task `score` and is included for suites
/// (e.g. perplexity) that emit non-binary values.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct EvaluationReport {
    pub suite: Suite,
    pub task_count: usize,
    pub correct_count: usize,
    pub accuracy: f64,
    pub mean_score: f64,
    pub results: Vec<TaskResult>,
}

impl EvaluationReport {
    /// Aggregate per-task results into a deterministic report. Results are
    /// sorted by `task_id` so the report is reproducible regardless of input
    /// order.
    pub fn from_results(suite: Suite, mut results: Vec<TaskResult>) -> Self {
        results.sort_by(|a, b| a.task_id.cmp(&b.task_id));
        let task_count = results.len();
        let correct_count = results.iter().filter(|r| r.correct).count();
        let accuracy = if task_count == 0 {
            0.0
        } else {
            correct_count as f64 / task_count as f64
        };
        let mean_score = if task_count == 0 {
            0.0
        } else {
            results.iter().map(|r| r.score).sum::<f64>() / task_count as f64
        };
        Self {
            suite,
            task_count,
            correct_count,
            accuracy,
            mean_score,
            results,
        }
    }
}

/// All errors produced by the eval harness. Every variant carries enough
/// context (suite, path, line, field) to identify the offending record.
///
/// `path` is always the pure origin (file path or logical source name);
/// `line` is a separate, structured 1-based line number wherever the
/// underlying parser exposes one. Loaders populate `line` from
/// `serde_json::Error::line()` or `serde_yaml::Error::location()`; when the
/// parser does not expose a location, callers fall back to `1` (the header
/// line) rather than reporting `0`.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("io error reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("json error in {path} at line {line}: {source}")]
    Json {
        path: String,
        /// 1-based line number from `serde_json::Error::line()`; `1` when
        /// the parser does not report a location.
        line: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("yaml error in {path} at line {line}: {source}")]
    Yaml {
        path: String,
        /// 1-based line number from `serde_yaml::Error::location()`; `1`
        /// when the parser does not report a location.
        line: usize,
        #[source]
        source: serde_yaml::Error,
    },
    #[error("csv format error in {path} at line {line}: {message}")]
    Csv {
        path: String,
        line: usize,
        message: String,
    },
    #[error("missing required field '{field}' in {path}")]
    MissingField { path: String, field: &'static str },
    #[error("malformed record in {path} at line {line}: {message}")]
    Malformed {
        path: String,
        line: usize,
        message: String,
    },
    #[error("inconsistent suite in {path}: expected {expected:?}, got {actual:?}")]
    SuiteMismatch {
        path: String,
        expected: Suite,
        actual: Suite,
    },
}

impl EvalError {
    /// Wrap an I/O error with the originating path for richer context.
    pub fn io(path: impl Into<String>, source: std::io::Error) -> Self {
        EvalError::Io {
            path: path.into(),
            source,
        }
    }

    /// Wrap a JSON parse error with the originating path and a 1-based line
    /// number. The line is taken from `serde_json::Error::line()` when the
    /// error carries one; callers that already know the line (e.g. JSONL
    /// record position) should use [`EvalError::json_at_line`] instead.
    pub fn json(path: impl Into<String>, source: serde_json::Error) -> Self {
        let line = source.line();
        EvalError::Json {
            path: path.into(),
            line,
            source,
        }
    }

    /// Like [`EvalError::json`] but uses a caller-supplied line number. Used
    /// by JSONL loaders that know the record's position in the source file
    /// (1-based).
    pub fn json_at_line(path: impl Into<String>, line: usize, source: serde_json::Error) -> Self {
        EvalError::Json {
            path: path.into(),
            line,
            source,
        }
    }

    /// Wrap a YAML parse error with the originating path and a 1-based line
    /// number. The line is taken from `serde_yaml::Error::location()` when
    /// the error carries one; otherwise `1` is reported as the stable fallback.
    pub fn yaml(path: impl Into<String>, source: serde_yaml::Error) -> Self {
        let line = source.location().map(|loc| loc.line()).unwrap_or(1);
        EvalError::Yaml {
            path: path.into(),
            line,
            source,
        }
    }

    /// Build a CSV format error with explicit path and 1-based line number.
    pub fn csv(path: impl Into<String>, line: usize, message: impl Into<String>) -> Self {
        EvalError::Csv {
            path: path.into(),
            line,
            message: message.into(),
        }
    }

    /// Build a missing-field error with explicit path.
    pub fn missing_field(path: impl Into<String>, field: &'static str) -> Self {
        EvalError::MissingField {
            path: path.into(),
            field,
        }
    }

    /// Build a malformed-record error with explicit path and 1-based line number.
    pub fn malformed(path: impl Into<String>, line: usize, message: impl Into<String>) -> Self {
        EvalError::Malformed {
            path: path.into(),
            line,
            message: message.into(),
        }
    }
}

pub type Result<T> = std::result::Result<T, EvalError>;

/// Normalize an answer for comparison: trim whitespace, lowercase, drop
/// trailing non-alphanumeric characters. Designed to keep substring
/// false-positives out of exact-match scoring (e.g. trailing punctuation,
/// newlines).
pub fn normalize_answer(s: &str) -> String {
    s.trim()
        .to_lowercase()
        .trim_end_matches(|c: char| !c.is_alphanumeric())
        .to_string()
}

/// Exact-match scoring using normalized comparison of completion and expected.
pub fn score_exact(completion: &str, expected: &str) -> bool {
    normalize_answer(completion) == normalize_answer(expected)
}

/// Multiple-choice scoring. The completion is scanned for the model's chosen
/// letter (e.g. `(b)` or trailing `B.`) and compared to `expected`. Letters
/// outside `[A, num_choices]` are ignored so prose like "the answer is C"
/// never falsely matches when `expected` is a different choice.
pub fn score_choice(completion: &str, expected: &str, num_choices: usize) -> bool {
    if num_choices == 0 || num_choices > 26 {
        return false;
    }
    let expected_letter = match expected
        .trim()
        .chars()
        .next()
        .map(|c| c.to_ascii_uppercase())
    {
        Some(c) if c.is_ascii_uppercase() => c,
        _ => return false,
    };
    let max_letter = (b'A' + num_choices as u8 - 1) as char;
    let valid = |c: char| c >= 'A' && c <= max_letter;

    // Prefer parenthesized markers like `(b)` to avoid picking up letters
    // embedded in prose such as "the correct answer".
    let chars: Vec<char> = completion.chars().collect();
    let mut i = 0;
    while i + 2 < chars.len() {
        if chars[i] == '(' && chars[i + 2] == ')' {
            let inner = chars[i + 1];
            if inner.is_ascii_alphabetic() && valid(inner.to_ascii_uppercase()) {
                return inner.to_ascii_uppercase() == expected_letter;
            }
        }
        i += 1;
    }

    // Fall back to the last standalone uppercase letter that is a valid choice.
    for ch in completion.chars().rev() {
        if ch.is_ascii_uppercase() && valid(ch) {
            return ch == expected_letter;
        }
    }
    false
}

/// Score a single completion against a task. The pure deterministic scorer
/// never executes commands or shells out — it inspects the completion string
/// only. Returns `Err` only for fundamentally malformed input.
pub fn evaluate(task: &TaskSpec, completion: &str) -> Result<TaskResult> {
    let normalized = normalize_answer(completion);
    let prompt_tokens = task.prompt.split_whitespace().count();
    let completion_tokens = completion.split_whitespace().count();

    let (correct, score, matched_answer) = if let Some(criteria) = task.criteria.as_ref() {
        // Criteria-based scoring: every expected command must appear, every
        // required output must appear, and no forbidden output may appear.
        let expected_ok = criteria
            .expected_commands
            .iter()
            .all(|c| completion.contains(c.as_str()));
        let required_ok = criteria
            .required_output
            .iter()
            .all(|c| completion.contains(c.as_str()));
        let forbidden_ok = criteria
            .forbidden_output
            .iter()
            .all(|c| !completion.contains(c.as_str()));
        let ok = expected_ok && required_ok && forbidden_ok;
        (ok, if ok { 1.0 } else { 0.0 }, None)
    } else if let Some(expected) = task.expected.as_ref() {
        if task.is_multiple_choice() {
            let correct = score_choice(completion, expected, task.choices.len());
            (
                correct,
                if correct { 1.0 } else { 0.0 },
                Some(expected.clone()),
            )
        } else {
            let correct = score_exact(completion, expected);
            (
                correct,
                if correct { 1.0 } else { 0.0 },
                Some(expected.clone()),
            )
        }
    } else {
        (false, 0.0, None)
    };

    Ok(TaskResult {
        task_id: task.id.clone(),
        suite: task.suite,
        prompt_tokens,
        completion_tokens,
        completion: completion.to_string(),
        normalized_completion: normalized,
        correct,
        score,
        latency_ms: 0.0,
        matched_answer,
    })
}

pub trait Backend {
    fn complete(&self, prompt: &str, max_tokens: usize) -> (String, f64);
}

pub fn run_suite<B: Backend>(suite: Suite, backend: &B, tasks: &[TaskSpec]) -> Vec<TaskResult> {
    tasks
        .iter()
        .filter(|t| t.suite == suite)
        .map(|t| {
            let (completion, latency_ms) = backend.complete(&t.prompt, 128);
            let mut result = evaluate(t, &completion).unwrap_or(TaskResult {
                task_id: t.id.clone(),
                suite: t.suite,
                prompt_tokens: t.prompt.split_whitespace().count(),
                completion_tokens: completion.split_whitespace().count(),
                completion: completion.clone(),
                normalized_completion: normalize_answer(&completion),
                correct: false,
                score: 0.0,
                latency_ms,
                matched_answer: None,
            });
            result.latency_ms = latency_ms;
            result
        })
        .collect()
}
