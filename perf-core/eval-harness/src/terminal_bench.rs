//! Terminal-bench dataset loader.
//!
//! Terminal-bench tasks are open-ended shell-style prompts scored on substring
//! matches rather than choice letters. The schema is intentionally
//! declarative:
//!
//! ```yaml
//! id: shell-find-fixme
//! prompt: Find all files containing FIXME.
//! criteria:
//!   expected_commands:
//!     - "grep -r 'FIXME' ."
//!   required_output:
//!     - "src/main.rs:FIXME"
//!   forbidden_output:
//!     - "permission denied"
//! ```
//!
//! The loader returns a [`crate::Dataset`] with explicit
//! [`crate::provenance::DatasetProvenance`]. No commands are executed; scoring
//! is pure substring evaluation by [`crate::evaluate`].

use crate::dataset::Dataset;
use crate::provenance::DatasetProvenance;
use crate::{Criteria, EvalError, Result, Suite, TaskSpec};
use serde::Deserialize;
use std::path::Path;

/// One terminal-bench task as it appears in a YAML file.
#[derive(Debug, Deserialize)]
struct TerminalBenchTask {
    id: String,
    prompt: String,
    criteria: Criteria,
}

/// Load a terminal-bench task from a YAML file on disk, returning a
/// [`Dataset`].
///
/// This is a convenience wrapper around [`load_yaml_with_provenance`] that
/// uses stable placeholder values (`"unspecified"`) for the upstream
/// `source_revision` and `split`. Callers that need to attribute results to a
/// specific dataset revision should call [`load_yaml_with_provenance`]
/// directly.
pub fn load_yaml<P: AsRef<Path>>(path: P) -> Result<Dataset> {
    load_yaml_with_provenance(path, "unspecified", "unspecified")
}

/// Load a terminal-bench task from a YAML file on disk, returning a
/// [`Dataset`] with provenance computed from the file bytes.
pub fn load_yaml_with_provenance<P: AsRef<Path>>(
    path: P,
    source_revision: impl Into<String>,
    split: impl Into<String>,
) -> Result<Dataset> {
    let path_ref = path.as_ref();
    let display = path_ref.display().to_string();
    let bytes = std::fs::read(path_ref).map_err(|e| EvalError::io(display.clone(), e))?;
    load_yaml_bytes(&bytes, display, source_revision, split)
}

/// Load a terminal-bench task from raw YAML bytes. Exposed for callers that
/// already hold the bytes and for tests that want to feed synthetic payloads.
pub fn load_yaml_bytes(
    bytes: &[u8],
    source: impl Into<String>,
    source_revision: impl Into<String>,
    split: impl Into<String>,
) -> Result<Dataset> {
    let source = source.into();
    let content = std::str::from_utf8(bytes)
        .map_err(|e| EvalError::malformed(source.clone(), 0, format!("non-utf8 bytes: {e}")))?;
    let raw: TerminalBenchTask =
        serde_yaml::from_str(content).map_err(|e| EvalError::yaml(source.clone(), e))?;
    validate(&raw, &source)?;
    let task = TaskSpec {
        id: raw.id,
        suite: Suite::TerminalBench,
        prompt: raw.prompt,
        expected: None,
        choices: vec![],
        criteria: Some(raw.criteria),
    };
    let provenance = DatasetProvenance::new(source, source_revision, split, bytes, 1);
    Ok(Dataset::new(Suite::TerminalBench, provenance, vec![task]))
}

fn validate(task: &TerminalBenchTask, path: &str) -> Result<()> {
    if task.id.trim().is_empty() {
        return Err(EvalError::malformed(path, 1, "id is empty"));
    }
    if task.prompt.trim().is_empty() {
        return Err(EvalError::malformed(path, 1, "prompt is empty"));
    }
    let c = &task.criteria;
    if c.expected_commands.is_empty()
        && c.required_output.is_empty()
        && c.forbidden_output.is_empty()
    {
        return Err(EvalError::malformed(
            path,
            1,
            "criteria has no expected_commands, required_output, or forbidden_output",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_YAML: &str = concat!(
        "id: shell-find-fixme\n",
        "prompt: Find all files containing FIXME.\n",
        "criteria:\n",
        "  expected_commands:\n",
        "    - \"grep -r 'FIXME' .\"\n",
        "  required_output:\n",
        "    - \"src/main.rs:FIXME\"\n",
        "  forbidden_output:\n",
        "    - \"permission denied\"\n",
    );

    #[test]
    fn parses_minimal_yaml_with_provenance() {
        let dataset = load_yaml_bytes(MINIMAL_YAML.as_bytes(), "task.yaml", "v1", "test").unwrap();
        assert_eq!(dataset.suite(), Suite::TerminalBench);
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset[0].id, "shell-find-fixme");
        assert_eq!(dataset[0].suite, Suite::TerminalBench);
        assert_eq!(dataset[0].expected, None);
        assert!(dataset[0].choices.is_empty());
        let criteria = dataset[0].criteria.as_ref().unwrap();
        assert_eq!(criteria.expected_commands, vec!["grep -r 'FIXME' ."]);
        assert_eq!(criteria.required_output, vec!["src/main.rs:FIXME"]);
        assert_eq!(criteria.forbidden_output, vec!["permission denied"]);
        // Provenance.
        assert_eq!(dataset.provenance().source, "task.yaml");
        assert_eq!(dataset.provenance().source_revision, "v1");
        assert_eq!(dataset.provenance().split, "test");
        assert_eq!(dataset.provenance().task_count, 1);
    }

    #[test]
    fn rejects_empty_id() {
        let bytes = b"id: \"\"\nprompt: P\ncriteria:\n  expected_commands: [\"x\"]\n";
        let err = load_yaml_bytes(bytes, "x.yaml", "v1", "test").unwrap_err();
        match err {
            EvalError::Malformed { message, .. } => {
                assert!(message.contains("id is empty"));
            }
            other => panic!("expected Malformed error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_prompt() {
        let bytes = b"id: a\nprompt: \"\"\ncriteria:\n  expected_commands: [\"x\"]\n";
        let err = load_yaml_bytes(bytes, "x.yaml", "v1", "test").unwrap_err();
        match err {
            EvalError::Malformed { message, .. } => {
                assert!(message.contains("prompt is empty"));
            }
            other => panic!("expected Malformed error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_criteria() {
        let bytes = b"id: a\nprompt: P\ncriteria: {}\n";
        let err = load_yaml_bytes(bytes, "x.yaml", "v1", "test").unwrap_err();
        match err {
            EvalError::Malformed { message, .. } => {
                assert!(message.contains("no expected_commands"));
            }
            other => panic!("expected Malformed error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_malformed_yaml() {
        let bytes = b"id: a\nprompt: P\ncriteria:\n  expected_commands: [unclosed\n";
        let err = load_yaml_bytes(bytes, "x.yaml", "v1", "test").unwrap_err();
        match err {
            EvalError::Yaml { .. } => {}
            other => panic!("expected Yaml error, got {other:?}"),
        }
    }
}
