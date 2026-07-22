//! GPQA (Graduate-Level Google-Proof Q&A) dataset loader.
//!
//! GPQA tasks are multiple-choice science questions. Each JSONL line must be
//! a JSON object with the following fields:
//!
//! ```text
//! {"id": "...", "question": "...", "choices": ["..."], "answer": "A"}
//! ```
//!
//! `answer` is a single letter; the loader validates that the letter is in
//! range for the supplied `choices`. Loaded task IDs are prefixed with
//! `gpqa_` and the resulting dataset is sorted by task id so the loader
//! output is deterministic.
//!
//! The loader produces a [`crate::Dataset`] with explicit
//! [`crate::provenance::DatasetProvenance`]. There are no built-in placeholder
//! tasks — callers must supply real JSONL bytes (or a file path) and an
//! upstream source revision.

use crate::dataset::Dataset;
use crate::provenance::DatasetProvenance;
use crate::{EvalError, Result, Suite, TaskSpec};
use serde::Deserialize;
use std::path::Path;

/// One GPQA record as it appears in a JSONL file.
#[derive(Debug, Deserialize)]
struct GpqaRow {
    id: String,
    question: String,
    choices: Vec<String>,
    answer: String,
}

/// Load GPQA tasks from a JSONL file on disk, returning a [`Dataset`].
///
/// This is a convenience wrapper around [`load_jsonl_with_provenance`] that
/// uses stable placeholder values (`"unspecified"`) for the upstream
/// `source_revision` and `split`. Callers that need to attribute results to a
/// specific dataset revision should call [`load_jsonl_with_provenance`]
/// directly.
pub fn load_jsonl<P: AsRef<Path>>(path: P) -> Result<Dataset> {
    load_jsonl_with_provenance(path, "unspecified", "unspecified")
}

/// Load GPQA tasks from a JSONL file on disk, returning a [`Dataset`] with
/// provenance computed from the file bytes.
pub fn load_jsonl_with_provenance<P: AsRef<Path>>(
    path: P,
    source_revision: impl Into<String>,
    split: impl Into<String>,
) -> Result<Dataset> {
    let path_ref = path.as_ref();
    let display = path_ref.display().to_string();
    let bytes = std::fs::read(path_ref).map_err(|e| EvalError::io(display.clone(), e))?;
    load_jsonl_bytes(&bytes, display, source_revision, split)
}

/// Load GPQA tasks from raw JSONL bytes. Exposed for callers that already
/// hold the bytes and for tests that want to feed synthetic payloads.
pub fn load_jsonl_bytes(
    bytes: &[u8],
    source: impl Into<String>,
    source_revision: impl Into<String>,
    split: impl Into<String>,
) -> Result<Dataset> {
    let source = source.into();
    let content = std::str::from_utf8(bytes)
        .map_err(|e| EvalError::malformed(source.clone(), 0, format!("non-utf8 bytes: {e}")))?;
    let tasks = parse_jsonl(content, &source)?;
    let provenance = DatasetProvenance::new(source, source_revision, split, bytes, tasks.len());
    Ok(Dataset::new(Suite::Gpqa, provenance, tasks))
}

fn parse_jsonl(content: &str, path: &str) -> Result<Vec<TaskSpec>> {
    let mut tasks = Vec::new();
    // Each non-empty line is a record; line numbers are 1-based.
    for (line_idx, line) in content.lines().enumerate() {
        let line_no = line_idx + 1;
        if line.trim().is_empty() {
            continue;
        }
        let row: GpqaRow =
            serde_json::from_str(line).map_err(|e| EvalError::json_at_line(path, line_no, e))?;
        validate_row(&row, path, line_no)?;
        let labeled: Vec<String> = row
            .choices
            .iter()
            .enumerate()
            .map(|(i, value)| format!("{}) {}", char::from(b'A' + i as u8), value))
            .collect();
        let prompt = format!("{}\n{}\nAnswer:", row.question, labeled.join("\n"));
        tasks.push(TaskSpec {
            id: format!("gpqa_{}", row.id),
            suite: Suite::Gpqa,
            prompt,
            expected: Some(row.answer),
            choices: row.choices,
            // GPQA is multiple-choice, so criteria-based scoring is disabled.
            criteria: None,
        });
    }
    if tasks.is_empty() {
        return Err(EvalError::malformed(path, 1, "no data records found"));
    }
    tasks.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(tasks)
}

fn validate_row(row: &GpqaRow, path: &str, line_no: usize) -> Result<()> {
    if row.id.trim().is_empty() {
        return Err(EvalError::malformed(path, line_no, "id is empty"));
    }
    if row.choices.is_empty() {
        return Err(EvalError::malformed(path, line_no, "choices is empty"));
    }
    if row.choices.len() > 26 {
        return Err(EvalError::malformed(
            path,
            line_no,
            format!("choices has {} entries; max is 26", row.choices.len()),
        ));
    }
    let letter = row
        .answer
        .trim()
        .chars()
        .next()
        .map(|c| c.to_ascii_uppercase())
        .unwrap_or('?');
    if !letter.is_ascii_uppercase() {
        return Err(EvalError::malformed(
            path,
            line_no,
            format!("answer '{}' is not a letter", row.answer),
        ));
    }
    let idx = (letter as u8 - b'A') as usize;
    if idx >= row.choices.len() {
        return Err(EvalError::malformed(
            path,
            line_no,
            format!(
                "answer '{}' is out of range for {} choices",
                row.answer,
                row.choices.len()
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_JSONL: &str =
        "{\"id\":\"chemistry-1\",\"question\":\"Q?\",\"choices\":[\"x\",\"y\"],\"answer\":\"A\"}\n\
{\"id\":\"biology-1\",\"question\":\"Q?\",\"choices\":[\"x\",\"y\"],\"answer\":\"B\"}\n";

    #[test]
    fn parses_minimal_jsonl_with_provenance() {
        let dataset =
            load_jsonl_bytes(MINIMAL_JSONL.as_bytes(), "test.jsonl", "v1", "diamond").unwrap();
        assert_eq!(dataset.suite(), Suite::Gpqa);
        assert_eq!(dataset.len(), 2);
        // Sorted by task id; biology sorts before chemistry.
        assert_eq!(dataset[0].id, "gpqa_biology-1");
        assert_eq!(dataset[1].id, "gpqa_chemistry-1");
        assert_eq!(dataset[1].choices, vec!["x", "y"]);
        assert_eq!(dataset[1].expected.as_deref(), Some("A"));
        assert!(dataset[1].is_multiple_choice());
        // Provenance.
        assert_eq!(dataset.provenance().source, "test.jsonl");
        assert_eq!(dataset.provenance().source_revision, "v1");
        assert_eq!(dataset.provenance().split, "diamond");
        assert_eq!(dataset.provenance().task_count, 2);
    }

    #[test]
    fn rejects_empty_input() {
        let bytes: &[u8] = b"";
        let err = load_jsonl_bytes(bytes, "x.jsonl", "v1", "test").unwrap_err();
        match err {
            EvalError::Malformed { line, .. } => assert_eq!(line, 1),
            other => panic!("expected Malformed error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_malformed_json() {
        let bytes = b"{not json\n";
        let err = load_jsonl_bytes(bytes, "x.jsonl", "v1", "test").unwrap_err();
        match err {
            EvalError::Json { path, line, .. } => {
                assert_eq!(path, "x.jsonl");
                assert_eq!(line, 1);
            }
            other => panic!("expected Json error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_id() {
        let bytes = b"{\"id\":\"\",\"question\":\"Q\",\"choices\":[\"x\"],\"answer\":\"A\"}\n";
        let err = load_jsonl_bytes(bytes, "x.jsonl", "v1", "test").unwrap_err();
        match err {
            EvalError::Malformed { line, message, .. } => {
                assert_eq!(line, 1);
                assert!(message.contains("id is empty"));
            }
            other => panic!("expected Malformed error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_choices() {
        let bytes = b"{\"id\":\"a\",\"question\":\"Q\",\"choices\":[],\"answer\":\"A\"}\n";
        let err = load_jsonl_bytes(bytes, "x.jsonl", "v1", "test").unwrap_err();
        match err {
            EvalError::Malformed { message, .. } => {
                assert!(message.contains("choices is empty"));
            }
            other => panic!("expected Malformed error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_too_many_choices() {
        let choices = (0..27)
            .map(|i| format!("\"c{i}\""))
            .collect::<Vec<_>>()
            .join(",");
        let bytes = format!(
            "{{\"id\":\"a\",\"question\":\"Q\",\"choices\":[{choices}],\"answer\":\"A\"}}\n"
        );
        let err = load_jsonl_bytes(bytes.as_bytes(), "x.jsonl", "v1", "test").unwrap_err();
        match err {
            EvalError::Malformed { message, .. } => {
                assert!(message.contains("max is 26"));
            }
            other => panic!("expected Malformed error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_answer_out_of_range() {
        let bytes =
            b"{\"id\":\"a\",\"question\":\"Q\",\"choices\":[\"x\",\"y\"],\"answer\":\"Z\"}\n";
        let err = load_jsonl_bytes(bytes, "x.jsonl", "v1", "test").unwrap_err();
        match err {
            EvalError::Malformed { line, message, .. } => {
                assert_eq!(line, 1);
                assert!(message.contains("out of range"));
            }
            other => panic!("expected Malformed error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_letter_answer() {
        let bytes =
            b"{\"id\":\"a\",\"question\":\"Q\",\"choices\":[\"x\",\"y\"],\"answer\":\"1\"}\n";
        let err = load_jsonl_bytes(bytes, "x.jsonl", "v1", "test").unwrap_err();
        match err {
            EvalError::Malformed { message, .. } => {
                assert!(message.contains("not a letter"));
            }
            other => panic!("expected Malformed error, got {other:?}"),
        }
    }

    #[test]
    fn skips_blank_lines_and_reports_correct_line_for_malformed() {
        // Two blank lines then a malformed JSON record. The malformed record
        // is on line 3 (1-indexed).
        let bytes = b"\n\n{not json\n";
        let err = load_jsonl_bytes(bytes, "x.jsonl", "v1", "test").unwrap_err();
        match err {
            EvalError::Json { path, line, .. } => {
                assert_eq!(path, "x.jsonl");
                assert_eq!(line, 3);
            }
            other => panic!("expected Json error, got {other:?}"),
        }
    }
}
