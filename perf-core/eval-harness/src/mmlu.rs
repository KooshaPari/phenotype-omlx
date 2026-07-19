//! MMLU (Massive Multitask Language Understanding) dataset loader.
//!
//! MMLU tasks are multiple-choice: each row provides a subject, a question,
//! N choice columns, and a one-letter answer. Loaders produce
//! [`crate::Dataset`] values with explicit [`crate::provenance::DatasetProvenance`]
//! — there are no built-in placeholder tasks; callers must supply a real CSV
//! file (or bytes) and a source revision.
//!
//! The CSV schema is:
//! ```text
//! subject,question,<choice columns...>,answer
//! ```
//! Choice columns may be named `A`/`B`/`C`/`D`, `choice1`/`choice2`, or any
//! other header order. The loader preserves header order for choices.

use crate::dataset::Dataset;
use crate::provenance::DatasetProvenance;
use crate::{EvalError, Result, Suite, TaskSpec};
use std::path::Path;

/// Load MMLU tasks from a CSV file on disk, returning a [`Dataset`].
///
/// This is a convenience wrapper around [`load_csv_with_provenance`] that
/// uses stable placeholder values (`"unspecified"`) for the upstream
/// `source_revision` and `split`. Callers that need to attribute results to a
/// specific dataset revision should call [`load_csv_with_provenance`]
/// directly.
pub fn load_csv<P: AsRef<Path>>(path: P) -> Result<Dataset> {
    load_csv_with_provenance(path, "unspecified", "unspecified")
}

/// Load MMLU tasks from a CSV file on disk, returning a [`Dataset`] with
/// provenance computed from the file bytes.
///
/// The path is recorded as the dataset `source`, the file bytes are hashed for
/// `sha256`, and the caller supplies the upstream `source_revision` and
/// `split`.
pub fn load_csv_with_provenance<P: AsRef<Path>>(
    path: P,
    source_revision: impl Into<String>,
    split: impl Into<String>,
) -> Result<Dataset> {
    let path_ref = path.as_ref();
    let display = path_ref.display().to_string();
    let bytes = std::fs::read(path_ref).map_err(|e| EvalError::io(display.clone(), e))?;
    load_csv_bytes(&bytes, display, source_revision, split)
}

/// Load MMLU tasks from raw CSV bytes. Used by [`load_csv`] and exposed for
/// tests that want to feed synthetic bytes without touching the filesystem.
pub fn load_csv_bytes(
    bytes: &[u8],
    source: impl Into<String>,
    source_revision: impl Into<String>,
    split: impl Into<String>,
) -> Result<Dataset> {
    let source = source.into();
    let content = std::str::from_utf8(bytes)
        .map_err(|e| EvalError::malformed(source.clone(), 0, format!("non-utf8 bytes: {e}")))?;
    let tasks = parse_csv(content, &source)?;
    let provenance = DatasetProvenance::new(source, source_revision, split, bytes, tasks.len());
    Ok(Dataset::new(Suite::MMLU, provenance, tasks))
}

/// Parse a CSV string into MMLU tasks. Exposed for callers that already hold
/// the bytes; the path is only used in error messages.
///
/// Uses the `csv` crate so quoted fields containing commas, escaped quotes
/// (`""`), and CRLF line endings are handled per RFC 4180. Row numbers in
/// errors are 1-based and refer to the data row position in the source file
/// (the header is line 1).
fn parse_csv(content: &str, path: &str) -> Result<Vec<TaskSpec>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(content.as_bytes());

    let header = reader
        .headers()
        .map_err(|e| EvalError::csv(path, 1, format!("failed to read header: {e}")))?;
    let header_cols: Vec<String> = header.iter().map(|c| c.trim().to_string()).collect();

    if header_cols.is_empty() {
        return Err(EvalError::csv(path, 1, "file is empty (missing header)"));
    }
    let mut unique_headers = std::collections::HashSet::with_capacity(header_cols.len());
    if let Some(duplicate) = header_cols
        .iter()
        .find(|column| !unique_headers.insert(column.as_str()))
    {
        return Err(EvalError::csv(
            path,
            1,
            format!("duplicate column '{duplicate}'"),
        ));
    }

    let subject_idx = header_cols
        .iter()
        .position(|c| c == "subject")
        .ok_or_else(|| EvalError::missing_field(path, "subject"))?;
    let question_idx = header_cols
        .iter()
        .position(|c| c == "question")
        .ok_or_else(|| EvalError::missing_field(path, "question"))?;
    let answer_idx = header_cols
        .iter()
        .position(|c| c == "answer")
        .ok_or_else(|| EvalError::missing_field(path, "answer"))?;

    if subject_idx == question_idx || subject_idx == answer_idx || question_idx == answer_idx {
        return Err(EvalError::csv(
            path,
            1,
            "subject, question, and answer must refer to distinct columns",
        ));
    }

    // Choice columns are everything between subject/question and answer, in
    // header order. This is robust to choice columns being named A/B/C/D,
    // choice1/2/3/4, or any other single-letter / column-name scheme.
    let mut choice_indices: Vec<usize> = (0..header_cols.len())
        .filter(|i| *i != subject_idx && *i != question_idx && *i != answer_idx)
        .collect();
    choice_indices.sort();
    if choice_indices.is_empty() {
        return Err(EvalError::csv(path, 1, "no choice columns found"));
    }

    let max_idx = [subject_idx, question_idx, answer_idx]
        .iter()
        .chain(choice_indices.iter())
        .copied()
        .max()
        .unwrap_or(0);

    let mut tasks = Vec::new();
    // Data rows start at line 2 (header is line 1). The csv crate reports
    // its own record positions which we map back to 1-based file line numbers.
    for (record_idx, record) in reader.records().enumerate() {
        let line_no = record_idx + 2;
        let record = record.map_err(|e| {
            EvalError::csv(path, line_no, format!("malformed CSV row: {e}"))
        })?;
        // Skip rows that are entirely empty (every field empty/whitespace).
        if record.iter().all(|f| f.trim().is_empty()) {
            continue;
        }
        if record.len() <= max_idx {
            return Err(EvalError::csv(
                path,
                line_no,
                format!(
                    "row has {} fields, expected at least {}",
                    record.len(),
                    max_idx + 1
                ),
            ));
        }
        let subject = record.get(subject_idx).unwrap_or("").trim();
        let prompt = record.get(question_idx).unwrap_or("").trim().to_string();
        let answer = record.get(answer_idx).unwrap_or("").trim().to_string();
        if answer.is_empty() {
            return Err(EvalError::malformed(path, line_no, "answer field is empty"));
        }
        let answer_letter = answer
            .chars()
            .next()
            .map(|c| c.to_ascii_uppercase())
            .unwrap_or('?');
        if answer_letter < 'A' || answer_letter > 'Z' {
            return Err(EvalError::malformed(
                path,
                line_no,
                format!("answer '{answer}' is not a letter"),
            ));
        }
        let expected_idx = (answer_letter as u8 - b'A') as usize;
        if expected_idx >= choice_indices.len() {
            return Err(EvalError::malformed(
                path,
                line_no,
                format!(
                    "answer '{answer}' is out of range for {} choices",
                    choice_indices.len()
                ),
            ));
        }
        let choices: Vec<String> = choice_indices
            .iter()
            .map(|i| record.get(*i).unwrap_or("").trim().to_string())
            .collect();

        let letter_labels: Vec<String> = (0..choices.len())
            .map(|i| char::from(b'A' + i as u8).to_string())
            .collect();
        let labeled: Vec<String> = letter_labels
            .iter()
            .zip(choices.iter())
            .map(|(label, value)| format!("{}) {}", label, value))
            .collect();
        let full_prompt = format!("{}\n{}\nAnswer:", prompt, labeled.join("\n"));

        tasks.push(TaskSpec {
            id: format!("mmlu_{}_{}", subject, line_no - 1),
            suite: Suite::MMLU,
            prompt: full_prompt,
            expected: Some(answer),
            choices,
            // MMLU is multiple-choice, so criteria-based scoring is disabled.
            criteria: None,
        });
    }

    if tasks.is_empty() {
        return Err(EvalError::malformed(path, 1, "no data rows found"));
    }

    tasks.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(tasks)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_CSV: &str = "subject,question,A,B,C,D,answer\n\
anatomy,The heart is located in which cavity?,Cranial,Thoracic,Abdominal,Pelvic,B\n";

    #[test]
    fn parses_minimal_csv_with_provenance() {
        let dataset = load_csv_bytes(MINIMAL_CSV.as_bytes(), "test.csv", "v1", "test").unwrap();
        assert_eq!(dataset.suite(), Suite::MMLU);
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset[0].id, "mmlu_anatomy_1");
        assert_eq!(
            dataset[0].choices,
            vec!["Cranial", "Thoracic", "Abdominal", "Pelvic"]
        );
        assert_eq!(dataset[0].expected.as_deref(), Some("B"));
        assert!(dataset[0].is_multiple_choice());
        // Provenance is computed from the bytes.
        assert_eq!(dataset.provenance().source, "test.csv");
        assert_eq!(dataset.provenance().source_revision, "v1");
        assert_eq!(dataset.provenance().split, "test");
        assert_eq!(dataset.provenance().task_count, 1);
        assert_eq!(dataset.provenance().sha256.len(), 64);
    }

    #[test]
    fn rejects_missing_header() {
        let bytes: &[u8] = b"";
        let err = load_csv_bytes(bytes, "x.csv", "v1", "test").unwrap_err();
        match err {
            EvalError::Csv { line, .. } => assert_eq!(line, 1),
            other => panic!("expected Csv error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_missing_required_columns() {
        // No "answer" column.
        let bytes = b"subject,question,A,B\nanatomy,Q,Cranial,Thoracic\n";
        let err = load_csv_bytes(bytes, "x.csv", "v1", "test").unwrap_err();
        match err {
            EvalError::MissingField { field, .. } => assert_eq!(field, "answer"),
            other => panic!("expected MissingField error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_no_choice_columns() {
        let bytes = b"subject,question,answer\nanatomy,Q,B\n";
        let err = load_csv_bytes(bytes, "x.csv", "v1", "test").unwrap_err();
        match err {
            EvalError::Csv { message, .. } => {
                assert!(message.contains("no choice"));
            }
            other => panic!("expected Csv error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_row_with_too_few_fields() {
        let bytes = b"subject,question,A,B,answer\nanatomy,Q,Cranial\n"; // row has only 3 fields
        let err = load_csv_bytes(bytes, "x.csv", "v1", "test").unwrap_err();
        match err {
            EvalError::Csv { line, .. } => assert_eq!(line, 2),
            other => panic!("expected Csv error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_answer_out_of_range() {
        let bytes = b"subject,question,A,B,answer\nanatomy,Q,Cranial,Thoracic,Z\n";
        let err = load_csv_bytes(bytes, "x.csv", "v1", "test").unwrap_err();
        match err {
            EvalError::Malformed { line, message, .. } => {
                assert_eq!(line, 2);
                assert!(message.contains("out of range"));
            }
            other => panic!("expected Malformed error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_letter_answer() {
        let bytes = b"subject,question,A,B,answer\nanatomy,Q,Cranial,Thoracic,1\n";
        let err = load_csv_bytes(bytes, "x.csv", "v1", "test").unwrap_err();
        match err {
            EvalError::Malformed { message, .. } => {
                assert!(message.contains("not a letter"));
            }
            other => panic!("expected Malformed error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_data() {
        let bytes = b"subject,question,A,B,answer\n"; // header only
        let err = load_csv_bytes(bytes, "x.csv", "v1", "test").unwrap_err();
        match err {
            EvalError::Malformed { .. } => {}
            other => panic!("expected Malformed error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_duplicate_columns() {
        let bytes = b"subject,subject,question,A,answer\ns1,s2,Q,Cranial,B\n";
        let err = load_csv_bytes(bytes, "x.csv", "v1", "test").unwrap_err();
        match err {
            EvalError::Csv { .. } => {}
            other => panic!("expected Csv error, got {other:?}"),
        }
    }

    #[test]
    fn multiple_rows_are_sorted_by_task_id() {
        let bytes = b"subject,question,A,B,answer\n\
physics,Force unit?,Joule,Newton,B\n\
anatomy,Heart cavity?,Cranial,Thoracic,B\n";
        let dataset = load_csv_bytes(bytes, "x.csv", "v1", "test").unwrap();
        let ids: Vec<&str> = dataset.iter().map(|t| t.id.as_str()).collect();
        // anatomy sorts before physics.
        assert_eq!(ids, vec!["mmlu_anatomy_2", "mmlu_physics_1"]);
    }

    #[test]
    fn parses_quoted_question_field_containing_comma() {
        // A question containing a literal comma must round-trip as one
        // field, not be split into extra columns.
        let bytes = b"subject,question,A,B,answer\n\
anatomy,\"Heart, lung, and bones form what?\",Cranial,Thoracic,B\n";
        let dataset = load_csv_bytes(bytes, "x.csv", "v1", "test").unwrap();
        assert_eq!(dataset.len(), 1);
        // The question must keep its embedded comma and span only column 1.
        assert!(dataset[0].prompt.contains("Heart, lung, and bones form what?"));
        assert_eq!(dataset[0].choices, vec!["Cranial", "Thoracic"]);
        assert_eq!(dataset[0].expected.as_deref(), Some("B"));
    }

    #[test]
    fn parses_quoted_choice_field_containing_comma_and_escaped_quote() {
        // A choice with both an embedded comma and an escaped quote ("")
        // must parse as one cell.
        let bytes = b"subject,question,A,B,answer\n\
anatomy,Where?,Cranial,\"Tho\"\"racic, ic\",B\n";
        let dataset = load_csv_bytes(bytes, "x.csv", "v1", "test").unwrap();
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset[0].choices, vec!["Cranial", "Tho\"racic, ic"]);
    }

    #[test]
    fn parses_crlf_line_endings() {
        // Windows-style CRLF line endings must be honored.
        let bytes = b"subject,question,A,B,answer\r\n\
anatomy,Heart cavity?,Cranial,Thoracic,B\r\n";
        let dataset = load_csv_bytes(bytes, "x.csv", "v1", "test").unwrap();
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset[0].id, "mmlu_anatomy_1");
        assert_eq!(dataset[0].choices, vec!["Cranial", "Thoracic"]);
        assert_eq!(dataset[0].expected.as_deref(), Some("B"));
    }

    #[test]
    fn rejects_unclosed_quote_in_field() {
        // An opening quote that is never closed must produce a structured
        // Csv error pointing at the offending line.
        let bytes = b"subject,question,A,B,answer\n\
anatomy,\"unterminated question,B\n";
        let err = load_csv_bytes(bytes, "x.csv", "v1", "test").unwrap_err();
        match err {
            EvalError::Csv { line, .. } => assert_eq!(line, 2),
            other => panic!("expected Csv error, got {other:?}"),
        }
    }
}
