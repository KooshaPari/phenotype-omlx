//! Internal CSV parser for the MMLU loader.
//!
//! This module is private to [`super`] and contains the row parser
//! ([`parse_csv`]) that walks a CSV byte stream into [`TaskSpec`]s. It is
//! separated from the public loader wrappers so the row-level logic can be
//! reasoned about and unit-tested independently from the I/O / provenance
//! wrapping the loaders perform.
//!
//! The parser uses the `csv` crate so quoted fields containing commas,
//! escaped quotes (`""`), and CRLF line endings are handled per RFC 4180.
//! Row numbers in errors are 1-based and refer to the data row position in
//! the source file (the header is line 1).

use crate::{EvalError, Result, Suite, TaskSpec};

/// Parse a CSV string into MMLU tasks. Exposed for callers that already hold
/// the bytes; the path is only used in error messages.
pub(crate) fn parse_csv(content: &str, path: &str) -> Result<Vec<TaskSpec>> {
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
    //! Parser-internal tests. These exercise `parse_csv` directly so the row
    //! parser is covered independently from the I/O wrapping in [`super`].

    use super::*;

    const MINIMAL_CSV: &str = "subject,question,A,B,C,D,answer\n\
anatomy,The heart is located in which cavity?,Cranial,Thoracic,Abdominal,Pelvic,B\n";

    #[test]
    fn parse_minimal_csv_returns_one_task() {
        let tasks = parse_csv(MINIMAL_CSV, "test.csv").unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, "mmlu_anatomy_1");
        assert_eq!(tasks[0].suite, Suite::MMLU);
        assert_eq!(
            tasks[0].choices,
            vec!["Cranial", "Thoracic", "Abdominal", "Pelvic"]
        );
        assert_eq!(tasks[0].expected.as_deref(), Some("B"));
        assert!(tasks[0].is_multiple_choice());
        // The prompt stitches the question + labeled choices together.
        assert!(tasks[0].prompt.starts_with("The heart is located in which cavity?\nA) Cranial"));
        assert!(tasks[0].prompt.ends_with("Answer:"));
        assert!(tasks[0].criteria.is_none());
    }

    #[test]
    fn parse_distinct_subject_question_and_answer_indices() {
        // subject, question, answer at 0, 1, 2; choice columns at 3, 4.
        let csv = "subject,question,A,B,answer\ns,q,x,y,A\n";
        let tasks = parse_csv(csv, "x.csv").unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].choices, vec!["x", "y"]);
        assert_eq!(tasks[0].expected.as_deref(), Some("A"));
    }

    #[test]
    fn parse_choice_columns_preserve_header_order() {
        // Header order is `B, A` so choice indices must be reported in
        // header order, not alphabetical order.
        let csv = "subject,question,B,A,answer\ns,q,Y-choice,X-choice,A\n";
        let tasks = parse_csv(csv, "x.csv").unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].choices, vec!["Y-choice", "X-choice"]);
        // And the labels A/B are assigned in header order.
        assert!(tasks[0].prompt.contains("A) Y-choice"));
        assert!(tasks[0].prompt.contains("B) X-choice"));
    }

    #[test]
    fn parse_preserves_many_choice_columns() {
        let csv = "subject,question,A,B,C,D,E,F,answer\ns,q,a,b,c,d,e,f,F\n";
        let tasks = parse_csv(csv, "x.csv").unwrap();
        assert_eq!(
            tasks[0].choices,
            vec!["a", "b", "c", "d", "e", "f"]
        );
        assert_eq!(tasks[0].expected.as_deref(), Some("F"));
    }

    #[test]
    fn parse_rejects_empty_header() {
        let err = parse_csv("", "x.csv").unwrap_err();
        match err {
            EvalError::Csv { line, message, .. } => {
                assert_eq!(line, 1);
                assert!(message.contains("missing header") || message.contains("failed to read header"));
            }
            other => panic!("expected Csv error, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_duplicate_columns() {
        let err = parse_csv("subject,subject,question,A,answer\ns1,s2,Q,x,B\n", "x.csv")
            .unwrap_err();
        match err {
            EvalError::Csv { message, .. } => assert!(message.contains("duplicate")),
            other => panic!("expected Csv error, got {other:?}"),
        }
    }

    // The `subject_idx == question_idx` belt-and-suspenders check inside
    // `parse_csv` is structurally unreachable from well-formed CSV (distinct
    // header names map to distinct positions), so it's intentionally not
    // covered by a dedicated test. The duplicate-column path above
    // (`parse_rejects_duplicate_columns`) is the user-visible safety net.

    #[test]
    fn parse_rejects_missing_required_column() {
        // No answer column.
        let err = parse_csv("subject,question,A,B\ns,q,x,y\n", "x.csv").unwrap_err();
        match err {
            EvalError::MissingField { field, .. } => assert_eq!(field, "answer"),
            other => panic!("expected MissingField error, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_no_choice_columns() {
        // No columns between subject/question and answer.
        let err = parse_csv("subject,question,answer\ns,q,B\n", "x.csv").unwrap_err();
        match err {
            EvalError::Csv { message, .. } => assert!(message.contains("no choice")),
            other => panic!("expected Csv error, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_short_row() {
        // Row 2 has only 3 fields but max_idx is 4.
        let err = parse_csv("subject,question,A,B,answer\ns,q,x\n", "x.csv").unwrap_err();
        match err {
            EvalError::Csv { line, .. } => assert_eq!(line, 2),
            other => panic!("expected Csv error, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_empty_answer() {
        // answer column is present but blank.
        let err = parse_csv("subject,question,A,B,answer\ns,q,x,y,\n", "x.csv").unwrap_err();
        match err {
            EvalError::Malformed { line, message, .. } => {
                assert_eq!(line, 2);
                assert!(message.contains("empty"));
            }
            other => panic!("expected Malformed error, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_answer_letter_out_of_range() {
        let err = parse_csv("subject,question,A,B,answer\ns,q,x,y,Z\n", "x.csv").unwrap_err();
        match err {
            EvalError::Malformed { line, message, .. } => {
                assert_eq!(line, 2);
                assert!(message.contains("out of range"));
            }
            other => panic!("expected Malformed error, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_non_letter_answer() {
        let err = parse_csv("subject,question,A,B,answer\ns,q,x,y,1\n", "x.csv").unwrap_err();
        match err {
            EvalError::Malformed { message, .. } => assert!(message.contains("not a letter")),
            other => panic!("expected Malformed error, got {other:?}"),
        }
    }

    #[test]
    fn parse_skips_blank_data_rows() {
        // Row 2 is entirely empty and must be skipped without error. The
        // csv crate reports only non-blank records, so the surviving anatomy
        // row gets `record_idx=0` -> `line_no=2` -> id suffix
        // `line_no-1 = 1`.
        let csv = "subject,question,A,B,answer\n\
\n\
anatomy,Q?,x,y,B\n";
        let tasks = parse_csv(csv, "x.csv").unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, "mmlu_anatomy_1");
    }

    #[test]
    fn parse_rejects_when_only_blank_rows_after_header() {
        // Header followed by only blank rows -> empty data -> Malformed.
        let err = parse_csv("subject,question,A,B,answer\n\n\n", "x.csv").unwrap_err();
        match err {
            EvalError::Malformed { message, .. } => assert!(message.contains("no data")),
            other => panic!("expected Malformed error, got {other:?}"),
        }
    }

    #[test]
    fn parse_returns_tasks_sorted_by_task_id() {
        let csv = "subject,question,A,B,answer\n\
physics,Force?,Joule,Newton,B\n\
anatomy,Cavity?,Cranial,Thoracic,B\n";
        let tasks = parse_csv(csv, "x.csv").unwrap();
        assert_eq!(
            tasks.iter().map(|t| t.id.as_str()).collect::<Vec<_>>(),
            vec!["mmlu_anatomy_2", "mmlu_physics_1"]
        );
    }

    #[test]
    fn parse_quoted_question_preserves_embedded_comma() {
        let csv = "subject,question,A,B,answer\n\
s,\"a, b, c?\",x,y,B\n";
        let tasks = parse_csv(csv, "x.csv").unwrap();
        assert_eq!(tasks.len(), 1);
        assert!(tasks[0].prompt.contains("a, b, c?"));
    }

    #[test]
    fn parse_quoted_choice_handles_escaped_quote_and_comma() {
        let csv = "subject,question,A,B,answer\n\
s,Q,Cranial,\"Tho\"\"racic, ic\",B\n";
        let tasks = parse_csv(csv, "x.csv").unwrap();
        assert_eq!(tasks[0].choices, vec!["Cranial", "Tho\"racic, ic"]);
    }

    #[test]
    fn parse_crlf_endings_work() {
        let csv = "subject,question,A,B,answer\r\n\
anatomy,Cavity?,Cranial,Thoracic,B\r\n";
        let tasks = parse_csv(csv, "x.csv").unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, "mmlu_anatomy_1");
    }

    #[test]
    fn parse_rejects_unclosed_quote() {
        let csv = "subject,question,A,B,answer\n\
s,\"unterminated,x,y,B\n";
        let err = parse_csv(csv, "x.csv").unwrap_err();
        match err {
            EvalError::Csv { line, .. } => assert_eq!(line, 2),
            other => panic!("expected Csv error, got {other:?}"),
        }
    }
}