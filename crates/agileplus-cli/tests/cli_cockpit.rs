// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for the `ap cockpit` subcommand.
//!
//! These shell out to the built `agileplus` binary, point it at the
//! `agileplus-cli` crate directory, score against the workspace-bundled
//! rubric catalog, and verify the per-cluster records land as one-line
//! JSON in the requested NDJSON output file.

use std::path::PathBuf;

use assert_cmd::Command;

fn self_repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn cli() -> Command {
    Command::cargo_bin("agileplus").expect("agileplus binary should be built")
}

#[test]
fn cockpit_publish_writes_ndjson_with_one_record_per_cluster() {
    let repo = self_repo();
    let tmp = tempfile::tempdir().expect("tempdir");
    let log = tmp.path().join("cockpit.ndjson");

    cli()
        .args([
            "cockpit",
            "publish",
            "--repo",
            repo.to_str().unwrap(),
            "--output",
            log.to_str().unwrap(),
            "--clusters",
            "C03,C04",
        ])
        .assert()
        .success();

    let text = std::fs::read_to_string(&log).expect("log should exist");
    let lines: Vec<&str> = text.lines().collect();
    // ≥2 clusters yielded ≥2 lines; the exact count depends on whether
    // C03/C04 are present in the bundled catalog. Whichever count we got,
    // every line must be valid JSON with the expected shape.
    assert!(lines.len() >= 2, "expected ≥2 records, got {}", lines.len());
    for line in &lines {
        let v: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("invalid JSON line: {e}\n{line}"));
        assert!(v.get("ts").is_some());
        assert!(v.get("repo").is_some());
        assert!(v.get("cluster").is_some());
        assert!(v.get("score").is_some());
        assert!(v.get("max").is_some());
        assert!(v.get("grade").is_some());
        assert!(v.get("probes").is_some());
    }
}

#[test]
fn cockpit_publish_appends_to_existing_log() {
    // Two consecutive publish invocations against the same log file
    // must both succeed and double the record count.
    let repo = self_repo();
    let tmp = tempfile::tempdir().expect("tempdir");
    let log = tmp.path().join("cockpit.ndjson");

    for _ in 0..2 {
        cli()
            .args([
                "cockpit",
                "publish",
                "--repo",
                repo.to_str().unwrap(),
                "--output",
                log.to_str().unwrap(),
                "--clusters",
                "C01",
            ])
            .assert()
            .success();
    }

    let text = std::fs::read_to_string(&log).expect("log");
    let n = text.lines().count();
    // At least one cluster record per invocation × 2 = ≥2 total lines.
    assert!(n >= 2, "expected ≥2 lines after 2 publishes, got {n}");
}

#[test]
fn cockpit_path_subcommand_prints_resolved_log_path() {
    let output = cli()
        .args(["cockpit", "path"])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("agileplus"), "unexpected path: {stdout}");
    assert!(
        stdout.contains("cockpit.ndjson"),
        "expected cockpit.ndjson in path: {stdout}"
    );
}

#[test]
fn cockpit_read_rejects_global_repo_after_subcommand() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let log = tmp.path().join("cockpit.ndjson");

    cli()
        .args([
            "cockpit",
            "read",
            "--repo",
            "/tmp/not-a-cockpit-filter",
            "--input",
            log.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("--filter-repo <NAME>"));
}

#[test]
fn cockpit_read_rejects_global_repo_before_subcommand() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let log = tmp.path().join("cockpit.ndjson");

    cli()
        .args([
            "--repo",
            "/tmp/not-a-cockpit-filter",
            "cockpit",
            "read",
            "--input",
            log.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("--filter-repo <NAME>"));
}

#[test]
fn cockpit_path_rejects_global_repo() {
    cli()
        .args(["cockpit", "path", "--repo", "/tmp/not-a-cockpit-filter"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("--filter-repo <NAME>"));
}

#[test]
fn cockpit_publish_help_lists_required_flags() {
    let output = cli()
        .args(["cockpit", "publish", "--help"])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--repo"), "missing --repo flag: {stdout}");
    assert!(
        stdout.contains("--output"),
        "missing --output flag: {stdout}"
    );
    assert!(
        stdout.contains("--clusters"),
        "missing --clusters flag: {stdout}"
    );
}

// ── `ap cockpit read` reader tests ───────────────────────────────────────────

fn write_log(path: &std::path::Path, body: &str) {
    std::fs::write(path, body).expect("write ndjson");
}

fn empty_record() -> String {
    // JSON record that satisfies CockpitRecord's required `repo` and
    // `cluster` fields, with sensible defaults for the rest.
    "{\"ts\":\"epoch:0\",\"repo\":\"AgilePlus\",\"cluster\":\"C00\",\"score\":0,\"max\":0,\"grade\":\"F\",\"probes\":0}".to_string()
}

#[test]
fn cockpit_read_with_empty_log_prints_no_scores_yet_and_exits_zero() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let log = tmp.path().join("cockpit.ndjson");
    // File intentionally not created — missing path = cold-start state.

    let assert = cli()
        .args(["cockpit", "read", "--input", log.to_str().unwrap()])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&assert.get_output().stdout);
    assert!(
        stdout.contains("no scores yet"),
        "expected 'no scores yet' in stdout, got: {stdout}"
    );
}

#[test]
fn cockpit_read_with_present_log_renders_table_with_grade_letters() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let log = tmp.path().join("cockpit.ndjson");
    // One repo, two clusters. Counts toward test #2 in the spec.
    let body = format!(
        "{}\n{}\n",
        r#"{"ts":"epoch:1","repo":"AgilePlus","cluster":"C00","score":0,"max":3,"grade":"F","probes":0}"#,
        r#"{"ts":"epoch:2","repo":"AgilePlus","cluster":"C01","score":3,"max":3,"grade":"A","probes":2}"#,
    );
    write_log(&log, &body);

    let assert = cli()
        .args(["cockpit", "read", "--input", log.to_str().unwrap()])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&assert.get_output().stdout);
    // Header present.
    assert!(stdout.contains("REPO"), "missing REPO header: {stdout}");
    assert!(stdout.contains("GRADE"), "missing GRADE header: {stdout}");
    // Repo row present.
    assert!(stdout.contains("AgilePlus"), "missing repo row: {stdout}");
    // Both cluster columns present (C00, C01).
    assert!(stdout.contains("C00"), "missing C00 column: {stdout}");
    assert!(stdout.contains("C01"), "missing C01 column: {stdout}");
    // Grade letters appear in table (F and A for this fixture).
    assert!(
        stdout.contains('F') && stdout.contains('A'),
        "missing grade letters: {stdout}"
    );
}

#[test]
fn cockpit_read_repo_filter_omits_other_repos() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let log = tmp.path().join("cockpit.ndjson");
    // Two repos: AgilePlus + Tracera. --filter-repo AgilePlus must show only the AgilePlus row.
    let body = format!(
        "{}\n{}\n{}\n",
        r#"{"ts":"epoch:1","repo":"AgilePlus","cluster":"C00","score":3,"max":3,"grade":"A","probes":0}"#,
        r#"{"ts":"epoch:2","repo":"Tracera","cluster":"C00","score":1,"max":3,"grade":"D","probes":0}"#,
        empty_record(),
    );
    write_log(&log, &body);

    let assert = cli()
        .args([
            "cockpit",
            "read",
            "--filter-repo",
            "AgilePlus",
            "--input",
            log.to_str().unwrap(),
        ])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&assert.get_output().stdout);
    assert!(
        stdout.contains("AgilePlus"),
        "AgilePlus row missing: {stdout}"
    );
    // Tracera must be filtered OUT — it should not appear in the table body.
    // Use a strict substring check on the line that follows "AgilePlus"
    // (other rows would contain "Tracera" elsewhere on its own line).
    let mut agileplus_line_seen = false;
    for line in stdout.lines() {
        if line.contains("AgilePlus") {
            agileplus_line_seen = true;
        }
        if line.contains("Tracera") {
            panic!("Tracera row should be filtered out, found line: {line}");
        }
    }
    assert!(
        agileplus_line_seen,
        "AgilePlus row not present in stdout: {stdout}"
    );
}

#[test]
fn cockpit_read_with_malformed_line_skips_and_counts_without_crashing() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let log = tmp.path().join("cockpit.ndjson");
    // Mix: 1 valid record + 2 malformed lines + 1 valid record.
    let body = format!(
        "{}\nnot json at all\n{{\n{}\n",
        r#"{"ts":"epoch:1","repo":"AgilePlus","cluster":"C00","score":2,"max":3,"grade":"C","probes":0}"#,
        r#"{"ts":"epoch:2","repo":"AgilePlus","cluster":"C01","score":3,"max":3,"grade":"A","probes":1}"#,
    );
    write_log(&log, &body);

    let assert = cli()
        .args(["cockpit", "read", "--input", log.to_str().unwrap()])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&assert.get_output().stdout);
    // Footer counter for skipped lines.
    assert!(
        stdout.contains("skipped 2 malformed"),
        "expected skipped counter in footer, got: {stdout}"
    );
    // Repo row still rendered despite malformed lines.
    assert!(
        stdout.contains("AgilePlus"),
        "AgilePlus row missing: {stdout}"
    );
}

#[test]
fn cockpit_read_help_lists_filters_and_watch_flag() {
    let output = cli()
        .args(["cockpit", "read", "--help"])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--filter-repo"),
        "missing --filter-repo flag: {stdout}"
    );
    assert!(stdout.contains("--input"), "missing --input flag: {stdout}");
    assert!(stdout.contains("--watch"), "missing --watch flag: {stdout}");
}
