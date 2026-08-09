// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ap cockpit read` subcommand — render the cockpit NDJSON log as a
//! per-repo × per-cluster grade-colored table.
//!
//! Reads `~/.agileplus/cockpit.ndjson` (override with `--input`). Each
//! line is a JSON record emitted by `ap cockpit publish`:
//!
//! ```json
//! {"ts":"epoch:...","repo":"agileplus","cluster":"C03","score":2,"max":3,"grade":"B","probes":3}
//! ```
//!
//! Lines that fail to parse are skipped and counted (printed at the
//! bottom of the table). Empty files print a friendly "no scores yet"
//! hint and exit 0.
//!
//! `--filter-repo <name>` filters to a single repo; `--watch` polls the file
//! every 2s and re-renders in place (v1 uses polling; inotify is a
//! future enhancement).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead as _, BufReader};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Args;
use serde::Deserialize;

/// Polling interval for `--watch` (v1 polls; inotify is a future enhancement).
const WATCH_INTERVAL: Duration = Duration::from_secs(2);

/// One NDJSON record from the cockpit log. Field set matches the
/// `ap cockpit publish` writer so the reader stays in lock-step.
#[derive(Debug, Clone, Deserialize)]
pub struct CockpitRecord {
    #[serde(default)]
    pub ts: String,
    pub repo: String,
    pub cluster: String,
    #[serde(default)]
    pub score: u32,
    #[serde(default)]
    pub max: u32,
    #[serde(default)]
    pub grade: String,
    #[serde(default)]
    pub probes: u32,
}

/// Args for `ap cockpit read`. Defaults the log path to
/// `~/.agileplus/cockpit.ndjson` (same resolution as the publisher).
#[derive(Debug, Args)]
pub struct CockpitReadArgs {
    /// Restrict output to a single repo (matches `repo` field exactly).
    #[arg(long, value_name = "NAME")]
    pub filter_repo: Option<String>,

    /// Override the NDJSON log path. Defaults to `~/.agileplus/cockpit.ndjson`.
    #[arg(long, value_name = "PATH")]
    pub input: Option<PathBuf>,

    /// Re-render every 2s when the log file changes.
    #[arg(long)]
    pub watch: bool,
}

/// Reader entry point — wired from `commands::cockpit::run`'s dispatch.
/// Kept here (separate from the publisher) so the two surfaces can
/// evolve independently.
pub fn run(args: &CockpitReadArgs) -> Result<()> {
    let path = match &args.input {
        Some(p) => p.clone(),
        None => default_log_path()?,
    };

    if args.watch {
        watch_loop(&path, args.filter_repo.as_deref())
    } else {
        render_once(&path, args.filter_repo.as_deref())
    }
}

fn default_log_path() -> Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("resolving $HOME for cockpit log path")?;
    Ok(home.join(".agileplus").join("cockpit.ndjson"))
}

/// Parse the log file, skip malformed lines with a counter, render the
/// per-repo × per-cluster table. Exits 0 even when nothing was found —
/// the absence of scores is not an error, it's a steady state.
fn render_once(path: &Path, repo_filter: Option<&str>) -> Result<()> {
    let (records, skipped) = parse_log(path)?;
    print_render(records, skipped, repo_filter);
    Ok(())
}

/// Watch loop: poll the log every `WATCH_INTERVAL`, re-render on
/// change. Use polling for v1 (broad UNIX-y compat without depending
/// on `inotify`/`kqueue`).
fn watch_loop(path: &Path, repo_filter: Option<&str>) -> Result<()> {
    // Initial render, even if missing — establishes the layout.
    let (records, skipped) = parse_log(path).unwrap_or_default();
    print_render(records, skipped, repo_filter);

    let mut last_mtime = std::fs::metadata(path).and_then(|m| m.modified()).ok();

    loop {
        std::thread::sleep(WATCH_INTERVAL);
        let mtime = std::fs::metadata(path).and_then(|m| m.modified()).ok();
        if mtime == last_mtime {
            continue;
        }
        last_mtime = mtime;
        // Move cursor to top-left, clear screen, re-render.
        eprint!("\x1b[2J\x1b[H");
        let (records, skipped) = parse_log(path).unwrap_or_default();
        print_render(records, skipped, repo_filter);
    }
}

/// Read the NDJSON file line-by-line. Returns `(records, skipped_count)`.
/// If the file does not exist, returns the empty record set and 0
/// skipped — a missing log is the common cold-start state and should
/// not error.
fn parse_log(path: &Path) -> Result<(Vec<CockpitRecord>, usize)> {
    if !path.exists() {
        return Ok((Vec::new(), 0));
    }
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(f);
    let mut records = Vec::new();
    let mut skipped = 0usize;
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<CockpitRecord>(trimmed) {
            Ok(rec) => records.push(rec),
            Err(_) => skipped += 1,
        }
    }
    Ok((records, skipped))
}

/// ANSI color codes for the grade column. Inline here to keep the dep
/// count flat — `owo-colors`/`yansi` aren't in the workspace today.
const ANSI_RESET: &str = "\x1b[0m";
const ANSI_BOLD: &str = "\x1b[1m";
const ANSI_RED: &str = "\x1b[31m";
const ANSI_GREEN: &str = "\x1b[32m";
const ANSI_YELLOW: &str = "\x1b[33m";
const ANSI_DIM: &str = "\x1b[2m";

/// Pick a color for a single-letter grade. Returns the bare letter so
/// the table layout stays monospace-predictable.
fn color_grade(grade: &str) -> String {
    let prefix = match grade {
        "A" | "B" => ANSI_GREEN,
        "C" => ANSI_YELLOW,
        "D" | "F" => ANSI_RED,
        _ => "",
    };
    if prefix.is_empty() {
        grade.to_string()
    } else {
        format!("{prefix}{grade}{ANSI_RESET}")
    }
}

/// Aggregate the records into a stable, sorted view: repos in lexical
/// order, clusters in lexical order within each repo.
type AggregatedRow =
    BTreeMap<String /* repo */, BTreeMap<String /* cluster */, CockpitRecord>>;

fn aggregate(records: Vec<CockpitRecord>) -> AggregatedRow {
    let mut map: AggregatedRow = BTreeMap::new();
    for r in records {
        map.entry(r.repo.clone())
            .or_default()
            .insert(r.cluster.clone(), r);
    }
    map
}

/// Compute a composite "%" completion = mean of `score / max` across
/// visible clusters. Returns 0..=100.
fn composite_pct(row: &BTreeMap<String, CockpitRecord>) -> u32 {
    let (sum_score, sum_max) = row
        .values()
        .filter(|r| r.max > 0)
        .fold((0u64, 0u64), |(s, m), r| {
            (s + r.score as u64, m + r.max as u64)
        });
    if sum_max == 0 {
        return 0;
    }
    ((sum_score * 100) / sum_max) as u32
}

/// Map composite % back to a letter grade so the table foot prints
/// the same A/B/C/D/F cutoffs the publisher used.
fn composite_grade(pct: u32) -> &'static str {
    match pct {
        90..=100 => "A",
        75..=89 => "B",
        60..=74 => "C",
        40..=59 => "D",
        _ => "F",
    }
}

/// Render the table. `records` is the parsed NDJSON; `skipped` is the
/// malformed-line count; `repo_filter` optionally narrows to one repo.
fn print_render(records: Vec<CockpitRecord>, skipped: usize, repo_filter: Option<&str>) {
    let filtered: Vec<CockpitRecord> = match repo_filter {
        Some(name) => records.into_iter().filter(|r| r.repo == name).collect(),
        None => records,
    };

    if filtered.is_empty() {
        if skipped > 0 {
            println!("no scores yet (skipped {skipped} malformed line(s))");
        } else {
            println!("no scores yet — run `ap cockpit publish --repo <path>` first");
        }
        return;
    }

    let agg = aggregate(filtered);

    // Build a stable set of cluster columns across all visible repos so
    // multi-repo tables line up column-wise.
    let mut cluster_cols: Vec<String> = agg
        .values()
        .flat_map(|r| r.keys().cloned())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    // Spec example expects C00..C11 order. Sort lexicographically
    // (C00 < C01 < ... < C11) which matches numeric-as-string order
    // for these two-digit cluster ids.
    cluster_cols.sort();

    // Header.
    print!("{ANSI_BOLD}{:<14}", "REPO");
    for c in &cluster_cols {
        print!(" {}", c);
    }
    println!("  {:>5}  GRADE{ANSI_RESET}", "COMP");

    // Rows.
    for (repo, clusters) in &agg {
        let pct = composite_pct(clusters);
        let grade_letter = composite_grade(pct);
        let grade_colored = color_grade(grade_letter);
        print!("{:<14}", repo);
        for c in &cluster_cols {
            let cell = match clusters.get(c) {
                Some(r) => color_grade(&r.grade),
                None => format!("{ANSI_DIM}-{ANSI_RESET}"),
            };
            // Pad raw text width to 3 (each cell is grade letter or "-").
            print!(" {:>3}", strip_ansi(&cell));
        }
        println!("  {:>4}%  {}", pct, grade_colored);
    }

    // Footer with skipped-line counter when any.
    if skipped > 0 {
        println!("{ANSI_DIM}(skipped {skipped} malformed line(s)){ANSI_RESET}");
    }
}

/// Strip ANSI escapes so we can compute the printable width of a cell
/// before padding it. Used only for layout math — the original cell
/// string still carries its color codes when it goes to the terminal.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            // Skip until 'm' (the end of an SGR sequence).
            for c in chars.by_ref() {
                if c == 'm' {
                    break;
                }
            }
        } else {
            out.push(ch);
        }
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(repo: &str, cluster: &str, score: u32, max: u32, grade: &str) -> CockpitRecord {
        CockpitRecord {
            ts: "epoch:0".into(),
            repo: repo.into(),
            cluster: cluster.into(),
            score,
            max,
            grade: grade.into(),
            probes: 0,
        }
    }

    #[test]
    fn color_grade_maps_correct_bands() {
        assert_eq!(color_grade("A"), format!("{ANSI_GREEN}A{ANSI_RESET}"));
        assert_eq!(color_grade("B"), format!("{ANSI_GREEN}B{ANSI_RESET}"));
        assert_eq!(color_grade("C"), format!("{ANSI_YELLOW}C{ANSI_RESET}"));
        assert_eq!(color_grade("D"), format!("{ANSI_RED}D{ANSI_RESET}"));
        assert_eq!(color_grade("F"), format!("{ANSI_RED}F{ANSI_RESET}"));
        // Unknown grade: pass through with no color.
        assert_eq!(color_grade("?"), "?");
    }

    #[test]
    fn aggregate_groups_records_by_repo_and_cluster() {
        let recs = vec![
            rec("AgilePlus", "C00", 1, 3, "D"),
            rec("AgilePlus", "C01", 3, 3, "A"),
            rec("Tracera", "C00", 2, 3, "C"),
        ];
        let agg = aggregate(recs);
        assert!(agg.contains_key("AgilePlus"));
        assert!(agg.contains_key("Tracera"));
        assert_eq!(agg["AgilePlus"]["C00"].score, 1);
        assert_eq!(agg["Tracera"]["C00"].grade, "C");
    }

    #[test]
    fn composite_pct_averages_score_over_max() {
        // 2+2+2 out of 3+3+3 = 6/9 = 66%
        let mut row: BTreeMap<String, CockpitRecord> = BTreeMap::new();
        row.insert("C00".into(), rec("r", "C00", 2, 3, "C"));
        row.insert("C01".into(), rec("r", "C01", 2, 3, "C"));
        row.insert("C02".into(), rec("r", "C02", 2, 3, "C"));
        assert_eq!(composite_pct(&row), 66);
    }

    #[test]
    fn composite_pct_handles_zero_max() {
        let row: BTreeMap<String, CockpitRecord> = BTreeMap::new();
        assert_eq!(composite_pct(&row), 0);
    }

    #[test]
    fn composite_grade_matches_publisher_bands() {
        // Mirror the publisher's `grade_for` cutoffs so the column
        // stays honest.
        assert_eq!(composite_grade(100), "A");
        assert_eq!(composite_grade(90), "A");
        assert_eq!(composite_grade(89), "B");
        assert_eq!(composite_grade(75), "B");
        assert_eq!(composite_grade(74), "C");
        assert_eq!(composite_grade(60), "C");
        assert_eq!(composite_grade(59), "D");
        assert_eq!(composite_grade(40), "D");
        assert_eq!(composite_grade(39), "F");
        assert_eq!(composite_grade(0), "F");
    }

    #[test]
    fn parse_log_skips_malformed_lines_and_keeps_valid_ones() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cockpit.ndjson");
        let body = concat!(
            // Valid record (line 1).
            "{\"ts\":\"epoch:1\",\"repo\":\"AgilePlus\",\"cluster\":\"C00\",\"score\":2,\"max\":3,\"grade\":\"C\",\"probes\":0}\n",
            // Malformed: not JSON.
            "this is not json\n",
            // Valid record (line 3).
            "{\"ts\":\"epoch:2\",\"repo\":\"AgilePlus\",\"cluster\":\"C01\",\"score\":3,\"max\":3,\"grade\":\"A\",\"probes\":1}\n",
            // Malformed: missing required `repo`.
            "{\"ts\":\"epoch:3\",\"cluster\":\"C02\",\"score\":1,\"max\":3,\"grade\":\"D\",\"probes\":0}\n",
        );
        std::fs::write(&path, body).expect("write");
        let (records, skipped) = parse_log(&path).expect("parse");
        assert_eq!(
            records.len(),
            2,
            "expected 2 valid records, got {}",
            records.len()
        );
        assert_eq!(skipped, 2, "expected 2 skipped lines, got {skipped}");
        assert_eq!(records[0].cluster, "C00");
        assert_eq!(records[1].grade, "A");
    }

    #[test]
    fn parse_log_returns_empty_for_missing_file() {
        let path = std::path::Path::new("/tmp/agileplus-nonexistent-cockpit.ndjson");
        // Make sure it really doesn't exist (a sibling test could have
        // created it; use a unique-looking name).
        let _ = std::fs::remove_file(path);
        let (records, skipped) = parse_log(path).expect("parse on missing file");
        assert!(records.is_empty());
        assert_eq!(skipped, 0);
    }

    #[test]
    fn strip_ansi_removes_color_codes_for_width_math() {
        let s = format!("{ANSI_GREEN}A{ANSI_RESET}");
        assert_eq!(strip_ansi(&s), "A");
        assert_eq!(strip_ansi("plain"), "plain");
        // Multi-sequence.
        let s2 = format!("{ANSI_BOLD}REPO{ANSI_RESET}");
        assert_eq!(strip_ansi(&s2), "REPO");
    }
}
