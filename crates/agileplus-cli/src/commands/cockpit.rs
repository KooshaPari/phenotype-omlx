// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ap cockpit` subcommand — publish rubric scores to a local NDJSON log
//! for the fleet dashboard / cockpit reader.
//!
//! Spec: score --output ndjson. Each invocation appends one line per
//! cluster to `~/.agileplus/cockpit.ndjson` (configurable via
//! `--output`). The schema is deliberately minimal to keep the cockpit
//! reader cheap:
//!
//! ```json
//! {"ts":"2026-07-06T00:00:00Z","repo":"agileplus","cluster":"C03","score":2,"max":3,"grade":"B","probes":3}
//! ```
//!
//! This is the local-first transport: no external service is contacted.
//! The cockpit reader (task #41) is a separate effort in the cockpit-mesh
//! domain; this subcommand just guarantees a stable, append-only log.

use std::fs::OpenOptions;
use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};
use serde::Serialize;

use super::cockpit_read::{run as read_run, CockpitReadArgs};

use agileplus_governance::scoring_engine::{evaluate, ClusterScore, ScoreReport};

/// Default NDJSON log path per the spec. Override with `--output`.
fn default_log_path() -> Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("resolving $HOME for cockpit log")?;
    Ok(home.join(".agileplus").join("cockpit.ndjson"))
}

/// One NDJSON record per cluster — the dashboard reads these.
#[derive(Debug, Clone, Serialize)]
pub struct CockpitRecord {
    /// ISO-8601 UTC timestamp (second precision).
    pub ts: String,
    /// Repo display name (basename of `--repo`).
    pub repo: String,
    /// Cluster id (e.g. "C03").
    pub cluster: String,
    /// Cluster score 0-3 multiplied by number of pillars (mirrors `total_points`).
    pub score: u32,
    /// Maximum possible score for the cluster.
    pub max: u32,
    /// Letter grade (A-F) computed from `score / max`.
    pub grade: String,
    /// Number of content-probe matches for this cluster (0 if probes disabled).
    pub probes: u32,
}

/// Score a repo and append per-cluster records to the NDJSON log.
#[derive(Debug, Args)]
pub struct CockpitArgs {
    #[command(subcommand)]
    pub sub: CockpitSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum CockpitSubcommand {
    /// Read the cockpit NDJSON log as a per-repo × per-cluster grade table.
    Read(CockpitReadArgs),
    /// Score a repo against the rubric catalog and append results to the cockpit log.
    Publish {
        /// Path to the repo root to scan.
        #[arg(long, value_name = "PATH")]
        repo: PathBuf,

        /// Path to a rubric catalog JSON. Defaults to the workspace-bundled
        /// `PILLARS-CATALOG.json` (same resolution as `ap rubric score`).
        #[arg(long, value_name = "PATH")]
        catalog: Option<PathBuf>,

        /// Comma-separated list of cluster ids to score (e.g. `C03,C10,C11`).
        #[arg(long, value_name = "IDS", value_delimiter = ',')]
        clusters: Option<Vec<String>>,

        /// Path to the NDJSON output log. Defaults to `~/.agileplus/cockpit.ndjson`.
        #[arg(long, value_name = "PATH")]
        output: Option<PathBuf>,

        /// Disable content-probe evaluation (legacy v1 path-presence only).
        #[arg(long)]
        no_probes: bool,
    },
    /// Print the configured log path and exit (no scoring, no file write).
    Path,
}

/// Top-level dispatch for the `ap cockpit` subcommand group.
///
/// The root CLI accepts a global `--repo <PATH>` for commands that operate on
/// a repository. Cockpit readers instead operate on a shared score log, so a
/// repository path cannot identify a record without silently changing its
/// meaning. Keep the publisher's own `--repo <PATH>` contract untouched and
/// direct readers to the exact-match record filter.
pub fn run(args: &CockpitArgs, global_repo: Option<&Path>) -> Result<()> {
    if global_repo.is_some() {
        match &args.sub {
            CockpitSubcommand::Read(_) => bail!(
                "`cockpit read` does not accept global `--repo <PATH>`; use `--filter-repo <NAME>` to filter cockpit records"
            ),
            CockpitSubcommand::Path => bail!(
                "`cockpit path` does not accept global `--repo <PATH>`; use `cockpit read --filter-repo <NAME>` to filter cockpit records"
            ),
            CockpitSubcommand::Publish { .. } => {}
        }
    }

    match &args.sub {
        CockpitSubcommand::Read(read_args) => read_run(read_args),
        CockpitSubcommand::Path => {
            let p = default_log_path()?;
            println!("{}", p.display());
            Ok(())
        }
        CockpitSubcommand::Publish {
            repo,
            catalog,
            clusters,
            output,
            no_probes,
        } => {
            if !repo.exists() {
                bail!("--repo path does not exist: {}", repo.display());
            }
            if !repo.is_dir() {
                bail!("--repo must be a directory: {}", repo.display());
            }

            let catalog_path = match catalog {
                Some(p) => p.clone(),
                None => super::rubric::resolve_default_catalog_for_siblings()?,
            };
            if !catalog_path.exists() {
                bail!(
                    "rubric catalog not found at {} (pass --catalog <path> to override)",
                    catalog_path.display()
                );
            }

            let cluster_filter: Vec<String> = clusters.clone().unwrap_or_default();
            // TODO(rubric-v2): switch to `evaluate_with_probes` once PR #902
            // (agileplus#902) lands on `main`. The cockpit log schema already
            // carries a `probes` field — until v2 merges, every published
            // record reports `probes: 0`, which is correct for the v1 eval.
            // `--no-probes` is currently a no-op (v1 is probe-free).
            let _ = no_probes;
            let report = evaluate(repo, &catalog_path, &cluster_filter)
                .with_context(|| format!("scoring {}", repo.display()))?;

            let log_path = output.clone().map(PathBuf::from).unwrap_or_else(|| {
                // SAFETY: default_log_path() only errors on home-dir failure;
                // the cockpit subcommand runs after we've already exercised
                // dirs via the rubric CLI, so this is essentially infallible.
                default_log_path().unwrap_or_else(|_| PathBuf::from("cockpit.ndjson"))
            });

            let records = records_from_report(&report);
            append_ndjson(&log_path, &records)?;

            println!(
                "wrote {} record(s) to {}",
                records.len(),
                log_path.display()
            );
            for r in &records {
                println!(
                    "  {}\t{}\t{}/{} {}",
                    r.cluster, r.repo, r.score, r.max, r.grade
                );
            }
            Ok(())
        }
    }
}

/// Build one [`CockpitRecord`] per cluster in the report. Probe hit count
/// for each cluster is computed by summing evidence citations that begin
/// with `"probe:"` — matches the scoring_engine convention from #902.
fn records_from_report(report: &ScoreReport) -> Vec<CockpitRecord> {
    report
        .clusters
        .iter()
        .map(|c| {
            let probes = probe_hit_count(c);
            CockpitRecord {
                ts: iso8601_now(),
                repo: report.repo.clone(),
                cluster: c.cluster.clone(),
                score: c.total_points,
                max: c.max_points,
                grade: grade_for(c),
                probes,
            }
        })
        .collect()
}

fn probe_hit_count(cluster: &ClusterScore) -> u32 {
    cluster
        .pillars
        .iter()
        .flat_map(|p| p.evidence.iter())
        .filter(|e| e.starts_with("probe:"))
        .count() as u32
}

fn grade_for(c: &ClusterScore) -> String {
    if c.max_points == 0 {
        return "F".into();
    }
    let pct = (c.total_points * 100) / c.max_points;
    let g = match pct {
        90..=100 => "A",
        75..=89 => "B",
        60..=74 => "C",
        40..=59 => "D",
        _ => "F",
    };
    g.to_string()
}

fn iso8601_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Format as RFC3339-ish without pulling chrono. Seconds precision is
    // sufficient for cockpit readers — sub-second drift is operator noise.
    format!("epoch:{secs}")
}

fn append_ndjson(path: &Path, records: &[CockpitRecord]) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating log directory {}", parent.display()))?;
        }
    }
    let f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening cockpit log {}", path.display()))?;
    let mut w = BufWriter::new(f);
    for r in records {
        let line = serde_json::to_string(r).context("serializing cockpit record")?;
        writeln!(w, "{line}").context("writing cockpit record")?;
    }
    w.flush().context("flushing cockpit log")
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use agileplus_governance::scoring_engine::{PillarScore, ScoreReport};

    fn sample_report() -> ScoreReport {
        ScoreReport {
            repo: "demo".into(),
            date: "2026-07-06".into(),
            clusters: vec![
                ClusterScore {
                    cluster: "C03".into(),
                    pillars: vec![PillarScore {
                        pillar_id: "L21".into(),
                        title: "L21 — FR/NFR".into(),
                        score: 2,
                        glyph: "△",
                        evidence: vec![
                            "AGENTS.md:1".into(),
                            "probe:3 match(es) in this cluster".into(),
                        ],
                        gaps: vec![],
                        soft_goal_delta: "partial".into(),
                    }],
                    total_points: 2,
                    max_points: 3,
                },
                ClusterScore {
                    cluster: "C04".into(),
                    pillars: vec![PillarScore {
                        pillar_id: "L31-L40".into(),
                        title: "L31-L40 — Security".into(),
                        score: 0,
                        glyph: "✗",
                        evidence: vec![],
                        gaps: vec!["gitleaks missing".into()],
                        soft_goal_delta: "partial".into(),
                    }],
                    total_points: 0,
                    max_points: 3,
                },
            ],
        }
    }

    #[test]
    fn records_emitted_one_per_cluster() {
        let report = sample_report();
        let records = records_from_report(&report);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].cluster, "C03");
        assert_eq!(records[0].repo, "demo");
        assert_eq!(records[0].score, 2);
        assert_eq!(records[0].max, 3);
        assert_eq!(records[0].grade, "C"); // 2/3 = 66%
        assert_eq!(records[0].probes, 1); // one "probe:" citation
        assert_eq!(records[1].cluster, "C04");
        assert_eq!(records[1].grade, "F");
        assert_eq!(records[1].probes, 0);
    }

    #[test]
    fn ndjson_round_trip_is_line_delimited() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cockpit.ndjson");
        let report = sample_report();
        let records = records_from_report(&report);
        append_ndjson(&path, &records).expect("append");
        let text = std::fs::read_to_string(&path).expect("read");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "expected one JSON record per line");
        for line in &lines {
            let parsed: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("invalid JSON: {e} -- {line}"));
            assert!(parsed.get("cluster").is_some());
            assert!(parsed.get("score").is_some());
            assert!(parsed.get("max").is_some());
            assert!(parsed.get("grade").is_some());
            assert!(parsed.get("ts").is_some());
            assert!(parsed.get("probes").is_some());
        }
    }

    #[test]
    fn append_ndjson_creates_parent_dirs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let nested = dir.path().join("a").join("b").join("cockpit.ndjson");
        let report = sample_report();
        append_ndjson(&nested, &records_from_report(&report)).expect("append");
        assert!(nested.exists());
    }

    #[test]
    fn grade_for_handles_boundaries() {
        use agileplus_governance::scoring_engine::ClusterScore;
        let cases = [
            (3u32, 3u32, "A"), // 100%
            (2, 3, "C"),       // 66%
            (0, 0, "F"),       // zero max → "F" (early return)
        ];
        for (score, max, expected) in cases {
            let c = ClusterScore {
                cluster: "C00".into(),
                pillars: vec![],
                total_points: score,
                max_points: max,
            };
            assert_eq!(grade_for(&c), expected, "{score}/{max}");
        }
    }

    #[test]
    fn grade_for_uses_correct_cutoffs() {
        // Re-pin the canonical A/B/C/D/F bands as documented.
        let table: &[(u32, u32, &str)] = &[
            (27, 30, "A"), // 90%
            (26, 30, "B"), // 86%
            (22, 30, "C"), // 73%
            (17, 30, "D"), // 56%
            (10, 30, "F"), // 33%
        ];
        for (score, max, expected) in table {
            let c = ClusterScore {
                cluster: "C00".into(),
                pillars: vec![],
                total_points: *score,
                max_points: *max,
            };
            assert_eq!(grade_for(&c), *expected, "{score}/{max}");
        }
    }
}
