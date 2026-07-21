#!/usr/bin/env python3
"""L2 — Contamination / n-gram leakage scan (offline, no MLX).

Compares harness fixture text and EvaluationReport cell replies for
suspicious overlap (exact needle reuse, high Jaccard on character
n-grams). Fail-loud verdict when synthetic all-pass runs also share
verbatim fixture blobs.

Outputs JSON+Markdown under ``research/contamination/`` by default.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


def _ngrams(tokens: list[str], n: int = 5) -> Counter[str]:
    if len(tokens) < n:
        return Counter()
    return Counter(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def jaccard(a: Counter[str], b: Counter[str]) -> float:
    if not a and not b:
        return 0.0
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 0.0


def load_cells(report_path: Path) -> list[dict]:
    data = json.loads(report_path.read_text())

    def find(obj, depth=0):
        if depth > 8:
            return None
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            if "pass_at_1" in obj[0] or "reply" in obj[0] or "task_id" in obj[0]:
                return obj
        if isinstance(obj, dict):
            for v in obj.values():
                r = find(v, depth + 1)
                if r is not None:
                    return r
        return None

    cells = find(data)
    if not cells:
        raise SystemExit(f"error: no cells found in {report_path}")
    return cells


def load_fixture_blobs(fixture_path: Path) -> list[tuple[str, str]]:
    raw = json.loads(fixture_path.read_text())
    out: list[tuple[str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                path = str(item.get("path") or item.get("id") or "fixture")
                content = str(item.get("content") or item.get("text") or "")
                if content.strip():
                    out.append((path, content))
    elif isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, str) and v.strip():
                out.append((str(k), v))
    return out


def scan(
    cells: list[dict],
    fixtures: list[tuple[str, str]],
    *,
    n: int = 5,
    jaccard_warn: float = 0.35,
) -> dict:
    fixture_grams = [(path, _ngrams(_tokens(text), n)) for path, text in fixtures]
    fixture_exact = {text.strip() for _, text in fixtures if len(text.strip()) >= 40}

    hits: list[dict] = []
    reply_pass = 0
    reply_total = 0
    for c in cells:
        reply = str(c.get("reply") or c.get("completion") or "")
        if not reply.strip():
            continue
        reply_total += 1
        if float(c.get("pass_at_1") or 0) >= 0.5:
            reply_pass += 1
        grams = _ngrams(_tokens(reply), n)
        best_path = None
        best_j = 0.0
        for path, fg in fixture_grams:
            j = jaccard(grams, fg)
            if j > best_j:
                best_j = j
                best_path = path
        exact = reply.strip() in fixture_exact or any(
            blob in reply for blob in fixture_exact if len(blob) >= 80
        )
        if exact or best_j >= jaccard_warn:
            hits.append(
                {
                    "suite": c.get("suite"),
                    "task_id": c.get("task_id"),
                    "variant": c.get("variant"),
                    "jaccard_ngram": round(best_j, 4),
                    "fixture_path": best_path,
                    "exact_fixture_blob": bool(exact),
                    "pass_at_1": c.get("pass_at_1"),
                    "judge_score": c.get("judge_score"),
                }
            )

    # Degenerate synthetic signal (from L1): all pass + empty/zero judge
    pas = [float(c.get("pass_at_1") or 0) for c in cells]
    jud = [float(c.get("judge_score") or 0) for c in cells]
    degenerate = (
        len(pas) > 0
        and all(p >= 1.0 for p in pas)
        and all(j == 0.0 for j in jud)
    )

    if degenerate:
        verdict = "UNTRUSTED_SYNTHETIC"
    elif hits:
        verdict = "LEAKAGE_SUSPECT"
    else:
        verdict = "CLEAN_OR_INCONCLUSIVE"

    return {
        "experiment": "L2-contamination",
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_cells": len(cells),
        "n_replies": reply_total,
        "n_pass_replies": reply_pass,
        "n_fixtures": len(fixtures),
        "ngram": n,
        "jaccard_warn": jaccard_warn,
        "n_hits": len(hits),
        "hits_sample": hits[:50],
        "degenerate_all_pass_zero_judge": degenerate,
        "verdict": verdict,
        "notes": [
            "Offline scan only — does not prove training contamination.",
            "UNTRUSTED_SYNTHETIC means stock-vs-ours quality claims are invalid until live non-synthetic envelopes exist.",
            "Prefer Harbor NIAH (`run_via_harbor.sh --niah`) for live retrieval evidence.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--report",
        type=Path,
        default=Path(
            "/Users/kooshapari/CodeProjects/Phenotype/pheno-harness/bench/results/"
            "stock-vs-ours/run-v5-qwen35-08b.json"
        ),
        help="EvaluationReport / stock-vs-ours JSON",
    )
    ap.add_argument(
        "--fixtures",
        type=Path,
        default=Path(
            "/Users/kooshapari/CodeProjects/Phenotype/pheno-harness/bench/fixtures/"
            "sample_candidates.json"
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root / "research" / "contamination",
    )
    args = ap.parse_args(argv)

    if not args.report.is_file():
        raise SystemExit(f"error: report missing: {args.report}")
    fixtures = load_fixture_blobs(args.fixtures) if args.fixtures.is_file() else []
    cells = load_cells(args.report)
    result = scan(cells, fixtures)
    result["source_report"] = str(args.report)
    result["source_fixtures"] = str(args.fixtures)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Also mirror under pheno-harness when writable
    mirrors = [args.out_dir]
    ph = Path(
        "/Users/kooshapari/CodeProjects/Phenotype/pheno-harness/bench/results/"
        "contamination/qwen35-08b"
    )
    mirrors.append(ph)

    md = (
        f"# L2 Contamination — Qwen3.5-0.8B\n\n"
        f"**Verdict:** `{result['verdict']}`\n\n"
        f"- cells={result['n_cells']} replies={result['n_replies']} "
        f"hits={result['n_hits']}\n"
        f"- degenerate_synthetic={result['degenerate_all_pass_zero_judge']}\n"
        f"- report=`{args.report}`\n\n"
        f"## Notes\n"
        + "\n".join(f"- {n}" for n in result["notes"])
        + "\n"
    )

    for d in mirrors:
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / "qwen35-08b-report.json").write_text(json.dumps(result, indent=2) + "\n")
            (d / "qwen35-08b-report.md").write_text(md)
            print(f"wrote {d}/qwen35-08b-report.{{json,md}}")
        except OSError as e:
            print(f"skip mirror {d}: {e}")

    print("verdict=", result["verdict"])
    return 0 if result["verdict"] != "LEAKAGE_SUSPECT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
