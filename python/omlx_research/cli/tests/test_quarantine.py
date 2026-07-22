"""Tests for the ``quarantine`` subcommand.

Each test redirects the module's ``_project_root`` so the audit file
resolves to a fresh ``tmp_path`` directory.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

import pytest

from omlx_research.cli.commands import quarantine as quarantine_mod
from omlx_research.cli.commands.quarantine import (
    audit_path,
    cache_root,
    cmd_quarantine,
)


# --- stdio capture ---------------------------------------------------------

class _IO:
    def __init__(self):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def __enter__(self):
        self._real_out, self._real_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self.stdout, self.stderr
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self._real_out, self._real_err


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


# --- fixtures --------------------------------------------------------------

@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setattr(quarantine_mod, "_project_root", lambda: str(tmp_path))
    return tmp_path / ".omlx" / "cache"


# --- helpers ---------------------------------------------------------------

def _read_audit(path: str) -> list[dict]:
    """Read a JSONL audit file; return one parsed dict per non-empty line."""
    if not os.path.exists(path):
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --- tests -----------------------------------------------------------------

def test_cmd_quarantine_appends_hold_entry(cache):
    with _IO() as io:
        rc = cmd_quarantine(_ns(
            kernel_id="q1", reason="latency regression in CI",
            action="hold", approver="alice", json=False,
        ))
    assert rc == 0
    assert "quarantine        : OK" in io.stdout.getvalue()

    entries = _read_audit(audit_path())
    assert len(entries) == 1
    entry = entries[0]
    assert entry["action"] == "hold"
    assert entry["candidate_id"] == "q1"
    assert entry["reason"] == "latency regression in CI"
    assert entry["approver"] == "alice"
    assert "record" in entry
    assert entry["record"]["candidate_id"] == "q1"
    assert entry["record"]["content_hash"]  # 64-char hex
    assert entry["content_hash"] == entry["record"]["content_hash"]


def test_cmd_quarantine_appends_rollback_entry(cache):
    with _IO():
        rc = cmd_quarantine(_ns(
            kernel_id="q2", reason="MMLU-Pro regression 5%",
            action="rollback", approver="bob", json=False,
        ))
    assert rc == 0
    entries = _read_audit(audit_path())
    assert entries[0]["action"] == "rollback"
    assert entries[0]["reason"] == "MMLU-Pro regression 5%"


def test_cmd_quarantine_rejects_empty_reason(cache):
    with _IO() as io:
        rc = cmd_quarantine(_ns(
            kernel_id="q3", reason="",
            action="hold", approver="x", json=False,
        ))
    assert rc == 2
    assert "non-empty" in io.stderr.getvalue()
    assert not os.path.exists(audit_path())


def test_cmd_quarantine_rejects_whitespace_reason(cache):
    with _IO() as io:
        rc = cmd_quarantine(_ns(
            kernel_id="q3b", reason="   ",
            action="hold", approver="x", json=False,
        ))
    assert rc == 2


def test_cmd_quarantine_rejects_unknown_action(cache):
    with _IO() as io:
        rc = cmd_quarantine(_ns(
            kernel_id="q4", reason="x",
            action="promote", approver="x", json=False,
        ))
    assert rc == 2
    assert "unknown action" in io.stderr.getvalue()


def test_cmd_quarantine_json_envelope(cache):
    with _IO() as io:
        rc = cmd_quarantine(_ns(
            kernel_id="q5", reason="drift",
            action="hold", approver="carol", json=True,
        ))
    assert rc == 0
    payload = json.loads(io.stdout.getvalue())
    assert payload["ok"] is True
    assert payload["entry"]["action"] == "hold"
    assert payload["entry"]["reason"] == "drift"
    assert payload["entry"]["candidate_id"] == "q5"
    assert payload["audit_path"].endswith("audit.jsonl")


def test_cmd_quarantine_appends_in_order(cache):
    """Multiple calls produce multiple audit lines in submission order."""
    for kid in ("a", "b", "c"):
        with _IO():
            cmd_quarantine(_ns(
                kernel_id=kid, reason=f"reason-{kid}",
                action="hold", approver="x", json=False,
            ))
    entries = _read_audit(audit_path())
    assert [e["candidate_id"] for e in entries] == ["a", "b", "c"]


def test_cmd_quarantine_appends_only_new_line(cache):
    """A second call does not overwrite the first line (JSONL semantics)."""
    for _ in range(2):
        with _IO():
            cmd_quarantine(_ns(
                kernel_id="q6", reason="repeat",
                action="hold", approver="x", json=False,
            ))
    entries = _read_audit(audit_path())
    assert len(entries) == 2


def test_cmd_quarantine_reuses_cached_evidence(cache, monkeypatch, tmp_path):
    """When a PromotionRecord exists, the audit entry reuses its evidence."""
    # Patch promote._project_root too so write_promotion_record() lands in tmp_path.
    from omlx_research.cli.commands import promote as promote_mod
    monkeypatch.setattr(promote_mod, "_project_root", lambda: str(tmp_path))
    from omlx_research.cli.commands.promote import (
        content_hash as promo_content_hash,
        write_promotion_record,
    )
    cached_record = {
        "candidate_id": "q7",
        "source_revision": "rev-9",
        "approved_at_unix_ms": 1000,
        "approver": "alice",
        "gates": [],
        "evidence": [{"id": "mmlu", "score": 0.5,
                      "dataset_revision": "rev", "source_revision": "rev-9",
                      "captured_at_unix_ms": 1000, "note": ""}],
        "justification": "evidence from previous run",
        "tuning_record_id": None,
        "signature": None,
        "content_hash": "",
    }
    cached_record["content_hash"] = promo_content_hash(cached_record)
    write_promotion_record("q7", cached_record)

    with _IO():
        cmd_quarantine(_ns(
            kernel_id="q7", reason="audit me",
            action="hold", approver="alice", json=False,
        ))
    entries = _read_audit(audit_path())
    assert entries[0]["record"]["evidence"] == cached_record["evidence"]
    assert entries[0]["record"]["source_revision"] == "rev-9"


def test_cmd_quarantine_approver_defaults_to_user_env(cache, monkeypatch):
    monkeypatch.setenv("USER", "defaulttester")
    with _IO():
        cmd_quarantine(_ns(
            kernel_id="q8", reason="x",
            action="hold", approver=None, json=False,
        ))
    entries = _read_audit(audit_path())
    assert entries[0]["approver"] == "defaulttester"
