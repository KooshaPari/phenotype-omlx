"""Tests for the ``promote`` subcommand.

Each test patches ``_project_root`` so the cache layout resolves to a
fresh ``tmp_path`` directory — no real ``.omlx`` directory is ever
touched. ``_IO`` mirrors the helper used in ``test_commands.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import json
import os
import sys

import pytest

from omlx_research.cli.commands import promote as promote_mod
from omlx_research.cli.commands.promote import (
    PromotionError,
    build_candidate,
    cache_root,
    canonical_bytes,
    cmd_promote,
    content_hash,
    gate_passes,
    parse_gates,
    promotion_path,
    sign_record,
    validate,
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


# --- fixture: redirect cache to tmp_path -----------------------------------

@pytest.fixture
def cache(monkeypatch, tmp_path):
    """Point the promote module's cache root at ``tmp_path/.omlx/cache``."""
    monkeypatch.setattr(promote_mod, "_project_root", lambda: str(tmp_path))
    return tmp_path / ".omlx" / "cache"


# --- pure-function tests ---------------------------------------------------

def test_parse_gates_basic():
    gates = parse_gates("mmlu=0.85,gpqa=0.75")
    assert [g["id"] for g in gates] == ["mmlu", "gpqa"]
    assert gates[0]["threshold"] == 0.85
    assert gates[0]["direction"] == "at_least"
    assert gates[1]["threshold"] == 0.75


def test_parse_gates_strips_whitespace():
    gates = parse_gates("  mmlu = 0.5 ,  gpqa = 0.25 ")
    assert gates[0]["id"] == "mmlu"
    assert gates[0]["threshold"] == 0.5
    assert gates[1]["id"] == "gpqa"


def test_parse_gates_rejects_empty():
    with pytest.raises(ValueError):
        parse_gates("")
    with pytest.raises(ValueError):
        parse_gates("  , , ")


def test_parse_gates_rejects_bad_threshold():
    with pytest.raises(ValueError):
        parse_gates("mmlu=abc")
    with pytest.raises(ValueError):
        parse_gates("mmlu=inf")
    with pytest.raises(ValueError):
        parse_gates("=0.5")


def test_gate_passes_at_least_and_at_most():
    g_atleast = {"id": "x", "threshold": 0.5, "direction": "at_least"}
    g_atmost = {"id": "y", "threshold": 0.5, "direction": "at_most"}
    assert gate_passes(g_atleast, 0.5) is True
    assert gate_passes(g_atleast, 0.49) is False
    assert gate_passes(g_atmost, 0.5) is True
    assert gate_passes(g_atmost, 0.51) is False


def test_gate_passes_unknown_direction_raises():
    with pytest.raises(ValueError):
        gate_passes({"id": "x", "threshold": 0.0, "direction": "weird"}, 0.0)


def test_canonical_bytes_excludes_signature_and_hash():
    rec = {
        "candidate_id": "knl",
        "source_revision": "rev",
        "approved_at_unix_ms": 1,
        "approver": "",
        "gates": [],
        "evidence": [],
        "justification": "",
        "tuning_record_id": None,
        "signature": "should-be-excluded",
        "content_hash": "should-be-excluded",
    }
    blob = canonical_bytes(rec)
    text = blob.decode("utf-8")
    assert "should-be-excluded" not in text
    assert "signature" not in text
    assert "content_hash" not in text


def test_canonical_bytes_is_stable_under_key_reorder():
    a = canonical_bytes({"candidate_id": "k", "source_revision": "r", "approved_at_unix_ms": 1,
                         "approver": "", "gates": [], "evidence": [],
                         "justification": "", "tuning_record_id": None})
    b = canonical_bytes({"tuning_record_id": None, "justification": "", "evidence": [],
                         "gates": [], "approver": "", "approved_at_unix_ms": 1,
                         "source_revision": "r", "candidate_id": "k"})
    assert a == b


def test_content_hash_changes_when_field_changes():
    a = content_hash({"candidate_id": "k1", "source_revision": "r",
                      "approved_at_unix_ms": 1, "approver": "",
                      "gates": [], "evidence": [], "justification": "",
                      "tuning_record_id": None})
    b = content_hash({"candidate_id": "k2", "source_revision": "r",
                      "approved_at_unix_ms": 1, "approver": "",
                      "gates": [], "evidence": [], "justification": "",
                      "tuning_record_id": None})
    assert a != b


def test_sign_record_uses_hmac_sha256():
    rec = {"candidate_id": "k", "source_revision": "r", "approved_at_unix_ms": 1,
           "approver": "", "gates": [], "evidence": [], "justification": "",
           "tuning_record_id": None}
    key = b"deadbeef"
    sig = sign_record(rec, key)
    expected = hmac.new(key, canonical_bytes(rec), hashlib.sha256).hexdigest()
    assert sig == expected
    # Different key produces a different signature.
    assert sign_record(rec, b"other") != sig


# --- validate() tests ------------------------------------------------------

def test_validate_passes_for_synthetic_evidence():
    """Synthetic scores fall in [0.5, 1.0); threshold 0.0 always passes."""
    rec = build_candidate("k1", parse_gates("mmlu=0.0"), None)
    validate(rec)  # no exception


def test_validate_rejects_missing_evidence():
    rec = build_candidate("k1", parse_gates("mmlu=0.0"), None)
    rec["evidence"] = []  # strip the synthetic evidence
    with pytest.raises(PromotionError) as exc_info:
        validate(rec)
    assert exc_info.value.kind == "gate_missing_evidence"
    assert exc_info.value.fields["gate"] == "mmlu"


def test_validate_rejects_gate_failure():
    """Force a failure by lowering the gate's threshold above the score."""
    rec = build_candidate("k1", parse_gates("mmlu=10.0"), None)
    with pytest.raises(PromotionError) as exc_info:
        validate(rec)
    assert exc_info.value.kind == "gate_rejected"
    assert exc_info.value.fields["gate"] == "mmlu"
    assert exc_info.value.fields["threshold"] == 10.0
    assert exc_info.value.fields["observed"] < 10.0


def test_validate_rejects_empty_gates():
    rec = {"gates": [], "evidence": [{"id": "mmlu", "score": 0.9}]}
    with pytest.raises(PromotionError) as exc_info:
        validate(rec)
    assert exc_info.value.kind == "promotion_without_gates"


def test_validate_rejects_duplicate_evidence():
    rec = {
        "gates": [{"id": "mmlu", "threshold": 0.0, "direction": "at_least"}],
        "evidence": [
            {"id": "mmlu", "score": 0.9},
            {"id": "mmlu", "score": 0.8},
        ],
    }
    with pytest.raises(PromotionError) as exc_info:
        validate(rec)
    assert exc_info.value.kind == "duplicate_evidence"


# --- cmd_promote integration tests ----------------------------------------

def test_cmd_promote_writes_signed_record(cache, monkeypatch):
    monkeypatch.setenv("USER", "tester")
    with _IO() as io:
        rc = cmd_promote(_ns(
            kernel_id="knl-promote-1",
            gates="mmlu=0.1,gpqa=0.1",
            sign_key="deadbeef",
            approver="alice",
            decision="auto",
            json=False,
        ))
    assert rc == 0
    out = io.stdout.getvalue()
    assert "promotion         : OK" in out
    assert "approver          : alice" in out
    path = promotion_path("knl-promote-1")
    assert os.path.exists(path)
    rec = json.loads(open(path).read())
    assert rec["candidate_id"] == "knl-promote-1"
    assert rec["approver"] == "alice"
    assert rec["signature"] is not None
    # Content hash on disk should match the freshly-computed one.
    assert rec["content_hash"] == content_hash(rec)


def test_cmd_promote_unsigned_record_has_no_signature(cache):
    with _IO() as io:
        rc = cmd_promote(_ns(
            kernel_id="knl-promote-2",
            gates="mmlu=0.0",
            sign_key=None,
            approver="bob",
            decision="auto",
            json=False,
        ))
    assert rc == 0
    rec = json.loads(open(promotion_path("knl-promote-2")).read())
    assert rec["signature"] is None
    assert rec["approver"] == "bob"


def test_cmd_promote_rejects_failing_gate(cache):
    with _IO() as io:
        rc = cmd_promote(_ns(
            kernel_id="knl-promote-fail",
            gates="mmlu=10.0",
            sign_key=None,
            approver="tester",
            decision="auto",
            json=False,
        ))
    assert rc == 2
    assert "promotion rejected" in io.stderr.getvalue()
    # Cache file should not be written on failure.
    assert not os.path.exists(promotion_path("knl-promote-fail"))


def test_cmd_promote_json_envelope_on_failure(cache):
    with _IO() as io:
        rc = cmd_promote(_ns(
            kernel_id="knl-promote-fail-json",
            gates="mmlu=10.0",
            sign_key=None,
            approver="tester",
            decision="auto",
            json=True,
        ))
    assert rc == 2
    payload = json.loads(io.stdout.getvalue())
    assert payload["kind"] == "gate_rejected"
    assert payload["gate"] == "mmlu"
    assert payload["threshold"] == 10.0


def test_cmd_promote_malformed_gates_returns_exit_2(cache):
    with _IO() as io:
        rc = cmd_promote(_ns(
            kernel_id="knl-promote-bad",
            gates="mmlu",
            sign_key=None,
            approver="tester",
            decision="auto",
            json=False,
        ))
    assert rc == 2
    assert "must be 'id=threshold'" in io.stderr.getvalue()


def test_cmd_promote_bad_sign_key_returns_exit_2(cache):
    with _IO() as io:
        rc = cmd_promote(_ns(
            kernel_id="knl-promote-badkey",
            gates="mmlu=0.0",
            sign_key="not-hex!",
            approver="tester",
            decision="auto",
            json=False,
        ))
    assert rc == 2
    assert "hex bytes" in io.stderr.getvalue()


def test_cmd_promote_reuses_cached_evidence(cache):
    """A second promote call should reuse the evidence from the cache."""
    with _IO():
        cmd_promote(_ns(
            kernel_id="knl-promote-reuse",
            gates="mmlu=0.0",
            sign_key=None,
            approver="t",
            decision="auto",
            json=False,
        ))
    rec_first = json.loads(open(promotion_path("knl-promote-reuse")).read())
    with _IO():
        cmd_promote(_ns(
            kernel_id="knl-promote-reuse",
            gates="mmlu=0.0",
            sign_key=None,
            approver="t",
            decision="auto",
            json=False,
        ))
    rec_second = json.loads(open(promotion_path("knl-promote-reuse")).read())
    # The two calls used the same scores; the only differences should be
    # approved_at_unix_ms (and possibly content_hash).
    assert rec_first["evidence"] == rec_second["evidence"]


def test_cmd_promote_json_output_is_structured(cache):
    with _IO() as io:
        rc = cmd_promote(_ns(
            kernel_id="knl-promote-json",
            gates="mmlu=0.0",
            sign_key="00ff",
            approver="alice",
            decision="manual",
            json=True,
        ))
    assert rc == 0
    payload = json.loads(io.stdout.getvalue())
    assert payload["ok"] is True
    assert payload["decision"] == "manual"
    assert payload["candidate_id"] == "knl-promote-json"
    assert payload["gate_count"] == 1
    assert payload["signature"] is not None
    assert len(payload["content_hash"]) == 64
