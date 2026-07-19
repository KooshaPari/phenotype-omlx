"""Tests for the ``gates`` subcommand and its list/add/remove/check actions."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

import pytest

from omlx_research.cli.commands import gates as gates_mod
from omlx_research.cli.commands.gates import (
    cache_root,
    cmd_gates,
    cmd_gates_add,
    cmd_gates_check,
    cmd_gates_list,
    cmd_gates_remove,
    find_gate,
    gates_path,
    load_config,
    save_config,
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
    monkeypatch.setattr(gates_mod, "_project_root", lambda: str(tmp_path))
    return tmp_path / ".omlx" / "cache"


# --- storage tests ---------------------------------------------------------

def test_load_config_returns_empty_when_missing(cache):
    cfg = load_config("missing-kernel")
    assert cfg["kernel_id"] == "missing-kernel"
    assert cfg["gates"] == []
    assert cfg["schema_version"] == 1


def test_load_config_normalizes_existing_file(cache):
    path = gates_path("k1")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": 1,
            "kernel_id": "k1",
            "updated_at_unix_ms": 0,
            "gates": [
                {"id": "mmlu", "threshold": 0.7, "direction": "at_least", "note": "ok"},
                {"id": "bogus", "threshold": "not-a-number"},  # dropped
                {"not-a-gate": True},  # dropped
            ],
        }, f)
    cfg = load_config("k1")
    assert len(cfg["gates"]) == 1
    assert cfg["gates"][0]["id"] == "mmlu"


def test_save_and_load_roundtrip(cache):
    cfg = load_config("k2")
    cfg["gates"].append({"id": "mmlu", "threshold": 0.8, "direction": "at_least", "note": ""})
    save_config("k2", cfg)
    cfg2 = load_config("k2")
    assert cfg2["gates"][0]["id"] == "mmlu"
    assert cfg2["gates"][0]["threshold"] == 0.8


def test_find_gate_returns_match(cache):
    cfg = {"gates": [{"id": "mmlu", "threshold": 0.5, "direction": "at_least", "note": ""}]}
    g = find_gate(cfg, "mmlu")
    assert g is not None and g["threshold"] == 0.5
    assert find_gate(cfg, "nope") is None


# --- list tests ------------------------------------------------------------

def test_cmd_gates_list_empty(cache):
    with _IO() as io:
        rc = cmd_gates_list(_ns(kernel_id="k-empty", json=False))
    assert rc == 0
    assert "no gates configured" in io.stdout.getvalue()


def test_cmd_gates_list_human(cache):
    cfg = load_config("k-list")
    cfg["gates"].extend([
        {"id": "mmlu", "threshold": 0.7, "direction": "at_least", "note": "main"},
        {"id": "gpqa", "threshold": 0.5, "direction": "at_least", "note": ""},
    ])
    save_config("k-list", cfg)
    with _IO() as io:
        rc = cmd_gates_list(_ns(kernel_id="k-list", json=False))
    assert rc == 0
    out = io.stdout.getvalue()
    assert "kernel_id         : k-list" in out
    assert "mmlu" in out and "gpqa" in out
    assert "0.7000" in out and "0.5000" in out


def test_cmd_gates_list_json(cache):
    cfg = load_config("k-list-json")
    cfg["gates"].append({"id": "mmlu", "threshold": 0.7, "direction": "at_least", "note": ""})
    save_config("k-list-json", cfg)
    with _IO() as io:
        rc = cmd_gates_list(_ns(kernel_id="k-list-json", json=True))
    assert rc == 0
    payload = json.loads(io.stdout.getvalue())
    assert payload["kernel_id"] == "k-list-json"
    assert payload["gates"][0]["id"] == "mmlu"
    assert payload["cache_path"].endswith("k-list-json.json")


# --- add tests -------------------------------------------------------------

def test_cmd_gates_add_creates_file(cache):
    with _IO() as io:
        rc = cmd_gates_add(_ns(
            kernel_id="k-add", gate="mmlu", threshold=0.85,
            at_least=True, at_most=False, note="MMLU-Pro main",
            json=False,
        ))
    assert rc == 0
    assert "added gate 'mmlu'" in io.stdout.getvalue()
    assert "0.85" in io.stdout.getvalue()
    cfg = load_config("k-add")
    assert any(g["id"] == "mmlu" for g in cfg["gates"])


def test_cmd_gates_add_updates_existing(cache):
    cfg = load_config("k-upd")
    cfg["gates"].append({"id": "mmlu", "threshold": 0.5, "direction": "at_least", "note": ""})
    save_config("k-upd", cfg)
    with _IO() as io:
        rc = cmd_gates_add(_ns(
            kernel_id="k-upd", gate="mmlu", threshold=0.9,
            at_least=True, at_most=False, note="bumped",
            json=False,
        ))
    assert rc == 0
    assert "updated" in io.stdout.getvalue()
    cfg = load_config("k-upd")
    assert len(cfg["gates"]) == 1
    assert cfg["gates"][0]["threshold"] == 0.9
    assert cfg["gates"][0]["note"] == "bumped"


def test_cmd_gates_add_at_most_direction(cache):
    with _IO():
        rc = cmd_gates_add(_ns(
            kernel_id="k-atmost", gate="ppl", threshold=10.0,
            at_least=False, at_most=True, note="",
            json=False,
        ))
    assert rc == 0
    cfg = load_config("k-atmost")
    assert cfg["gates"][0]["direction"] == "at_most"


def test_cmd_gates_add_missing_gate_arg(cache):
    with _IO() as io:
        rc = cmd_gates_add(_ns(
            kernel_id="k-x", gate=None, threshold=0.5,
            at_least=True, at_most=False, note="", json=False,
        ))
    assert rc == 2
    assert "--gate ID is required" in io.stderr.getvalue()


def test_cmd_gates_add_missing_threshold(cache):
    with _IO() as io:
        rc = cmd_gates_add(_ns(
            kernel_id="k-x", gate="mmlu", threshold=None,
            at_least=True, at_most=False, note="", json=False,
        ))
    assert rc == 2
    assert "threshold" in io.stderr.getvalue()


def test_cmd_gates_add_json(cache):
    with _IO() as io:
        rc = cmd_gates_add(_ns(
            kernel_id="k-add-json", gate="mmlu", threshold=0.7,
            at_least=True, at_most=False, note="", json=True,
        ))
    assert rc == 0
    payload = json.loads(io.stdout.getvalue())
    assert payload["verb"] == "added"
    assert payload["gate"]["id"] == "mmlu"
    assert payload["gate"]["threshold"] == 0.7


# --- remove tests ----------------------------------------------------------

def test_cmd_gates_remove_happy(cache):
    cfg = load_config("k-rm")
    cfg["gates"].extend([
        {"id": "mmlu", "threshold": 0.5, "direction": "at_least", "note": ""},
        {"id": "gpqa", "threshold": 0.5, "direction": "at_least", "note": ""},
    ])
    save_config("k-rm", cfg)
    with _IO():
        rc = cmd_gates_remove(_ns(kernel_id="k-rm", gate="mmlu", json=False))
    assert rc == 0
    cfg = load_config("k-rm")
    assert [g["id"] for g in cfg["gates"]] == ["gpqa"]


def test_cmd_gates_remove_unknown_returns_exit_2(cache):
    with _IO() as io:
        rc = cmd_gates_remove(_ns(kernel_id="k-rm2", gate="nope", json=False))
    assert rc == 2
    assert "no gate 'nope'" in io.stderr.getvalue()


def test_cmd_gates_remove_missing_gate_arg(cache):
    with _IO() as io:
        rc = cmd_gates_remove(_ns(kernel_id="k-rm3", gate=None, json=False))
    assert rc == 2
    assert "--gate ID is required" in io.stderr.getvalue()


def test_cmd_gates_remove_json(cache):
    cfg = load_config("k-rm-json")
    cfg["gates"].append({"id": "mmlu", "threshold": 0.5, "direction": "at_least", "note": ""})
    save_config("k-rm-json", cfg)
    with _IO() as io:
        rc = cmd_gates_remove(_ns(kernel_id="k-rm-json", gate="mmlu", json=True))
    assert rc == 0
    payload = json.loads(io.stdout.getvalue())
    assert payload["removed"] == "mmlu"
    assert payload["remaining"] == 0


# --- check tests -----------------------------------------------------------

def test_cmd_gates_check_passes(cache):
    cfg = load_config("k-check")
    cfg["gates"].append({"id": "mmlu", "threshold": 0.7, "direction": "at_least", "note": ""})
    save_config("k-check", cfg)
    with _IO() as io:
        rc = cmd_gates_check(_ns(
            kernel_id="k-check", gate="mmlu", score=0.85, json=False,
        ))
    assert rc == 0
    out = io.stdout.getvalue()
    assert "result : OK" in out


def test_cmd_gates_check_fails(cache):
    cfg = load_config("k-check-fail")
    cfg["gates"].append({"id": "mmlu", "threshold": 0.7, "direction": "at_least", "note": ""})
    save_config("k-check-fail", cfg)
    with _IO() as io:
        rc = cmd_gates_check(_ns(
            kernel_id="k-check-fail", gate="mmlu", score=0.5, json=False,
        ))
    assert rc == 0  # the *command* succeeds; "FAIL" is the gate verdict, not exit code
    out = io.stdout.getvalue()
    assert "result : FAIL" in out


def test_cmd_gates_check_at_most(cache):
    cfg = load_config("k-check-am")
    cfg["gates"].append({"id": "ppl", "threshold": 10.0, "direction": "at_most", "note": ""})
    save_config("k-check-am", cfg)
    with _IO() as io:
        rc = cmd_gates_check(_ns(
            kernel_id="k-check-am", gate="ppl", score=5.0, json=False,
        ))
    assert rc == 0
    assert "result : OK" in io.stdout.getvalue()
    with _IO() as io:
        rc = cmd_gates_check(_ns(
            kernel_id="k-check-am", gate="ppl", score=20.0, json=False,
        ))
    assert "result : FAIL" in io.stdout.getvalue()


def test_cmd_gates_check_unknown_gate_returns_exit_2(cache):
    with _IO() as io:
        rc = cmd_gates_check(_ns(
            kernel_id="k-check-unk", gate="mmlu", score=0.5, json=False,
        ))
    assert rc == 2


def test_cmd_gates_check_missing_score(cache):
    with _IO() as io:
        rc = cmd_gates_check(_ns(
            kernel_id="k-check-2", gate="mmlu", score=None, json=False,
        ))
    assert rc == 2
    assert "--score" in io.stderr.getvalue()


def test_cmd_gates_check_json(cache):
    cfg = load_config("k-check-json")
    cfg["gates"].append({"id": "mmlu", "threshold": 0.7, "direction": "at_least", "note": ""})
    save_config("k-check-json", cfg)
    with _IO() as io:
        rc = cmd_gates_check(_ns(
            kernel_id="k-check-json", gate="mmlu", score=0.85, json=True,
        ))
    assert rc == 0
    payload = json.loads(io.stdout.getvalue())
    assert payload["passes"] is True
    assert payload["score"] == 0.85
    assert payload["threshold"] == 0.7
    assert payload["direction"] == "at_least"


# --- cmd_gates dispatch ----------------------------------------------------

def test_cmd_gates_dispatches_to_list(cache):
    with _IO() as io:
        rc = cmd_gates(_ns(
            gates_action="list", kernel_id="dispatch-list", json=False,
        ))
    assert rc == 0
    assert "no gates configured" in io.stdout.getvalue()


def test_cmd_gates_dispatches_to_add(cache):
    with _IO():
        rc = cmd_gates(_ns(
            gates_action="add", kernel_id="dispatch-add",
            gate="mmlu", threshold=0.7,
            at_least=True, at_most=False, note="", json=False,
        ))
    assert rc == 0
    assert any(g["id"] == "mmlu" for g in load_config("dispatch-add")["gates"])


def test_cmd_gates_unknown_subcommand(cache):
    with _IO() as io:
        rc = cmd_gates(_ns(
            gates_action="bogus", kernel_id="k-x", json=False,
        ))
    assert rc == 2
    assert "unknown subcommand" in io.stderr.getvalue()
