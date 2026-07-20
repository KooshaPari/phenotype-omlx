"""Tests for the omlx-research CLI commands.

Invokes cmd_*() functions directly so the suite stays stable when the
argument wiring shifts; ``main(argv=...)`` is exercised by two tests to
confirm dispatch.
"""

from __future__ import annotations

import argparse
import io
import json
import sys

import pytest

from omlx_research.cli import main
from omlx_research.cli.commands import (
    _shared, cmd_compare, cmd_evidence, cmd_explain,
    cmd_inspect, cmd_replay, cmd_tune,
)


# --- stdio capture ---------------------------------------------------------

class _IO:
    """Swap sys.stdout/sys.stderr for one cmd_* call."""

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

GOOD_PLAN = {
    "plan_id": "plan-test-1",
    "name": "tiny-llm",
    "family": "research-toy",
    "scheduler_policy": "fifo",
    "operators": [
        {"op_id": "op0", "kind": "DenseMatmul",
         "inputs": ["A", "B"], "outputs": ["Y0"]},
        {"op_id": "op1", "kind": "RMSNorm",
         "inputs": ["Y0", "weight"], "outputs": ["Y1"]},
    ],
    "states": [
        {"state_id": "st0", "kind": "tensor", "persistence": "scratch",
         "dtype": "f16", "owning_op": "op0"},
    ],
    "edges": [{"from_id": "op0", "to_id": "op1"}],
}

BAD_PLAN = {
    "plan_id": "bad", "name": "bad-plan", "family": "x",
    "scheduler_policy": "round-robin",  # not in SCHEDULER_POLICIES
    "operators": [{"op_id": "op0", "kind": "BizarreOp"}],  # unknown kind
}

GOOD_TRACE = {
    "plan_id": "plan-test-1", "op_id": "op0",
    "selected": {"kernel": "op0_metal", "reason": "best_latency",
                 "fallback": "op0_mlx"},
    "rejected": [
        {"kernel": "op0_mlx", "reason": "memory_pressure"},
        {"kernel": "op0_fallback", "reason": "slow"},
    ],
    "latency_ns": 1234,
}


def _trace(plan_id: str, op_id: str, sel_kernel: str, lat: int) -> dict:
    return {
        "plan_id": plan_id, "op_id": op_id,
        "selected": {"kernel": sel_kernel, "reason": "r", "fallback": "fb"},
        "rejected": [{"kernel": "k1", "reason": "r1"}],
        "latency_ns": lat,
    }


# --- inspect ---------------------------------------------------------------

def test_inspect_happy_path(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(GOOD_PLAN))
    with _IO() as io:
        rc = cmd_inspect(_ns(plan=str(p), empty=False,
                             show_states=False, show_deps=False))
    assert rc == 0
    out = io.stdout.getvalue()
    assert "plan-test-1" in out and "tiny-llm" in out
    assert "validation        : OK" in out
    assert "operator_count    : 2" in out
    assert "state_count       : 1" in out
    assert "edge_count        : 1" in out


def test_inspect_missing_file_returns_exit_2(tmp_path):
    with _IO() as io:
        rc = cmd_inspect(_ns(plan=str(tmp_path / "nope.json"),
                             empty=False, show_states=False, show_deps=False))
    assert rc == 2 and "not found" in io.stderr.getvalue()


def test_inspect_empty_synthetic_plan():
    with _IO() as io:
        rc = cmd_inspect(_ns(plan=None, empty=True,
                             show_states=False, show_deps=False))
    assert rc == 0
    out = io.stdout.getvalue()
    assert "synthetic-2op" in out and "validation        : OK" in out


def test_inspect_rejects_missing_plan_without_empty():
    with _IO() as io:
        rc = cmd_inspect(_ns(plan=None, empty=False,
                             show_states=False, show_deps=False))
    assert rc == 2 and "must provide" in io.stderr.getvalue()


def test_inspect_show_states_adds_rows():
    with _IO() as io:
        rc = cmd_inspect(_ns(plan=None, empty=True,
                             show_states=True, show_deps=False))
    assert rc == 0
    out = io.stdout.getvalue()
    assert "states:" in out and "st0" in out


def test_inspect_show_deps_adds_rows():
    with _IO() as io:
        rc = cmd_inspect(_ns(plan=None, empty=True,
                             show_states=False, show_deps=True))
    assert rc == 0
    out = io.stdout.getvalue()
    assert "edges:" in out and "op0 -> op1" in out


def test_inspect_validation_failure_returns_exit_3(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(BAD_PLAN))
    with _IO() as io:
        rc = cmd_inspect(_ns(plan=str(p), empty=False,
                             show_states=False, show_deps=False))
    assert rc == 3 and "validation        : FAIL" in io.stdout.getvalue()


def test_inspect_malformed_json_returns_exit_2(tmp_path):
    p = tmp_path / "garbage.json"
    p.write_text("{not json")
    with _IO() as io:
        rc = cmd_inspect(_ns(plan=str(p), empty=False,
                             show_states=False, show_deps=False))
    assert rc == 2 and "invalid JSON" in io.stderr.getvalue()


# --- explain ---------------------------------------------------------------

def test_explain_known_op_produces_non_empty_output():
    with _IO() as io:
        rc = cmd_explain(_ns(op_kind="DenseMatmul", shape=None))
    assert rc == 0
    out = io.stdout.getvalue()
    assert "op_kind: DenseMatmul" in out
    assert "shape_rule:" in out and "deps:" in out
    assert "A (f16)" in out


def test_explain_with_shape_lists_kernels():
    with _IO() as io:
        rc = cmd_explain(_ns(op_kind="RoPE", shape="1024,4096"))
    assert rc == 0
    out = io.stdout.getvalue()
    assert "layout:" in out and "kernel_selection" in out
    assert "rope_metal" in out  # primary when any dim >= 64


def test_explain_unknown_op_returns_exit_4():
    with _IO() as io:
        rc = cmd_explain(_ns(op_kind="BogusOp", shape=None))
    assert rc == 4
    err = io.stderr.getvalue()
    assert "unknown op-kind" in err and "DenseMatmul" in err


def test_explain_malformed_shape_returns_exit_2():
    with _IO() as io:
        rc = cmd_explain(_ns(op_kind="DenseMatmul", shape="1024,abc"))
    assert rc == 2 and "non-integer dimension" in io.stderr.getvalue()


# --- tune ------------------------------------------------------------------

def test_tune_deterministic_with_fixed_seed(tmp_path, monkeypatch):
    monkeypatch.setattr("omlx_research.cli.commands.tune.CACHE_DIR", str(tmp_path))
    with _IO() as io1:
        cmd_tune(_ns(op_kind="DenseMatmul", shape="1024,1024,4096",
                     samples=16, warmup=3, seed=123))
    with _IO() as io2:
        cmd_tune(_ns(op_kind="DenseMatmul", shape="1024,1024,4096",
                     samples=16, warmup=3, seed=123))
    assert io1.stdout.getvalue() == io2.stdout.getvalue()


def test_tune_different_seed_changes_output(tmp_path, monkeypatch):
    monkeypatch.setattr("omlx_research.cli.commands.tune.CACHE_DIR", str(tmp_path))
    with _IO() as a, _IO() as b:
        cmd_tune(_ns(op_kind="DenseMatmul", shape="1024,1024,4096",
                     samples=16, warmup=3, seed=1))
        cmd_tune(_ns(op_kind="DenseMatmul", shape="1024,1024,4096",
                     samples=16, warmup=3, seed=2))
    assert a.stdout.getvalue() != b.stdout.getvalue()


def test_tune_cache_file_exists_and_matches_stdout(tmp_path, monkeypatch):
    monkeypatch.setattr("omlx_research.cli.commands.tune.CACHE_DIR", str(tmp_path))
    with _IO() as io:
        rc = cmd_tune(_ns(op_kind="RMSNorm", shape="1024,4096",
                          samples=8, warmup=1, seed=7))
    assert rc == 0
    payload = io.stdout.getvalue()
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].read_text() == payload


def test_tune_record_has_required_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("omlx_research.cli.commands.tune.CACHE_DIR", str(tmp_path))
    with _IO() as io:
        cmd_tune(_ns(op_kind="Softmax", shape="64,128",
                     samples=10, warmup=1, seed=99))
    rec = json.loads(io.stdout.getvalue())
    for f in ("op_kind", "shape_hash", "samples", "p95_ns"):
        assert f in rec
    assert rec["op_kind"] == "Softmax"
    assert rec["samples"] == 10
    assert rec["shape_hash"] == _shared.shape_hash([64, 128], "Softmax")
    assert isinstance(rec["p95_ns"], (int, float))


def test_tune_malformed_shape_returns_exit_2():
    with _IO() as io:
        rc = cmd_tune(_ns(op_kind="DenseMatmul", shape="1024,xx",
                          samples=4, warmup=1, seed=0))
    assert rc == 2 and "non-integer dimension" in io.stderr.getvalue()


# --- replay ----------------------------------------------------------------

def test_replay_happy_path(tmp_path):
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(GOOD_TRACE))
    with _IO() as io:
        rc = cmd_replay(_ns(trace_file=str(p),
                            filter_rejected=False, filter_selected=False))
    assert rc == 0
    out = io.stdout.getvalue()
    assert "plan_id : plan-test-1" in out
    assert "op0_metal" in out and "memory_pressure" in out


def test_replay_filter_rejected(tmp_path):
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(GOOD_TRACE))
    with _IO() as io:
        rc = cmd_replay(_ns(trace_file=str(p),
                            filter_rejected=True, filter_selected=False))
    assert rc == 0
    out = io.stdout.getvalue()
    assert "rejected candidates" not in out
    assert "selected candidate" in out


def test_replay_filter_selected(tmp_path):
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(GOOD_TRACE))
    with _IO() as io:
        rc = cmd_replay(_ns(trace_file=str(p),
                            filter_rejected=False, filter_selected=True))
    assert rc == 0
    out = io.stdout.getvalue()
    assert "rejected candidates" in out
    assert "selected candidate:" not in out


def test_replay_missing_file_returns_exit_5(tmp_path):
    with _IO() as io:
        rc = cmd_replay(_ns(trace_file=str(tmp_path / "missing.json"),
                            filter_rejected=False, filter_selected=False))
    assert rc == 5 and "not found" in io.stderr.getvalue()


def test_replay_malformed_json_returns_exit_6(tmp_path):
    p = tmp_path / "trace.json"
    p.write_text("{not json")
    with _IO() as io:
        rc = cmd_replay(_ns(trace_file=str(p),
                            filter_rejected=False, filter_selected=False))
    assert rc == 6 and "invalid JSON" in io.stderr.getvalue()


# --- compare ---------------------------------------------------------------

def test_compare_happy_path(tmp_path):
    pa = tmp_path / "a.json"
    pb = tmp_path / "b.json"
    pa.write_text(json.dumps(_trace("plan-x", "op0", "metal", 1000)))
    pb.write_text(json.dumps(_trace("plan-x", "op0", "mlx",   1100)))
    with _IO() as io:
        rc = cmd_compare(_ns(trace_a=str(pa), trace_b=str(pb)))
    assert rc == 0
    parsed = json.loads(io.stdout.getvalue())
    assert parsed["plan_id"] == "plan-x"
    assert parsed["a"]["selected_kernel"] == "metal"
    assert parsed["b"]["selected_kernel"] == "mlx"
    assert parsed["a"]["latency_p95_ns"] == 1000
    assert parsed["b"]["latency_p95_ns"] == 1100


def test_compare_different_plan_id_returns_exit_7(tmp_path):
    pa = tmp_path / "a.json"
    pb = tmp_path / "b.json"
    pa.write_text(json.dumps(_trace("plan-x", "op0", "metal", 1000)))
    pb.write_text(json.dumps(_trace("plan-y", "op0", "mlx",   1100)))
    with _IO() as io:
        rc = cmd_compare(_ns(trace_a=str(pa), trace_b=str(pb)))
    assert rc == 7 and "plan_id mismatch" in io.stderr.getvalue()


def test_compare_missing_file_returns_exit_5(tmp_path):
    pa = tmp_path / "a.json"
    pa.write_text(json.dumps(_trace("p", "o", "k", 1)))
    with _IO() as io:
        rc = cmd_compare(_ns(trace_a=str(pa), trace_b=str(tmp_path / "missing")))
    assert rc == 5


def test_compare_malformed_json_returns_exit_2(tmp_path):
    pa = tmp_path / "a.json"; pa.write_text("{nope")
    pb = tmp_path / "b.json"; pb.write_text(json.dumps(_trace("p", "o", "k", 1)))
    with _IO():
        rc = cmd_compare(_ns(trace_a=str(pa), trace_b=str(pb)))
    assert rc == 2


# --- evidence --------------------------------------------------------------

def test_evidence_happy_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(GOOD_PLAN))
    with _IO() as io:
        rc = cmd_evidence(_ns(plan_file=str(p), _argv=[str(p)]))
    assert rc == 0
    bundle = json.loads(io.stdout.getvalue())
    for k in ("plan", "validation", "kernel_trace", "tuning_record",
              "sys_info", "git_rev", "unix_ts", "command"):
        assert k in bundle
    assert bundle["plan"]["plan_id"] == "plan-test-1"
    assert bundle["validation"]["ok"] is True
    written = list(tmp_path.glob("omlx-evidence-*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text()) == bundle


def test_evidence_missing_plan_file_returns_exit_8(tmp_path):
    with _IO() as io:
        rc = cmd_evidence(_ns(plan_file=str(tmp_path / "missing.json"),
                              _argv=[]))
    assert rc == 8 and "not found" in io.stderr.getvalue()


# --- main() dispatch -------------------------------------------------------

def test_main_rejects_unknown_subcommand():
    with pytest.raises(SystemExit) as e:
        main(["no-such-command"])
    assert e.value.code != 0


def test_main_routes_known_subcommand_by_name(capsys, tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(GOOD_PLAN))
    rc = main(["inspect", str(p)])
    assert rc == 0
    assert "plan-test-1" in capsys.readouterr().out
