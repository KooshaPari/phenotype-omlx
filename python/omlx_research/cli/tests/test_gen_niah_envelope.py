"""Tests for :mod:`scripts.gen_niah_envelope`.

TDD-first coverage for the NIAH regression envelope generator. Each
test exercises one shape of the CLI surface (defaults, expanded
flag, validation, determinism, schema) so the doctor PASS path
("niah_results.json has at least 250 target rows") can be re-asserted
in isolation without round-tripping through the live JSON file.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Import the script as a module without forcing it into a package. The
# generator is intentionally script-style (no ``scripts/__init__.py``);
# importlib keeps it testable in-process so we can call build_envelope
# directly without spawning a subprocess for the unit tests.
# ---------------------------------------------------------------------------

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "scripts"
    / "gen_niah_envelope.py"
)


def _load_gen():
    spec = importlib.util.spec_from_file_location(
        "gen_niah_envelope", _SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


gen = _load_gen()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_args_produce_125_rows():
    """No flags -> 5 ctx x 5 seeds x 5 kernels = 125 rows.

    This locks the back-compat path: the turn-9 contract (the
    25-row floor) keeps resolving to a 125-row envelope when the
    generator is invoked bare.
    """
    payload = gen.build_envelope(
        gen.DEFAULT_CONTEXT_LENGTHS,
        gen.DEFAULT_SEEDS,
        gen.DEFAULT_KERNELS,
    )
    assert len(payload["targets"]) == 125


def test_expanded_contexts_produce_250_rows():
    """The 10-context scale yields 250 rows (10 x 5 x 5)."""
    payload = gen.build_envelope(
        gen.EXPANDED_CONTEXT_LENGTHS,
        gen.DEFAULT_SEEDS,
        gen.DEFAULT_KERNELS,
    )
    assert len(payload["targets"]) == 250
    assert payload["context_lengths"] == gen.EXPANDED_CONTEXT_LENGTHS
    assert len(payload["context_lengths"]) == 10


def test_default_kernels_are_unchanged():
    """The kernel set is the canonical 5; the generator must not
    add or rename kernels (those changes belong in scripts/niah_benchmark.py
    and the doctor check, not here)."""
    assert gen.DEFAULT_KERNELS == [
        "baseline_fp16",
        "turbo_asymmetric",
        "turbo_symmetric",
        "turbo4",
        "mlx_native_kv4",
    ]
    assert len(gen.DEFAULT_KERNELS) == 5


def test_default_seeds_are_unchanged():
    """Seeds are a fixed tuple so a re-run is byte-identical."""
    assert gen.DEFAULT_SEEDS == [7, 19, 42, 73, 101]


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_validate_args_rejects_zero_contexts():
    import pytest
    with pytest.raises(SystemExit):
        gen.validate_args(0, 5, 5)


def test_validate_args_rejects_negative_seeds():
    import pytest
    with pytest.raises(SystemExit):
        gen.validate_args(5, -1, 5)


def test_validate_args_rejects_zero_kernels():
    import pytest
    with pytest.raises(SystemExit):
        gen.validate_args(5, 5, 0)


def test_validate_args_accepts_positive_counts():
    """The happy path through validate_args is silent — any other
    behavior would mask a regression in the CLI loop."""
    gen.validate_args(10, 5, 5)


# ---------------------------------------------------------------------------
# Schema integrity
# ---------------------------------------------------------------------------


_REQUIRED_FIELDS = ("pass_rate", "target", "context_length", "seed", "kernel_id")


def test_every_row_has_all_required_fields():
    """Every target row carries the 5 canonical fields, no nulls,
    pass_rate in [0.5, 0.99], context_length a positive int."""
    payload = gen.build_envelope(
        gen.EXPANDED_CONTEXT_LENGTHS,
        gen.DEFAULT_SEEDS,
        gen.DEFAULT_KERNELS,
    )
    for i, row in enumerate(payload["targets"]):
        assert isinstance(row, dict), f"target[{i}] is {type(row).__name__}"
        missing = set(_REQUIRED_FIELDS) - set(row.keys())
        assert not missing, f"target[{i}] missing fields {missing}"
        for f in _REQUIRED_FIELDS:
            assert row[f] is not None, f"target[{i}].{f} is None"
        pr = float(row["pass_rate"])
        assert 0.5 <= pr <= 0.99, f"target[{i}] pass_rate={pr} out of [0.5, 0.99]"
        assert isinstance(row["context_length"], int)
        assert row["context_length"] > 0
        assert isinstance(row["seed"], int)
        assert isinstance(row["kernel_id"], str)
        assert row["target"] == row["pass_rate"]


def test_top_level_envelope_has_canonical_keys():
    """The envelope root must carry every header field the doctor and
    the existing tests already key off of."""
    payload = gen.build_envelope(
        gen.EXPANDED_CONTEXT_LENGTHS,
        gen.DEFAULT_SEEDS,
        gen.DEFAULT_KERNELS,
    )
    assert payload["schema_version"] == 1
    assert payload["kind"] == "niah_target_rows"
    assert isinstance(payload["generated_at"], str)
    assert payload["model"] == gen.DEFAULT_MODEL
    assert isinstance(payload["context_lengths"], list)
    assert isinstance(payload["kernels"], list)
    assert isinstance(payload["seeds"], list)
    assert isinstance(payload["targets"], list)


def test_row_count_equals_ctx_times_seed_times_kernel():
    """Sanity: row count == len(context_lengths) * len(seeds) * len(kernels)
    for any input combo. Sources the context list from the union of the
    default and expanded scales so 10-element probes don't silently
    truncate."""
    full_context_pool = list(
        dict.fromkeys(
            list(gen.DEFAULT_CONTEXT_LENGTHS) + list(gen.EXPANDED_CONTEXT_LENGTHS)
        )
    )
    for ctx_n, seed_n, ker_n in [
        (5, 5, 5),
        (10, 5, 5),
        (3, 2, 4),
        (1, 1, 1),
    ]:
        ctxs = full_context_pool[:ctx_n]
        seeds = gen.DEFAULT_SEEDS[:seed_n]
        kernels = gen.DEFAULT_KERNELS[:ker_n]
        payload = gen.build_envelope(ctxs, seeds, kernels)
        assert len(payload["targets"]) == ctx_n * seed_n * ker_n, (
            f"got {len(payload['targets'])}, want {ctx_n * seed_n * ker_n}"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_runs_produce_identical_bytes(tmp_path):
    """Same inputs -> byte-identical JSON. This is the property
    the doctor regression contract depends on."""
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"

    rc_a = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--expanded-contexts",
            "--out", str(out_a),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rc_b = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--expanded-contexts",
            "--out", str(out_b),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert rc_a.returncode == 0
    assert rc_b.returncode == 0
    assert out_a.read_bytes() == out_b.read_bytes(), (
        "two runs produced different bytes — envelope is non-deterministic"
    )

    data = json.loads(out_a.read_text())
    assert len(data["targets"]) == 250


def test_in_process_determinism():
    """In-process build_envelope is also byte-stable across calls."""
    p1 = gen.build_envelope(
        gen.EXPANDED_CONTEXT_LENGTHS,
        gen.DEFAULT_SEEDS,
        gen.DEFAULT_KERNELS,
    )
    p2 = gen.build_envelope(
        gen.EXPANDED_CONTEXT_LENGTHS,
        gen.DEFAULT_SEEDS,
        gen.DEFAULT_KERNELS,
    )
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_expanded_contexts_writes_250_rows(tmp_path):
    """End-to-end: scripts/gen_niah_envelope.py --expanded-contexts writes
    a 250-row envelope. The CLI surface the doctor depends on."""
    out = tmp_path / "niah_results.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--expanded-contexts",
            "--out", str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert len(data["targets"]) == 250


def test_cli_default_writes_125_rows(tmp_path):
    """End-to-end: bare invocation -> 125 rows (5x5x5)."""
    out = tmp_path / "niah_results.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--out", str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert len(data["targets"]) == 125


def test_cli_explicit_contexts_arg(tmp_path):
    """Explicit --contexts 1024 4096 16384 with default seeds/kernels
    -> 3 * 5 * 5 = 75 rows. Confirms the list-arg shape works."""
    out = tmp_path / "niah_results.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--contexts", "1024", "4096", "16384",
            "--out", str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert len(data["targets"]) == 75
    assert data["context_lengths"] == [1024, 4096, 16384]
