"""``tune`` — produce a deterministic synthetic TuningRecord.

We don't actually measure anything here — this is the *research* CLI. We
fake a measurement loop, seeded by ``--seed``, that produces a stable
``TuningRecord`` JSON with all the fields a real tuner would emit
(``latency_ns``, ``p50_ns``, ``p95_ns``, etc.).

The record is printed to stdout AND written to::

    ~/.cache/omlx/tune/<op-kind>-<shape-hash>.json

Exit codes:
    0 — success (always, unless ``--shape`` is malformed)
    2 — malformed ``--shape``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from typing import Any

from ._shared import parse_shape, shape_hash


CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "omlx", "tune")


def _percentile(values: list[float], q: float) -> float:
    """Return the q-th percentile (0..100) of a non-empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _synthetic_measurements(
    op_kind: str,
    shape: list[int] | None,
    samples: int,
    warmup: int,
    seed: int,
) -> dict[str, Any]:
    """Produce a deterministic record given the same inputs.

    The base latency is a stable hash of (op_kind, shape, seed). We then
    apply a +/-5% jitter seeded by ``seed`` so the output is reproducible.
    """
    payload = f"{op_kind}|{','.join(str(d) for d in (shape or []))}|base"
    base_ns = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16) % 5000 + 500

    rng = random.Random(seed)
    # Warmup: not recorded.
    for _ in range(max(0, warmup)):
        rng.gauss(0.0, 1.0)
    # Samples.
    latencies = [base_ns * (1.0 + rng.uniform(-0.05, 0.05)) for _ in range(max(1, samples))]

    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)
    mean = sum(latencies) / len(latencies)
    return {
        "op_kind": op_kind,
        "shape": shape,
        "shape_hash": shape_hash(shape, op_kind),
        "samples": samples,
        "warmup": warmup,
        "seed": seed,
        "base_latency_ns": base_ns,
        "mean_latency_ns": round(mean, 2),
        "p50_ns": round(p50, 2),
        "p95_ns": round(p95, 2),
        "p99_ns": round(p99, 2),
        "latency_ns": round(p50, 2),  # canonical "chosen" latency
        "kernels_evaluated": [
            f"{op_kind.lower()}_metal",
            f"{op_kind.lower()}_mlx",
            f"{op_kind.lower()}_fallback",
        ],
        "selected_kernel": f"{op_kind.lower()}_metal",
    }


def cmd_tune(args: argparse.Namespace) -> int:
    """CLI entry point: ``tune <op-kind> [--shape ...] [--samples N] [--warmup N] [--seed N]``"""
    try:
        shape = parse_shape(args.shape)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    record = _synthetic_measurements(
        op_kind=args.op_kind,
        shape=shape,
        samples=args.samples,
        warmup=args.warmup,
        seed=args.seed,
    )

    payload = json.dumps(record, indent=2, sort_keys=True)
    sys.stdout.write(payload + "\n")
    sys.stdout.flush()

    # Cache write. mkdir -p style.
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(
            CACHE_DIR,
            f"{args.op_kind}-{record['shape_hash']}.json",
        )
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
        record["_cache_path"] = cache_path  # exposed for tests only
    except OSError as e:
        print(f"warning: could not write cache file: {e}", file=sys.stderr)

    return 0
