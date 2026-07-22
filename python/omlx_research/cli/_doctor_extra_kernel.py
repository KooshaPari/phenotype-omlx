"""Kernel-adjacent doctor checks added 2026-07-19.

This module is one of three siblings carved out of the original
:mod:`omlx_research.cli._doctor_extra_checks` module (turn-9 split,
module-size sweep). It owns the regress-baseline dispatch envelope
check (the kernel-registry / dispatch-budget probe) plus the
``omlx_research.__version__`` package version check.

Layout reminder
---------------
- :mod:`omlx_research.cli._doctor_extra_niah` — NIAH benchmark check
  + ``niah_results.json`` helpers
- :mod:`omlx_research.cli._doctor_extra_eval` — eval-harness check
- :mod:`omlx_research.cli._doctor_extra_kernel` — package version +
  regress-baseline dispatch envelope

The regress-baseline envelope check reuses the
:func:`omlx_research.cli._doctor_extra_niah._load_niah_results`
helper to key off the populated ``niah_results.json`` target table.
The version check is a package-level probe that anchors every doctor
report.

Each check returns a :class:`omlx_research.cli.doctor.Check`; the
``run_doctor`` orchestrator wraps each call in a broad ``Exception``
guard so a single broken check can never abort the report.
"""

from __future__ import annotations

from typing import Optional

from ._doctor_extra_niah import _load_niah_results  # shared niah_results.json helper
from ._doctor_registry import register_check
from ._doctor_shared import (
    FAIL,
    PASS,
    WARN,
    Check,
    project_root,
)


__all__ = [
    "omlx_research_version",
    "regress_baseline_dispatch_envelope",
]


# ---------------------------------------------------------------------------
# omlx_research.__version__
# ---------------------------------------------------------------------------


@register_check
def omlx_research_version() -> Check:
    """Report the installed ``omlx_research.__version__`` (always pass)."""
    desc = "omlx_research package version (omlx_research.__version__)"
    try:
        import omlx_research  # type: ignore  # noqa: F401
    except Exception as e:
        return Check(
            id="omlx_research_version",
            description=desc,
            status=FAIL,
            details=f"import failed: {type(e).__name__}: {e}",
        )
    version = getattr(omlx_research, "__version__", None)
    if not isinstance(version, str) or not version:
        return Check(
            id="omlx_research_version",
            description=desc,
            status=WARN,
            details="omlx_research is importable but __version__ is missing or empty",
        )
    return Check(
        id="omlx_research_version",
        description=desc,
        status=PASS,
        details=version,
    )


# ---------------------------------------------------------------------------
# regress-baseline dispatch envelope
# ---------------------------------------------------------------------------


# Mirror the floor constant from _doctor_extra_niah so the check's
# description string renders the same value the loader enforces. The
# import-side check still reads through _load_niah_results so there
# is no risk of drift between the two.
_NIAH_TARGET_ROW_FLOOR = 25  # 5 context lengths × 5 seeds


@register_check
def regress_baseline_dispatch_envelope() -> Check:
    """Probe the regression-baseline dispatch envelope.

    The envelope that anchors NIAH regression comparisons lives in
    ``niah_results.json``: a per-(kernel_id, context_length, seed)
    table of expected pass rates against which future real runs are
    diffed. The check PASSes when the file is populated with ≥
    :data:`_NIAH_TARGET_ROW_FLOOR` rows (the canonical floor — 5
    context lengths × 5 seeds).

    As a secondary (supplementary) signal, the check also tries the
    Python ``regress_baseline`` extension and surfaces its
    ``dispatch_budget((m=64, n=64, k=64))`` ceiling. That extension
    is consumed by the regress-baseline Rust unit tests but is not
    bound into the Python wheel by default, so its absence is a
    legitimate non-error state.
    """
    desc = (
        "regression-baseline dispatch envelope: niah_results.json "
        f"populated (≥ {_NIAH_TARGET_ROW_FLOOR} target rows); "
        "dispatch_budget((m=64, n=64, k=64)) finite as a secondary signal"
    )

    # Sub-check 1: niah_results.json populated (primary PASS signal).
    results_loaded, results_label, target_count = _load_niah_results()
    results_ok = results_loaded and target_count >= _NIAH_TARGET_ROW_FLOOR

    # Sub-check 2: regress_baseline Python extension (supplementary).
    rb_status: Optional[str] = None
    rb_budget: Optional[int] = None
    try:
        import regress_baseline  # type: ignore  # noqa: F401
    except Exception as e:
        rb_status = (
            f"regress_baseline Python extension not importable "
            f"({type(e).__name__}: {e}); rust extension not built"
        )
    else:
        # Match the Rust crate's public surface: either a module-level
        # `dispatch_budget(m, n, k)` callable, or `dispatch_budget(ShapeKey)`.
        budget_fn = getattr(regress_baseline, "dispatch_budget", None)
        if budget_fn is None:
            rb_status = "regress_baseline imports but exposes no `dispatch_budget`"
        else:
            try:
                shape_key = getattr(regress_baseline, "ShapeKey", None)
                if shape_key is None:
                    budget_value = int(budget_fn(64, 64, 64))
                else:
                    budget_value = int(budget_fn(shape_key(64, 64, 64)))
            except Exception as e:
                rb_status = f"dispatch_budget raised: {type(e).__name__}: {e}"
            else:
                if budget_value <= 0:
                    rb_status = (
                        f"dispatch_budget((m=64, n=64, k=64)) = "
                        f"{budget_value} (non-positive)"
                    )
                else:
                    rb_budget = budget_value

    if results_ok:
        if rb_budget is not None:
            details = (
                f"{results_label} (≥ {_NIAH_TARGET_ROW_FLOOR} floor); "
                f"dispatch_budget((m=64, n=64, k=64)) = {rb_budget}"
            )
        elif rb_status is not None:
            details = f"{results_label} (≥ {_NIAH_TARGET_ROW_FLOOR} floor); {rb_status}"
        else:
            details = f"{results_label} (≥ {_NIAH_TARGET_ROW_FLOOR} floor)"
        return Check(
            id="regress_baseline_dispatch_envelope",
            description=desc,
            status=PASS,
            details=details,
        )

    # JSON not populated — fall back to the extension result.
    if rb_budget is not None:
        details = (
            f"niah_results.json not populated yet ({results_label}); "
            f"regress_baseline extension present, dispatch_budget = "
            f"{rb_budget}"
        )
    elif rb_status is not None:
        details = f"niah_results.json not populated yet ({results_label}); {rb_status}"
    else:
        details = (
            f"niah_results.json not populated yet ({results_label}); "
            f"regress_baseline extension not exercised"
        )
    return Check(
        id="regress_baseline_dispatch_envelope",
        description=desc,
        status=WARN,
        details=details,
    )
