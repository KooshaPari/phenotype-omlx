"""Internal doctor checks added 2026-07-19 — turn-10 batch (post-split).

Two checks live here:

1. ``coverage_tag_count_at_least_25`` — parses
   ``perf-core/kernel-registry/tests/sota_operators/coverage_matrix.rs``
   and counts distinct tag-style declarations. Threshold ladder:
   ``≥25 → PASS``, ``15..24 → WARN``, ``<15 → FAIL``. Subprocess or
   file-not-found conditions degrade to WARN — never FAIL — so a
   missing file in a partial checkout does not break the doctor.
2. ``eval_harness_suite_count_at_least_4`` — parses
   ``perf-core/eval-harness/src/lib.rs`` and counts distinct ``Suite``
   variants. Threshold ladder: ``≥4 → PASS``, ``2..3 → WARN``,
   ``<2 → FAIL``. Same WARN-on-missing-file policy.

The other two turn-10 checks — ``metal_runtime_lib_test_count_at_least_25``
and ``python_cli_subcommand_count_at_least_6`` — were carved out into the
sibling module :mod:`omlx_research.cli._doctor_internal_checks_split`
so this module stays under the 500-line cap (was 576 lines). Both
sibling modules are still INTERNAL structural invariants and behave
identically to the two checks documented here.

Turn-12 added two more checks — ``cargo_workspace_crate_count_at_least_15``
and ``ddm_continuous_schedule_variants_at_least_4`` — which live in
:mod:`omlx_research.cli._doctor_internal_checks_turn12`. All three
sibling modules together hold the six INTERNAL check callables, and
lift the live doctor check count from 19 → 25.

Per the threshold-raise rule from turn-9 resume notes §7, the drift
detector config ``doctor_config.toml`` must move from
``min_check_count = 21`` to ``min_check_count = 25`` in the same
batch. See the lockstep rule documented in that file.

Re-exports
----------
The two check callables defined here are re-exported by
:mod:`omlx_research.cli._doctor_checks` so the existing
``checks.<name>`` access pattern keeps working. Callers should not
import this module directly.
"""

from __future__ import annotations

import os
import re
from typing import Tuple

from ._doctor_registry import register_check
from ._doctor_shared import (
    FAIL,
    PASS,
    WARN,
    Check,
    project_root,
)


__all__ = [
    "coverage_tag_count_at_least_25",
    "eval_harness_suite_count_at_least_4",
]


# ---------------------------------------------------------------------------
# Check 1 — coverage_matrix.rs distinct tag count
# ---------------------------------------------------------------------------


#: Relative path (from project root) to the coverage matrix source file.
_COVERAGE_MATRIX_REL_PATH: str = (
    "perf-core/kernel-registry/tests/sota_operators/coverage_matrix.rs"
)

#: Regex matching one tag-style tuple declaration in coverage_matrix.rs:
#: a top-level ``(  "<Identifier>", ... )`` row in a const table like
#: ``MATRIX_FAMILIES`` / ``DISPATCH_ENVELOPE_FAMILIES`` / ``NAMED_KERNEL_OPS``.
#: The pattern is intentionally tolerant of leading whitespace and any
#: trailing payload so it captures every row across all the coverage
#: tables in the file. The first capture group is the identifier string.
_COVERAGE_TAG_LINE_RE = re.compile(r'^\s*\(\s*"([^"]+)"')

#: PASS threshold — at least this many distinct tag declarations must
#: be present. Below this, the check escalates.
_COVERAGE_TAG_THRESHOLD_PASS: int = 25

#: WARN floor — counts below this escalate to FAIL. The band between
#: :data:`_COVERAGE_TAG_THRESHOLD_FAIL` (inclusive) and
#: :data:`_COVERAGE_TAG_THRESHOLD_PASS` (exclusive) is WARN.
_COVERAGE_TAG_THRESHOLD_FAIL: int = 15


def _count_coverage_tags(path: str) -> Tuple[bool, str]:
    """Best-effort distinct-tag count for ``coverage_matrix.rs``.

    Returns ``(success, label)`` where:

    - On success, ``label`` is a human-readable count summary
      (e.g. ``"found 43 distinct tag(s)"``).
    - On failure (file missing, OS error), ``success`` is ``False`` and
      ``label`` carries the exception class + message so the caller can
      surface it in the WARN details.

    The function never raises — every error path returns ``(False, ...)``.
    """
    if not os.path.isfile(path):
        return False, f"{_COVERAGE_MATRIX_REL_PATH} not on disk"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"
    tags: set[str] = set()
    for line in text.splitlines():
        m = _COVERAGE_TAG_LINE_RE.match(line)
        if not m:
            continue
        tag = m.group(1).strip()
        if tag:
            tags.add(tag)
    return True, f"found {len(tags)} distinct tag(s) in {os.path.basename(path)}"


@register_check
def coverage_tag_count_at_least_25() -> Check:
    """Verify the SOTA coverage matrix carries >= 25 distinct tag declarations.

    Parses ``perf-core/kernel-registry/tests/sota_operators/coverage_matrix.rs``
    and counts distinct tag identifiers across all the coverage tables
    (``MATRIX_FAMILIES``, ``OPERATOR_KIND_COVERED``,
    ``DISPATCH_ENVELOPE_FAMILIES``, ``NAMED_KERNEL_OPS``, etc.). The
    threshold guards against accidental mass-deletion of coverage rows.

    Status ladder:

    - ``distinct_tags >= 25`` → PASS
    - ``15 <= distinct_tags < 25`` → WARN
    - ``distinct_tags < 15`` → FAIL
    - file-not-found / OS error → WARN (never FAIL the doctor from a
      missing internal file)
    """
    path = os.path.join(project_root(), _COVERAGE_MATRIX_REL_PATH)
    desc = (
        f"SOTA coverage matrix distinct tag count >= {_COVERAGE_TAG_THRESHOLD_PASS} "
        f"(fails < {_COVERAGE_TAG_THRESHOLD_FAIL}, warns < {_COVERAGE_TAG_THRESHOLD_PASS})"
    )
    ok, label = _count_coverage_tags(path)
    if not ok:
        return Check(
            id="coverage_tag_count_at_least_25",
            description=desc,
            status=WARN,
            details=(
                f"{_COVERAGE_MATRIX_REL_PATH} unreadable: {label} — "
                f"check skipped (never FAIL on missing internal file)"
            ),
        )
    # Parse the leading integer out of "found N distinct tag(s) ..." for
    # the threshold comparison. Defensive — fall back to FAIL rather
    # than raising if the label format is unexpected (which would
    # indicate a bug in this module).
    try:
        count = int(label.split()[1])
    except (IndexError, ValueError):
        return Check(
            id="coverage_tag_count_at_least_25",
            description=desc,
            status=FAIL,
            details=(f"unexpected label format from _count_coverage_tags: {label!r}"),
        )
    if count >= _COVERAGE_TAG_THRESHOLD_PASS:
        status = PASS
    elif count >= _COVERAGE_TAG_THRESHOLD_FAIL:
        status = WARN
    else:
        status = FAIL
    return Check(
        id="coverage_tag_count_at_least_25",
        description=desc,
        status=status,
        details=label,
    )


# ---------------------------------------------------------------------------
# Check 2 — eval-harness Suite variant count
# ---------------------------------------------------------------------------


#: Relative path to the eval-harness library entry point. The Suite
#: enum is re-exported from this module, so counting ``Suite::Variant``
#: references here captures every declared variant.
_EVAL_HARNESS_LIB_REL_PATH: str = "perf-core/eval-harness/src/lib.rs"

#: Regex matching any ``Suite::<Identifier>`` reference. Used to count
#: distinct declared variants across the whole file (the enum body
#: itself plus the ``match`` arms in ``Suite::as_str``).
_SUITE_VARIANT_RE = re.compile(r"\bSuite::([A-Za-z][A-Za-z0-9_]*)")

#: PASS threshold — at least this many distinct Suite variants.
_EVAL_SUITE_THRESHOLD_PASS: int = 4

#: WARN floor — counts below this escalate to FAIL. The band between
#: :data:`_EVAL_SUITE_THRESHOLD_FAIL` (inclusive) and
#: :data:`_EVAL_SUITE_THRESHOLD_PASS` (exclusive) is WARN.
_EVAL_SUITE_THRESHOLD_FAIL: int = 2


def _count_eval_suites(path: str) -> Tuple[bool, str]:
    """Best-effort distinct-suite count for ``eval-harness/src/lib.rs``.

    Counts every ``Suite::Variant`` reference in the file. Variants are
    deduplicated, so the count is the number of *distinct* declared
    variants — not the number of references (the enum body declares
    each once, the ``as_str`` match arms reference each once, and we
    dedupe across all of them).

    Returns ``(success, label)`` — on success, ``label`` carries the
    count summary; on failure (file missing, OS error) the success
    flag is ``False`` and ``label`` carries the exception class +
    message. Never raises.
    """
    if not os.path.isfile(path):
        return False, f"{_EVAL_HARNESS_LIB_REL_PATH} not on disk"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"
    variants: set[str] = set()
    for m in _SUITE_VARIANT_RE.finditer(text):
        variants.add(m.group(1))
    return True, f"found {len(variants)} distinct Suite variant(s)"


@register_check
def eval_harness_suite_count_at_least_4() -> Check:
    """Verify the eval-harness crate exposes >= 4 distinct Suite variants.

    Parses ``perf-core/eval-harness/src/lib.rs`` and counts distinct
    ``Suite::Variant`` references. The CLI's ``eval`` subcommand
    routes to these suites (mmlu, gpqa, terminal-bench, perplexity).

    Status ladder:

    - ``distinct_suites >= 4`` → PASS
    - ``2 <= distinct_suites < 4`` → WARN
    - ``distinct_suites < 2`` → FAIL
    - file-not-found / OS error → WARN (never FAIL the doctor from a
      missing internal file)
    """
    path = os.path.join(project_root(), _EVAL_HARNESS_LIB_REL_PATH)
    desc = (
        f"eval-harness distinct Suite variant count >= {_EVAL_SUITE_THRESHOLD_PASS} "
        f"(fails < {_EVAL_SUITE_THRESHOLD_FAIL}, warns < {_EVAL_SUITE_THRESHOLD_PASS})"
    )
    ok, label = _count_eval_suites(path)
    if not ok:
        return Check(
            id="eval_harness_suite_count_at_least_4",
            description=desc,
            status=WARN,
            details=(
                f"{_EVAL_HARNESS_LIB_REL_PATH} unreadable: {label} — "
                f"check skipped (never FAIL on missing internal file)"
            ),
        )
    try:
        count = int(label.split()[1])
    except (IndexError, ValueError):
        return Check(
            id="eval_harness_suite_count_at_least_4",
            description=desc,
            status=FAIL,
            details=(f"unexpected label format from _count_eval_suites: {label!r}"),
        )
    if count >= _EVAL_SUITE_THRESHOLD_PASS:
        status = PASS
    elif count >= _EVAL_SUITE_THRESHOLD_FAIL:
        status = WARN
    else:
        status = FAIL
    return Check(
        id="eval_harness_suite_count_at_least_4",
        description=desc,
        status=status,
        details=label,
    )
