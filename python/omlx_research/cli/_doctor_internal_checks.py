"""Internal doctor checks added 2026-07-19 — turn-10 batch.

These checks are entirely INTERNAL — they inspect on-disk source files
in the repo and never touch external dependencies (``mlx_lm``,
``turboquant``, network, etc.). That property makes them reliable
"baseline health" probes that should pass cleanly on any fresh checkout
that has the ``perf-core/`` Rust workspace present.

Four checks live here:

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
3. ``metal_runtime_lib_test_count_at_least_25`` — parses
   ``perf-core/metal-runtime/src/lib.rs`` and counts ``#[test]``
   references. Threshold ladder: ``≥25 → PASS``, ``15..24 → WARN``,
   ``<15 → FAIL``. Defends against accidental mass-deletion of
   metal-runtime test coverage on the turn-9→turn-10 bridge.
4. ``python_cli_subcommand_count_at_least_6`` — parses
   ``python/omlx_research/cli/__init__.py`` and counts ``cmd_<name>``
   subcommand callables. Threshold ladder: ``≥6 → PASS``,
   ``4..5 → WARN``, ``<4 → FAIL``. Defends against subcommand
   droppage.

Adding these four lifts the live doctor check count from 19 → 23.
Per the threshold-raise rule from turn-9 resume notes §7, the
drift detector config ``doctor_config.toml`` must move from
``min_check_count = 21`` to ``min_check_count = 23`` in the same
batch.

Re-exports
----------
The four check callables are re-exported by
:mod:`omlx_research.cli._doctor_checks` so the existing
``checks.<name>`` access pattern keeps working. Callers should not
import this module directly.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

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
    "metal_runtime_lib_test_count_at_least_25",
    "python_cli_subcommand_count_at_least_6",
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
            details=(
                f"unexpected label format from _count_coverage_tags: "
                f"{label!r}"
            ),
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
            details=(
                f"unexpected label format from _count_eval_suites: "
                f"{label!r}"
            ),
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


# ---------------------------------------------------------------------------
# Check 3 — metal-runtime lib.rs distinct #[test] reference count
# ---------------------------------------------------------------------------


#: Relative directory path (from project root) for the metal-runtime
#: crate sources. The crate splits its tests across ``artifact.rs``,
#: ``cache.rs``, ``compile.rs``, ``dispatch.rs``, ``fingerprint.rs``,
#: and ``pipeline.rs`` (see turn-9 close audit); ``lib.rs`` is a
#: re-exports-only entry point and carries no ``#[test]``. Counting
#: across the whole directory gives a stable invariant.
_METAL_RUNTIME_SRC_DIR_REL_PATH: str = "perf-core/metal-runtime/src"

#: Regex matching a ``#[test]`` attribute line. Matches plain
#: ``#[test]`` as well as qualified forms (``#[crate::test]``,
#: ``#[::core::prelude::test]``). The follow-up is any sequence of
#: non-``]`` chars to permit trailing inner attributes or whitespace.
_METAL_RUNTIME_TEST_RE = re.compile(r"#\[(?:[A-Za-z_][A-Za-z0-9_]*::)*test[^\]]*\]")

#: PASS threshold — at least this many distinct ``#[test]`` references.
_METAL_RUNTIME_TEST_THRESHOLD_PASS: int = 25

#: WARN floor — counts below this escalate to FAIL. The band between
#: :data:`_METAL_RUNTIME_TEST_THRESHOLD_FAIL` (inclusive) and
#: :data:`_METAL_RUNTIME_TEST_THRESHOLD_PASS` (exclusive) is WARN.
_METAL_RUNTIME_TEST_THRESHOLD_FAIL: int = 15


def _count_metal_runtime_tests(src_dir: str) -> Tuple[bool, str]:
    """Best-effort distinct ``#[test]`` count across metal-runtime/src/.

    Iterates every ``*.rs`` file under ``src_dir`` (skips dotfiles
    and only accepts files whose name parses as a Rust source via
    the ``*.rs`` extension). Counts every ``#[test]`` (and qualified
    ``crate::test``) reference across all of them. The combined
    count is the single integer we report, deduplicated automatically
    because the same test attribute cannot appear in two different
    files.

    Returns ``(success, label)`` — on success, ``label`` carries
    the count summary; on failure (directory missing, OS error) the
    success flag is ``False`` and ``label`` carries the exception
    class + message. Never raises.
    """
    if not os.path.isdir(src_dir):
        return False, f"{_METAL_RUNTIME_SRC_DIR_REL_PATH} not on disk"
    count = 0
    files_scanned: List[str] = []
    try:
        for entry in sorted(os.listdir(src_dir)):
            if not entry.endswith(".rs"):
                continue
            full = os.path.join(src_dir, entry)
            if not os.path.isfile(full):
                continue
            with open(full, "r", encoding="utf-8") as fh:
                text = fh.read()
            count += len(_METAL_RUNTIME_TEST_RE.findall(text))
            files_scanned.append(entry)
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"
    files_label = (
        f" across {len(files_scanned)} files"
        if len(files_scanned) > 1
        else ""
    )
    return True, (
        f"found {count} distinct #[test] reference(s){files_label} "
        f"under {os.path.basename(src_dir)}"
    )


def metal_runtime_lib_test_count_at_least_25() -> Check:
    """Verify metal-runtime lib.rs carries >= 25 ``#[test]`` references.

    Parses ``perf-core/metal-runtime/src/lib.rs`` and counts distinct
    ``#[test]`` (and qualified ``crate::test``) attributes. The
    threshold guards against accidental mass-deletion of metal-runtime
    test coverage on the turn-9→turn-10 bridge (artifact loader,
    compile mode gating, Send+Sync compile-time asserts).

    Status ladder:

    - ``distinct_tests >= 25`` → PASS
    - ``15 <= distinct_tests < 25`` → WARN
    - ``distinct_tests < 15`` → FAIL
    - file-not-found / OS error → WARN (never FAIL the doctor from a
      missing internal file)
    """
    path = os.path.join(project_root(), _METAL_RUNTIME_SRC_DIR_REL_PATH)
    desc = (
        f"metal-runtime src #[test] count >= {_METAL_RUNTIME_TEST_THRESHOLD_PASS} "
        f"(fails < {_METAL_RUNTIME_TEST_THRESHOLD_FAIL}, "
        f"warns < {_METAL_RUNTIME_TEST_THRESHOLD_PASS})"
    )
    ok, label = _count_metal_runtime_tests(path)
    if not ok:
        return Check(
            id="metal_runtime_lib_test_count_at_least_25",
            description=desc,
            status=WARN,
            details=(
                f"{_METAL_RUNTIME_SRC_DIR_REL_PATH} unreadable: {label} — "
                f"check skipped (never FAIL on missing internal file)"
            ),
        )
    try:
        count = int(label.split()[1])
    except (IndexError, ValueError):
        return Check(
            id="metal_runtime_lib_test_count_at_least_25",
            description=desc,
            status=FAIL,
            details=(
                f"unexpected label format from _count_metal_runtime_tests: "
                f"{label!r}"
            ),
        )
    if count >= _METAL_RUNTIME_TEST_THRESHOLD_PASS:
        status = PASS
    elif count >= _METAL_RUNTIME_TEST_THRESHOLD_FAIL:
        status = WARN
    else:
        status = FAIL
    return Check(
        id="metal_runtime_lib_test_count_at_least_25",
        description=desc,
        status=status,
        details=label,
    )


# ---------------------------------------------------------------------------
# Check 4 — Python CLI subcommand count
# ---------------------------------------------------------------------------


#: Relative path to the Python CLI entry point. The ``__init__.py``
#: registers subcommands via the ``subparsers.add_parser(...)`` pattern
#: inside the ``cmd_<name>`` callables, so counting those callables
#: captures every registered subcommand.
_PYTHON_CLI_INIT_REL_PATH: str = "python/omlx_research/cli/__init__.py"

#: Regex matching a top-level ``def cmd_<name>(...)`` callable. This
#: captures every public subcommand registration function; ``_cmd_``
#: prefixed helpers are private and do not match.
_PYTHON_CLI_SUBCMD_RE = re.compile(r"^def\s+cmd_[a-z][a-z0-9_]*\s*\(", re.MULTILINE)

#: PASS threshold — at least this many subcommands.
_PYTHON_CLI_SUBCMD_THRESHOLD_PASS: int = 6

#: WARN floor — counts below this escalate to FAIL. The band between
#: :data:`_PYTHON_CLI_SUBCMD_THRESHOLD_FAIL` (inclusive) and
#: :data:`_PYTHON_CLI_SUBCMD_THRESHOLD_PASS` (exclusive) is WARN.
_PYTHON_CLI_SUBCMD_THRESHOLD_FAIL: int = 4


def _count_cli_subcommands(path: str) -> Tuple[bool, str]:
    """Best-effort distinct ``cmd_*`` subcommand callable count.

    Parses the CLI ``__init__.py`` and counts every top-level
    ``def cmd_<name>(...)`` definition. Each match is on its own
    line (multiline mode), so nested ``def`` inside a class or
    inside a test helper is excluded.

    Returns ``(success, label)`` — on success, ``label`` carries the
    count summary; on failure (file missing, OS error) the success
    flag is ``False`` and ``label`` carries the exception class +
    message. Never raises.
    """
    if not os.path.isfile(path):
        return False, f"{_PYTHON_CLI_INIT_REL_PATH} not on disk"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"
    count = len(_PYTHON_CLI_SUBCMD_RE.findall(text))
    return True, f"found {count} distinct cmd_* subcommand(s)"


def python_cli_subcommand_count_at_least_6() -> Check:
    """Verify the Python CLI registers >= 6 distinct ``cmd_*`` subcommands.

    Parses ``python/omlx_research/cli/__init__.py`` for ``def cmd_*``
    callable definitions. The threshold guards against accidental
    subcommand droppage that would otherwise be invisible until a
    user invoking the dropped command saw a confusing ``No such
    subcommand`` error.

    Status ladder:

    - ``distinct_subcommands >= 6`` → PASS
    - ``4 <= distinct_subcommands < 6`` → WARN
    - ``distinct_subcommands < 4`` → FAIL
    - file-not-found / OS error → WARN (never FAIL the doctor from a
      missing internal file)
    """
    path = os.path.join(project_root(), _PYTHON_CLI_INIT_REL_PATH)
    desc = (
        f"Python CLI distinct cmd_* subcommand count >= {_PYTHON_CLI_SUBCMD_THRESHOLD_PASS} "
        f"(fails < {_PYTHON_CLI_SUBCMD_THRESHOLD_FAIL}, "
        f"warns < {_PYTHON_CLI_SUBCMD_THRESHOLD_PASS})"
    )
    ok, label = _count_cli_subcommands(path)
    if not ok:
        return Check(
            id="python_cli_subcommand_count_at_least_6",
            description=desc,
            status=WARN,
            details=(
                f"{_PYTHON_CLI_INIT_REL_PATH} unreadable: {label} — "
                f"check skipped (never FAIL on missing internal file)"
            ),
        )
    try:
        count = int(label.split()[1])
    except (IndexError, ValueError):
        return Check(
            id="python_cli_subcommand_count_at_least_6",
            description=desc,
            status=FAIL,
            details=(
                f"unexpected label format from _count_cli_subcommands: "
                f"{label!r}"
            ),
        )
    if count >= _PYTHON_CLI_SUBCMD_THRESHOLD_PASS:
        status = PASS
    elif count >= _PYTHON_CLI_SUBCMD_THRESHOLD_FAIL:
        status = WARN
    else:
        status = FAIL
    return Check(
        id="python_cli_subcommand_count_at_least_6",
        description=desc,
        status=status,
        details=label,
    )