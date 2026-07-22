"""Internal doctor checks added 2026-07-19 — turn-10 (carved-out half).

After turn-12 discovered :mod:`omlx_research.cli._doctor_internal_checks`
was 576 lines (over the 500-line cap), the two longest turn-10 checks
were carved out into this sibling module:

1. ``metal_runtime_lib_test_count_at_least_25`` — parses
   ``perf-core/metal-runtime/src/`` and counts ``#[test]`` references
   across all ``*.rs`` files in the directory. Threshold ladder:
   ``≥25 → PASS``, ``15..24 → WARN``, ``<15 → FAIL``. Defends against
   accidental mass-deletion of metal-runtime test coverage on the
   turn-9→turn-10 bridge.
2. ``python_cli_subcommand_count_at_least_6`` — parses
   ``python/omlx_research/cli/__init__.py`` and counts ``cmd_<name>``
   subcommand callables (including re-exports for sibling modules).
   Threshold ladder: ``≥6 → PASS``, ``4..5 → WARN``, ``<4 → FAIL``.
   Defends against subcommand droppage.

The two checks that stayed in the original module are
:func:`coverage_tag_count_at_least_25` and
:func:`eval_harness_suite_count_at_least_4`. All four turn-10 checks
remain INTERNAL structural invariants and behave identically. The
turn-12 additions still live in
:mod:`omlx_research.cli._doctor_internal_checks_turn12`. All three
sibling modules together hold the six INTERNAL check callables and
lift the live doctor check count from 19 → 25.

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
from typing import List, Tuple

from ._doctor_registry import register_check
from ._doctor_shared import (
    FAIL,
    PASS,
    WARN,
    Check,
    project_root,
)


__all__ = [
    "metal_runtime_lib_test_count_at_least_25",
    "python_cli_subcommand_count_at_least_6",
]


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
        f" across {len(files_scanned)} files" if len(files_scanned) > 1 else ""
    )
    return True, (
        f"found {count} distinct #[test] reference(s){files_label} "
        f"under {os.path.basename(src_dir)}"
    )


@register_check
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
                f"unexpected label format from _count_metal_runtime_tests: {label!r}"
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
_PYTHON_CLI_SUBCMD_DEF_RE = re.compile(r"^def\s+cmd_[a-z][a-z0-9_]*\s*\(", re.MULTILINE)

#: Regex matching a ``from .<module> import (... cmd_<name> ...)`` re-export
#: line in ``cli/__init__.py``. After a subcommand's implementation is
#: carved out into a sibling module (turn-9 extracted ``_cmd_eval.py``;
#: this turn extracted ``_cmd_inference.py``), the CLI package still
#: exposes the same ``cmd_*`` symbols via a re-export line of this shape.
#: Counting those re-exports preserves the check's stated intent ("defend
#: against subcommand droppage from the CLI surface") without forcing
#: every future extraction to leave a stub ``def`` behind just to keep
#: this counter happy.
_PYTHON_CLI_SUBCMD_REEXPORT_NAMES_RE = re.compile(
    r"^from\s+\.[A-Za-z_][A-Za-z0-9_]*\s+import\s*\([^)]*\bcmd_[a-z][a-z0-9_]*",
    re.MULTILINE,
)
_PYTHON_CLI_SUBCMD_NAME_RE = re.compile(r"\bcmd_[a-z][a-z0-9_]*")

#: PASS threshold — at least this many subcommands.
_PYTHON_CLI_SUBCMD_THRESHOLD_PASS: int = 6

#: WARN floor — counts below this escalate to FAIL. The band between
#: :data:`_PYTHON_CLI_SUBCMD_THRESHOLD_FAIL` (inclusive) and
#: :data:`_PYTHON_CLI_SUBCMD_THRESHOLD_PASS` (exclusive) is WARN.
_PYTHON_CLI_SUBCMD_THRESHOLD_FAIL: int = 4


def _count_cli_subcommands(path: str) -> Tuple[bool, str]:
    """Best-effort distinct ``cmd_*`` subcommand callable count.

    Parses the CLI ``__init__.py`` and counts every top-level
    ``def cmd_<name>(...)`` definition PLUS every ``cmd_*`` name
    re-exported via a ``from .<sibling_module> import (..., cmd_<n>, ...)``
    line. Both shapes register a subcommand on the CLI surface — the
    former is the inline pattern, the latter is the documented
    pattern for ``cmd_*`` symbols whose implementation lives in a
    sibling module (turn-9 extracted ``_cmd_eval.py``; this turn
    extracted ``_cmd_inference.py``). Counting both keeps the
    invariant meaningful across future extractions.

    Names are deduplicated (set) so the same ``cmd_*`` symbol that
    is both defined inline AND re-exported (shouldn't happen in
    practice) is counted exactly once.

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
    names: set[str] = set()
    for m in _PYTHON_CLI_SUBCMD_DEF_RE.finditer(text):
        # Extract the name out of "def cmd_<name>(" for set membership.
        snippet = m.group(0)
        name_match = _PYTHON_CLI_SUBCMD_NAME_RE.search(snippet)
        if name_match:
            names.add(name_match.group(0))
    for m in _PYTHON_CLI_SUBCMD_REEXPORT_NAMES_RE.finditer(text):
        names.update(_PYTHON_CLI_SUBCMD_NAME_RE.findall(m.group(0)))
    count = len(names)
    return True, f"found {count} distinct cmd_* subcommand(s)"


@register_check
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
            details=(f"unexpected label format from _count_cli_subcommands: {label!r}"),
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
