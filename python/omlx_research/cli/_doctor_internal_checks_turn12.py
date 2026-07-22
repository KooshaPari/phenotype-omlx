"""Internal doctor checks added 2026-07-19 — turn-12 batch.

Sibling module to :mod:`omlx_research.cli._doctor_internal_checks`.
Hosts the two turn-12 structural-invariant checks that were added on
top of the turn-10 batch. Kept in its own module so each module
stays at a manageable size (the turn-10 module is already ~570
lines and putting another ~280 lines there would push the file well
past the 500-line module-size cap documented in
``AGENTS.md`` §"Polyglot language policy").

Two checks live here:

1. ``cargo_workspace_crate_count_at_least_15`` — parses
   ``perf-core/Cargo.toml``'s ``[workspace].members`` list and counts
   the declared crates. Threshold ladder: ``≥15 → PASS``,
   ``10..14 → WARN``, ``<10 → FAIL``. Defends against workspace
   shrinkage (a workspace dropping below 10 indicates an
   eviction accident; below 15 is a strong drift signal given
   the polyglot expansion).
2. ``ddm_continuous_schedule_variants_at_least_4`` — parses
   ``perf-core/kernel-registry/tests/sota_operators/discrete_diffusion_oracle.rs``
   and counts distinct ``ContinuousScheduleKind::`` variant
   references (Linear, Cosine, Sqrt, Sigmoid). Threshold ladder:
   ``≥4 → PASS``, ``2..3 → WARN``, ``<2 → FAIL``. Pins the turn-11
   schedule-coverage surface against accidental enum shrinkage.

Both checks are entirely INTERNAL — they inspect on-disk source
files in the repo and never touch external dependencies
(``mlx_lm``, ``turboquant``, network, etc.). Same property as the
turn-10 batch: a missing file degrades to WARN — never FAIL.

Adding these two (alongside the four from the turn-10 batch) lifts
the live doctor check count from 19 → 25. Per the threshold-raise
rule from turn-9 resume notes §7, the drift detector config
``doctor_config.toml`` moves from ``min_check_count = 23`` to
``min_check_count = 25`` in the same batch. See the lockstep rule
documented in that file.

Re-exports
----------
The two check callables are re-exported by
:mod:`omlx_research.cli._doctor_checks` so the existing
``checks.<name>`` access pattern keeps working. Callers should not
import this module directly.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

try:
    # Python 3.11+ stdlib
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover — Python <3.11 fallback
    import tomli as tomllib  # type: ignore[import-untyped,no-redef]

from ._doctor_registry import register_check
from ._doctor_shared import (
    FAIL,
    PASS,
    WARN,
    Check,
    project_root,
)


__all__ = [
    "cargo_workspace_crate_count_at_least_15",
    "ddm_continuous_schedule_variants_at_least_4",
]


# ---------------------------------------------------------------------------
# Check 1 — Cargo workspace member count
# ---------------------------------------------------------------------------


#: Relative path to the perf-core Cargo workspace manifest. Parsed for
#: the ``[workspace].members`` list.
_CARGO_WORKSPACE_TOML_REL_PATH: str = "perf-core/Cargo.toml"

#: Regex matching the start of the workspace ``members = [`` block.
#: Captured so a multi-line ``members`` block can be sliced cleanly
#: out of the file even when ``tomllib`` is unavailable (e.g. the
#: ``tomli`` fallback import failed).
_WORKSPACE_MEMBERS_BLOCK_START_RE = re.compile(
    r"members\s*=\s*\[",
    re.MULTILINE,
)

#: Regex matching a single quoted string inside the multiline
#: ``members = [ "..." ]`` block. Conservative: only matches plain
#: double-quoted crate path strings (no escape sequences, no
#: globs). Cargo allows globs (``members = ["crates/*"]``) but the
#: phenotype-omlx workspace uses an explicit list of 20 crates, so
#: this regex is sufficient and avoids glob-expansion complexity.
_WORKSPACE_MEMBER_ENTRY_RE = re.compile(r'"([^"]+)"')

#: PASS threshold — at least this many declared workspace members.
_CARGO_WORKSPACE_CRATE_THRESHOLD_PASS: int = 15

#: WARN floor — counts below this escalate to FAIL.
_CARGO_WORKSPACE_CRATE_THRESHOLD_FAIL: int = 10


def _count_workspace_members(path: str) -> Tuple[bool, str]:
    """Best-effort workspace-member count for ``perf-core/Cargo.toml``.

    Prefers :mod:`tomllib` (stdlib in 3.11+) for an exact structural
    parse; falls back to a multiline regex slicing the
    ``members = [ ... ]`` block when ``tomllib`` is unavailable.
    Either way, returns a deduplicated count.

    Returns ``(success, label)`` — on success, ``label`` carries the
    count summary; on failure (file missing, OS error, malformed TOML)
    the success flag is ``False`` and ``label`` carries the exception
    class + message. Never raises.
    """
    if not os.path.isfile(path):
        return False, f"{_CARGO_WORKSPACE_TOML_REL_PATH} not on disk"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"
    members: List[str] = []
    # Primary path: structural TOML parse. tomllib cannot raise here
    # because the text was already read; any decode failure is caught
    # below and we fall through to the regex.
    parsed_members: list[str] | None = None
    try:
        data = tomllib.loads(text)
        ws = data.get("workspace") if isinstance(data, dict) else None
        if isinstance(ws, dict):
            raw = ws.get("members")
            if isinstance(raw, list):
                parsed_members = [str(m) for m in raw if isinstance(m, str)]
    except Exception:
        parsed_members = None
    if parsed_members is not None:
        members = parsed_members
    else:
        # Regex fallback: locate the ``members = [`` opener, then walk
        # forward to the matching ``]`` (bracket-counting across the
        # text) and collect every double-quoted crate path. We skip
        # over characters inside quoted strings so a ``]`` embedded
        # in a quoted path doesn't prematurely end the block.
        m = _WORKSPACE_MEMBERS_BLOCK_START_RE.search(text)
        if not m:
            return False, (
                f"{_CARGO_WORKSPACE_TOML_REL_PATH} has no `members = [` "
                f"block (could not parse workspace)"
            )
        i = text.find("[", m.end() - 1)
        if i < 0:
            return False, "malformed `members = [` block (no opening `[`)"
        depth = 0
        in_str = False
        escape = False
        end_idx = -1
        for j in range(i, len(text)):
            c = text[j]
            if in_str:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    end_idx = j
                    break
        if end_idx < 0:
            return False, "malformed `members = [` block (no closing `]`)"
        block = text[i + 1 : end_idx]
        # Only count non-empty, unique members (the same crate path
        # can appear twice if someone copy-pasted; dedupe).
        seen: set[str] = set()
        for raw in _WORKSPACE_MEMBER_ENTRY_RE.findall(block):
            stripped = raw.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                members.append(stripped)
    count = len(members)
    return True, (
        f"found {count} declared workspace member(s) "
        f"in {_CARGO_WORKSPACE_TOML_REL_PATH}"
    )


@register_check
def cargo_workspace_crate_count_at_least_15() -> Check:
    """Verify ``perf-core/Cargo.toml`` declares >= 15 workspace members.

    Parses the ``[workspace].members`` list and counts the declared
    crates. The threshold guards against workspace shrinkage:
    dropping below 10 members is treated as an eviction accident
    (FAIL); the band ``[10, 15)`` is a strong drift signal given
    the polyglot expansion that landed across turns 8–11.

    Status ladder:

    - ``distinct_members >= 15`` → PASS
    - ``10 <= distinct_members < 15`` → WARN
    - ``distinct_members < 10`` → FAIL
    - file-not-found / OS error / malformed TOML → WARN (never FAIL
      the doctor from a missing internal file)
    """
    path = os.path.join(project_root(), _CARGO_WORKSPACE_TOML_REL_PATH)
    desc = (
        f"Cargo workspace member count >= {_CARGO_WORKSPACE_CRATE_THRESHOLD_PASS} "
        f"(fails < {_CARGO_WORKSPACE_CRATE_THRESHOLD_FAIL}, "
        f"warns < {_CARGO_WORKSPACE_CRATE_THRESHOLD_PASS})"
    )
    ok, label = _count_workspace_members(path)
    if not ok:
        return Check(
            id="cargo_workspace_crate_count_at_least_15",
            description=desc,
            status=WARN,
            details=(
                f"{_CARGO_WORKSPACE_TOML_REL_PATH} unreadable: {label} — "
                f"check skipped (never FAIL on missing internal file)"
            ),
        )
    try:
        count = int(label.split()[1])
    except (IndexError, ValueError):
        return Check(
            id="cargo_workspace_crate_count_at_least_15",
            description=desc,
            status=FAIL,
            details=(
                f"unexpected label format from _count_workspace_members: {label!r}"
            ),
        )
    if count >= _CARGO_WORKSPACE_CRATE_THRESHOLD_PASS:
        status = PASS
    elif count >= _CARGO_WORKSPACE_CRATE_THRESHOLD_FAIL:
        status = WARN
    else:
        status = FAIL
    return Check(
        id="cargo_workspace_crate_count_at_least_15",
        description=desc,
        status=status,
        details=label,
    )


# ---------------------------------------------------------------------------
# Check 2 — DDM ContinuousScheduleKind variant count
# ---------------------------------------------------------------------------


#: Relative path to the discrete-diffusion oracle source. Carries the
#: ``ContinuousScheduleKind`` enum introduced in turn-11.
_DDM_ORACLE_REL_PATH: str = (
    "perf-core/kernel-registry/tests/sota_operators/discrete_diffusion_oracle.rs"
)

#: Regex matching a single ``ContinuousScheduleKind::Variant`` reference
#: in the discrete-diffusion-oracle file. Matches the full enum path
#: so we only count references inside the schedule enum/match arms
#: and not false positives elsewhere in the file (e.g. ``Sigmoid``
#: as a method name elsewhere).
_DDM_SCHEDULE_KIND_RE = re.compile(r"\bContinuousScheduleKind::([A-Za-z][A-Za-z0-9_]*)")

#: PASS threshold — at least this many distinct
#: ``ContinuousScheduleKind`` variants (Linear, Cosine, Sqrt, Sigmoid).
_DDM_SCHEDULE_VARIANT_THRESHOLD_PASS: int = 4

#: WARN floor — counts below this escalate to FAIL.
_DDM_SCHEDULE_VARIANT_THRESHOLD_FAIL: int = 2


def _count_ddm_schedule_variants(path: str) -> Tuple[bool, str]:
    """Best-effort distinct ``ContinuousScheduleKind`` variant count.

    Parses ``discrete_diffusion_oracle.rs`` and counts every distinct
    ``ContinuousScheduleKind::<Variant>`` reference across the file.
    The variants are deduplicated (set), so the count is the number
    of distinct *declared* variants — not the number of references.
    The enum body declares each once; the ``alpha_at`` ``match`` arm
    references each once; we dedupe across all of them.

    Returns ``(success, label)`` — on success, ``label`` carries the
    count summary; on failure (file missing, OS error) the success
    flag is ``False`` and ``label`` carries the exception class +
    message. Never raises.
    """
    if not os.path.isfile(path):
        return False, f"{_DDM_ORACLE_REL_PATH} not on disk"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"
    variants: set[str] = set()
    for m in _DDM_SCHEDULE_KIND_RE.finditer(text):
        variants.add(m.group(1))
    return True, (f"found {len(variants)} distinct ContinuousScheduleKind variant(s)")


@register_check
def ddm_continuous_schedule_variants_at_least_4() -> Check:
    """Verify the DDM oracle exposes >= 4 distinct ContinuousScheduleKind variants.

    Parses ``perf-core/kernel-registry/tests/sota_operators/discrete_diffusion_oracle.rs``
    and counts distinct ``ContinuousScheduleKind::<Variant>``
    references. The threshold pins the turn-11 schedule surface
    (Linear, Cosine, Sqrt, Sigmoid) against accidental enum
    shrinkage — e.g. someone removing the Sqrt/Sigmoid variants when
    refactoring the alpha math.

    Status ladder:

    - ``distinct_variants >= 4`` → PASS
    - ``2 <= distinct_variants < 4`` → WARN
    - ``distinct_variants < 2`` → FAIL
    - file-not-found / OS error → WARN (never FAIL the doctor from a
      missing internal file)
    """
    path = os.path.join(project_root(), _DDM_ORACLE_REL_PATH)
    desc = (
        f"DDM ContinuousScheduleKind variant count >= "
        f"{_DDM_SCHEDULE_VARIANT_THRESHOLD_PASS} "
        f"(fails < {_DDM_SCHEDULE_VARIANT_THRESHOLD_FAIL}, "
        f"warns < {_DDM_SCHEDULE_VARIANT_THRESHOLD_PASS})"
    )
    ok, label = _count_ddm_schedule_variants(path)
    if not ok:
        return Check(
            id="ddm_continuous_schedule_variants_at_least_4",
            description=desc,
            status=WARN,
            details=(
                f"{_DDM_ORACLE_REL_PATH} unreadable: {label} — "
                f"check skipped (never FAIL on missing internal file)"
            ),
        )
    try:
        count = int(label.split()[1])
    except (IndexError, ValueError):
        return Check(
            id="ddm_continuous_schedule_variants_at_least_4",
            description=desc,
            status=FAIL,
            details=(
                f"unexpected label format from _count_ddm_schedule_variants: {label!r}"
            ),
        )
    if count >= _DDM_SCHEDULE_VARIANT_THRESHOLD_PASS:
        status = PASS
    elif count >= _DDM_SCHEDULE_VARIANT_THRESHOLD_FAIL:
        status = WARN
    else:
        status = FAIL
    return Check(
        id="ddm_continuous_schedule_variants_at_least_4",
        description=desc,
        status=status,
        details=label,
    )
