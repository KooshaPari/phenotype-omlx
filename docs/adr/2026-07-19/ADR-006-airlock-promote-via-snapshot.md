# ADR-006 — Airlock v2 "promote" via `snapshot` (no `promote` subcommand exists)

**Status:** Accepted
**Date:** 2026-07-19
**Related:** turn-7 resume notes (Airlock v2 install), turn-8 resume notes (Lane E),
`scripts/snapshot.sh` (commit `9069ce9`)

## Context

Turn 8's lane plan included "Lane E: scripts/promote.sh wires airlock-v2 promote".
This was written before the Airlock v2 CLI surface was re-verified. The phrase
"airlock-v2 promote" implies a subcommand that, on inspection, does not exist.

`airlock-v2 --help` (verified twice this session: at turn-8 start and again
on the res-and-correct pass) shows exactly the following subcommand set:

```
register | unregister | list | status | snapshot | autocommit
| cleanup | daemon | audit | restore | quickstatus | help
```

There is no `promote` subcommand. `restore` is documented as "Run all cycles
once on the live registry. Used by the `airlock-v2 autocommit` and `airlock-v2
cleanup` subcommands. The `restore` command restores a `wip/<date>-<uuid>`
branch onto a target ref (a no-op alias of `snapshot` for parity)."

So Airlock v2's promotion model is:

1. The daemon (or `snapshot` one-shot) captures work-in-progress by creating
   + pushing a `wip/<date>-<uuid>` branch.
2. The operator promotes that wip branch to a stable ref with a normal
   `git merge wip/<date>-<uuid>` (or equivalent PR flow).

There is no in-tool promotion step. The wip branch IS the promotion
candidate; merging it IS the promotion. `scripts/promote.sh` would therefore
either (a) be a misleading name for a wrapper around `airlock-v2 snapshot`,
or (b) attempt to call a non-existent subcommand and fail.

## Decision

1. **Do not write `scripts/promote.sh`.** It would either rename
   `airlock-v2 snapshot` (misleading) or invoke a missing subcommand
   (broken).
2. **Write `scripts/snapshot.sh`.** This is what "promote the current
   working tree into Airlock v2's pipeline" actually means. The script
   runs six CI gates (airlock-v2 installed, tree clean, Rust tests,
   clippy, pytest, doctor with 0 FAIL) and, only if all pass, calls
   `airlock-v2 snapshot`. `DRY_RUN=1` short-circuits the actual call so
   the script is testable without daemon side effects.
3. **Promotion to a stable ref is a `git merge` of the resulting
   `wip/<date>-<uuid>` branch.** No new tooling required. If/when we
   want automation of that step, it belongs in a separate script
   (e.g. `scripts/promote.sh`) that wraps `git merge wip/<date>-<uuid>`
   — but only after the wip-branch model is the dominant promotion
   mechanism in this repo, which today it is not (we have 9 commits on
   `chore/archive-no-simd-lib-rs-2026-07-18` with no wip merges to date).

## Consequences

Positive:
- `scripts/snapshot.sh` matches the tool's actual semantics.
- The naming is consistent with `airlock-v2 snapshot`, so anyone reading
  the script knows exactly what it does without having to cross-reference
  the airlock docs.
- The 6-gate CI wrapper is reusable: any future "promote" automation can
  call `scripts/snapshot.sh` to capture wip, then `git merge` to release.

Negative:
- The lane name in the turn-8 plan ("promote.sh") is misleading and
  surfaces again as a system_reminder every time the todo list is
  materialized. Cancellation of that todo item is the correct response.
- An ADR is heavier than a one-line code change. We accept that cost
  because the why-not-promote decision needs to be auditable: future
  agents or humans reading the session notes will see "scripts/promote.sh"
  mentioned and need an answer for why it doesn't exist.

## Verification

- `airlock-v2 --help` (output captured above; both verifications this
  session agreed).
- `DRY_RUN=1 bash scripts/snapshot.sh` → exit 0, prints "DRY RUN — would
  have called: `airlock-v2 snapshot`".
- Working-tree dirty run of `bash scripts/snapshot.sh` (no `DRY_RUN`)
  → exits non-zero at the "tree clean" gate, confirming the gate fires.
- Commit `9069ce9` (turn 8, Lane E): "feat(scripts): `snapshot.sh` —
  gated `airlock-v2 snapshot` wrapper (6 CI gates, `DRY_RUN` support)".

## Reopen conditions

This ADR should be revisited if:

1. A future Airlock v2 release introduces an actual `promote` subcommand.
   In that case, `scripts/snapshot.sh` may be renamed `scripts/promote.sh`
   or both may coexist (snapshot for capture, promote for release).
2. The repo's promotion flow becomes predominantly wip-branch-based (e.g.
   if we adopt a `release/wip-merge.sh` script that auto-merges the
   latest wip branch on a schedule). At that point, write
   `scripts/promote.sh` as a thin wrapper over that script + the
   `scripts/snapshot.sh` gate set.