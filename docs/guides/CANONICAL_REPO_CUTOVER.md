# Canonical repo cutover — phenotype-omlx

**Use [`KooshaPari/phenotype-omlx`](https://github.com/KooshaPari/phenotype-omlx) only.**

This repository is the single canonical home for the Phenotype oMLX stack.
Do not open new work, PRs, or clones against the archived supersets.

## Archived supersets (do not use)

| Name | Status | Role |
| --- | --- | --- |
| `zz-archive-phenotype-omlx-tmp` | Archived remote / local shelf | Historical only — never clone or push new work |
| `zz-archive-phenotype-omlx-temp` | Archived remote / local shelf | Historical only — never clone or push new work |
| Legacy GitHub: `phenotype-omlx-tmp` / `phenotype-omlx-temp` | Archived on GitHub | Same lineage; renamed under `zz-archive-*` locally |

Each archived repo root may carry a `SUPERSEDED.md` pointing here. Treat those
trees as read-only history — never as the active development target.

## Local layout (Apple / Phenotype hub)

| Path | Purpose |
| --- | --- |
| `repos/phenotype-omlx` | Canonical clone — always on `main`; pull / merge only |
| `repos/worktrees/phenotype-omlx/<topic>` | Feature work, quality gates, PR prep, analysis |

Create a worktree from the canonical clone:

```bash
git -C /Users/kooshapari/CodeProjects/Phenotype/repos/phenotype-omlx \
  worktree add ../worktrees/phenotype-omlx/<topic> -b <topic>
```

Or use the thin helper (same layout):

```bash
# from any phenotype-omlx checkout
./scripts/worktree_add.sh <topic>
```

Do **not** author feature work in the canonical folder, under
`repos/worktrees/phenotype-omlx-tmp`, or in any leftover tmp/temp checkout.

## Remotes (local cutover)

If a local clone still points at an archived URL, retarget `origin` to
`git@github.com:KooshaPari/phenotype-omlx.git` and keep the old URL under a
named remote (`tmp` / `temp` → `zz-archive-phenotype-omlx-*`) — do not delete
remotes without confirmation.

## Stale hubs (inventory only — do not delete)

As of 2026-07-21, the preferred hub is `repos/worktrees/phenotype-omlx/`.
These leftover locations may still exist on disk; **do not delete worktrees or
branches without explicit confirmation**:

| Path | Notes |
| --- | --- |
| `repos/worktrees/phenotype-omlx-tmp` | Former tmp hub — absent on this machine at inventory time; do not recreate |
| `repos/worktrees/phenotype-omlx-temp` | Former temp hub — absent at inventory time; do not recreate |
| `repos/worktrees/phenotype-omlx-recovered` | Recovery shelf — treat as stale; migrate topics into `worktrees/phenotype-omlx/` if still needed |
| `repos/.archive/phenotype-omlx-tmp-broken-wts-20260721` | Archived broken worktrees from the old tmp hub (read-only history) |
| `repos/zz-archive-phenotype-omlx-tmp` | Local archive clone of the tmp remote |
| `repos/zz-archive-phenotype-omlx-temp` | Local archive clone of the temp remote |

When flushing stale hubs: confirm no other agent is using the path, then
review each topic into `main` via PR before removal.
