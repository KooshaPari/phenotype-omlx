# Canonical repo cutover — phenotype-omlx

**Use [`KooshaPari/phenotype-omlx`](https://github.com/KooshaPari/phenotype-omlx) only.**

This repository is the single canonical home for the Phenotype oMLX stack.
Do not open new work, PRs, or clones against the archived supersets.

## Archived supersets

| Repo | Status | Role |
| --- | --- | --- |
| [`phenotype-omlx-tmp`](https://github.com/KooshaPari/phenotype-omlx-tmp) | Archived on GitHub | Historical working copy; all branches already exist on canonical |
| [`phenotype-omlx-temp`](https://github.com/KooshaPari/phenotype-omlx-temp) | Archived on GitHub | Historical working copy; all branches already exist on canonical |

Each archived repo root carries a `SUPERSEDED.md` pointing here. Treat those
trees as read-only history — never as the active development target.

## Local layout

| Path | Purpose |
| --- | --- |
| `repos/phenotype-omlx` | Canonical clone — stay on `main`; pull / merge only |
| `repos/worktrees/phenotype-omlx/<topic>` | Feature work, quality gates, PR prep |

Example:

```bash
git -C /Users/kooshapari/CodeProjects/Phenotype/repos/phenotype-omlx \
  worktree add ../worktrees/phenotype-omlx/<topic> -b <topic>
```

Do not author feature work in the canonical folder or in any leftover
`phenotype-omlx-tmp` / `phenotype-omlx-temp` checkout.

## Remotes (local cutover)

If a local clone still points at an archived URL, retarget `origin` to
canonical and keep the old URL under a named remote (for example
`archived-tmp`) — do not delete remotes.
