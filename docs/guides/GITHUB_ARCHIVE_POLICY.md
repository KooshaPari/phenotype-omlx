# GitHub archive policy (STRICT)

**Never delete GitHub repositories on this account.**

## Required pattern

When a repo is superseded:

1. Mirror all needed history/branches into the canonical home.
2. Rename to `zz-archive-<original-name>` (sorts to bottom of org list).
3. Archive the repo (read-only).
4. Description: `SUPERSEDED YYYY-MM-DD — … mirrored to <canonical>. Do not push here.`
5. Homepage → canonical URL.
6. Optional root `SUPERSEDED.md`.

## Forbidden

- `gh repo delete`
- UI "Delete this repository"
- Any action that destroys the only copy of history

## oMLX stand-ins (2026-07-22)

| Role | Repo |
|------|------|
| Active | `KooshaPari/phenotype-omlx` |
| Archive | `KooshaPari/zz-archive-phenotype-omlx-tmp` |
| Archive | `KooshaPari/zz-archive-phenotype-omlx-temp` |
