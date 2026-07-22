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
- UI “Delete this repository”
- Any action that destroys the only copy of history

## oMLX stand-ins (2026-07-22)

| Role | Repo |
|------|------|
| Active | `KooshaPari/phenotype-omlx` |
| Archive | `KooshaPari/zz-archive-phenotype-omlx-tmp` |
| Archive | `KooshaPari/zz-archive-phenotype-omlx-temp` |
EOF
git add docs/guides/GITHUB_ARCHIVE_POLICY.md
git -c user.email=kooshapari@local -c user.name=kooshapari commit -m "docs: STRICT no-delete; zz-archive pattern only"
git push -u origin docs/github-archive-policy
gh pr create --repo KooshaPari/phenotype-omlx --base main --head docs/github-archive-policy \
  --title "docs: STRICT GitHub no-delete / zz-archive policy" \
  --body "## Summary
- Codifies org rule: never delete repos; use \`zz-archive-*\` + archive only.
- Documents oMLX tmp/temp stand-ins after restore.

## Note
A prior desk agent incorrectly deleted stand-ins; they were restored by Support and re-applied as zz-archive.

Made with [Cursor](https://cursor.com)" 2>&1 | tail -20