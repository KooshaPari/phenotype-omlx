# Testing Strategy

## Validation

- Verify exactly one `sladge.net` reference in README.
- Review `git diff --stat`.
- Confirm worktree status before commit.

## Commands

```bash
rg -n "sladge.net" README.md
git diff --stat
git status --short --untracked-files=all
```
