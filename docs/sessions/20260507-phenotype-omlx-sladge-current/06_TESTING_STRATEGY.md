# Testing Strategy

Planned checks:

- `git diff --check`: passed.
- README/session badge search with `rg`: passed.
- `task lint`: blocked by sandbox denial opening
  `/Users/kooshapari/.cache/uv/sdists-v9/.git`.
- `uv run --no-sync ruff check .`: same sandbox blocker.
- `uv run --no-sync ruff format --check .`: same sandbox blocker.
- `uv run --no-sync pytest --collect-only -q`: same sandbox blocker.
