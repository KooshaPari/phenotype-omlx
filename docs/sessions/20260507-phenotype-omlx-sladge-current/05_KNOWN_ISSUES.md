# Known Issues

- The older prepared branch `docs/sladge-badge` is stale and should remain
  historical evidence only.
- Full runtime validation may require Apple Silicon MLX dependencies and local
  model assets; record exact blockers if they appear.
- `task lint`, `uv run --no-sync ruff check .`, `uv run --no-sync ruff format
  --check .`, and `uv run --no-sync pytest --collect-only -q` are blocked in
  this sandbox by `Operation not permitted` while opening
  `/Users/kooshapari/.cache/uv/sdists-v9/.git`.
