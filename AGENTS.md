# AGENTS.md — phenotype-omlx (oMLX)

LLM inference server with continuous batching and tiered KV caching, optimized for Apple Silicon and managed from the macOS menu bar.

## Repository identity

- Language: Python 3.10+ (Apple Silicon / MLX)
- License: Apache-2.0
- Entry point: `pyproject.toml` (project name `omlx`); package source under `omlx/`.
- Test config: `pytest.ini`; tests under `tests/`.
- Packaging: macOS `.dmg` via `packaging/`; Homebrew tap via `Formula/`.

## Build & test (verified from pyproject.toml + repo layout)

```bash
# Install in editable mode for development
pip install -e .

# Run tests (pytest config in pytest.ini)
pytest

# CLI entry (post-install)
omlx --help
```

Heavy deps: `mlx`, `mlx-lm` (pinned to upstream commit), `mlx-embeddings`, `transformers <5.4.0`. Apple Silicon required.

## Governance

- README: `README.md` (also `README.zh.md`, `README.ko.md`, `README.ja.md`).
- MCP example config: `mcp.example.json`.
- Docs: `docs/`.

## Commit & branch convention

- Conventional Commits.
- Branch: `<type>/<topic>`.
- Releases tagged via `Formula/` updates and `.dmg` builds in `packaging/`.

## Agent guardrails

- Verify any pinned dependency change against the comments in `pyproject.toml` (several deps are pinned to specific commits or version ranges for known-good behavior — do not bump blindly).
- macOS-only runtime: do NOT add Linux/Windows-specific code paths without an explicit feature flag.
