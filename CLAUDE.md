# CLAUDE.md — phenotype-omlx

Fork of [jundot/omlx](https://github.com/jundot/omlx) for Phenotype-org Homebrew tap and custom integrations.

## Repository Identity

- **Upstream**: `https://github.com/jundot/omlx` (Apache-2.0, jundot maintainer)
- **Phenotype fork**: `https://github.com/KooshaPari/phenotype-omlx`
- **Language**: Python 3.10+, Apple Silicon (MLX)
- **Package**: `omlx` (pip editable + Homebrew)
- **License**: Apache-2.0

## Stack

| Layer | Technology |
|-------|------------|
| Inference engine | `mlx`, `mlx-lm` (pinned to upstream commit) |
| Server | FastAPI (OpenAI/Anthropic-compatible) |
| CLI | `omlx` command (typer) |
| Cache | Paged KV cache (hot RAM + cold SSD tier) |
| macOS app | PyObjC menubar app + venvstacks |
| Packaging | `.dmg` via `packaging/`, Homebrew via `Formula/` |
| Docs | `docs/`, `README.zh.md`, `README.ko.md`, `README.ja.md` |

## Dev Commands

```bash
# Install (editable, core only)
pip install -e .

# Install with MCP support
pip install -e ".[mcp]"

# Install with dev deps
pip install -e ".[dev]"

# Run tests
pytest

# CLI
omlx --help
omlx serve --model-dir ~/models

# macOS app build (requires venvstacks)
cd packaging && python build.py
```

## Homebrew Formula

The `Formula/omlx.rb` file drives the Phenotype Homebrew tap. Update this file for new releases:

```bash
# Bump version in Formula/omlx.rb
brew extract --version <new-version> --versioned-name omlx-<version> <source-tap> phenotype-omlx
# Or edit Formula/omlx.rb directly with new version + sha256
```

## Pinned Dependencies

Several deps are pinned to specific commits or version ranges for known-good behavior. **Do not bump blindly.** Always verify against comments in `pyproject.toml`.

Key pins:
- `mlx`, `mlx-lm` — pinned to upstream commit (MLX API stability)
- `mlx-embeddings` — version range constraint
- `transformers <5.4.0` — API compatibility

## macOS-Only Runtime

This codebase is Apple Silicon exclusive. Do NOT add Linux/Windows-specific code paths without an explicit feature flag.

## Quality

- Conventional Commits
- Branch: `<type>/<topic>`
- Zero new lint suppressions without inline justification

## Governance

- Reference: `/Users/kooshapari/CodeProjects/Phenotype/repos/AgilePlus`
- Specs: `AgilePlus/kitty-specs/<feature-id>/`
- Worklog: `AgilePlus/.work-audit/worklog.md`
