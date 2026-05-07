# Implementation Strategy

- Use a new isolated worktree from current local `main`.
- Limit the code diff to README badge metadata plus session evidence.
- Prefer lightweight validation because oMLX runtime checks can require
  macOS-specific MLX dependencies and model assets.
