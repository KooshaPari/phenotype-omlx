# Specifications

Acceptance criteria:

- Add one Sladge badge to the README badge block.
- Do not update localized READMEs in this docs-only governance pass.
- Keep all runtime and dependency files unchanged.
- Validate with diff hygiene, badge search, and available project checks.

ARUs:

- Assumption: README badge metadata is sufficient for the governance rollout.
- Risk: full runtime tests may require Apple Silicon MLX dependencies or local
  model assets. Mitigation: run lightweight repo-native checks first and record
  any environment blocker exactly.
