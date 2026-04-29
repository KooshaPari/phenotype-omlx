# Implementation Strategy

## Approach

Use an isolated docs-only rollout:

- Add the sladge badge to the existing README badge block.
- Store rollout evidence under `docs/sessions/`.
- Avoid runtime, generated, and catalog changes.

## Rationale

The badge marks a direct LLM-serving surface without changing package behavior.
