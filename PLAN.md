# Plan - phenotype-shared

## Phase 1: Core Crates (Complete — 2026-03-26)

| Task | Description | Status |
|------|-------------|--------|
| P1.1 | Event sourcing crate | ✅ Done |
| P1.2 | Cache adapter crate | ✅ Done |
| P1.3 | Policy engine crate | ✅ Done |
| P1.4 | State machine crate | ✅ Done |
| P1.5 | CI workflow | ✅ Done |
| P1.6 | **phenotype-domain** crate (DDD VOs, entities, aggregates) | ✅ Done (2026-03-26) |
| P1.7 | **phenotype-application** crate (CQRS commands, queries, handlers) | ✅ Done (2026-03-26) |

**Workspace crates now building clean** (`cargo check -p phenotype-domain -p phenotype-application -p phenotype-port-interfaces`).

## Phase 2: Governance and Long-Term Cleanup (Done — 2026-03-26)

| Task | Description | Status |
|------|-------------|--------|
| P2.1 | Governance baseline PR scope | ✅ Done |
| P2.2 | Wrap-over-hand-roll rules and extraction guidance | ✅ Done |
| P2.3 | Prioritized shared-extraction backlog | ✅ Done |
| P2.4 | Repo-structure normalization and worktree placement rules | ✅ Done |
| P2.5 | ADR promotion path and canonical status source | ✅ Done |

## Phase 3: Remaining Work (Next)

### P0 — Fix broken pre-existing crates (no deps on shared extraction)
- `phenotype-event-sourcing`: 4 compilation errors (E0210, E0277, E0282, E0432, E0599, E0603)
- `phenotype-cache-adapter`: 38 compilation errors
- Both crates need import fixes and trait impl corrections
- **Recommended**: fix in place or archive and replace with fresh implementations

### P0 — Open PR for core crate fixes
- Commit the 3 fixed `#[validate(nested)]` + `RepositoryError` → `PortError` fixes
- PR to `phenotype-shared` main

### P1 — Shared extraction candidates (identified)
1. **phenotype-port-interfaces** already has clean Repository, EventStore, MessageBus, Config, Cache ports
2. **phenotype-domain** VOs (AgentId, TaskId, Priority, etc.) are reusable across all adapters
3. **phenotype-application** command/query types are language-neutral DTOs (Rust, Go, TS, Python all have equivalents)

### P1 — Propagate governance to other phenotype-* repos
- template-commons, phenotype-config, phenodocs, phench as next candidates
- Each gets a lightweight PR: CI stub + ADR stub + CLAUDE.md update

## Phase 2 Backlog

### P0 — Governance baseline
1. Standardize the `phenotype-shared` governance PR as the canonical baseline for CI, linting, build checks, and policy gates.
2. Keep branch-protection-aligned checks lightweight, deterministic, and repo-local where possible.
3. Add ADR-driven decision records for governance, layout, and extraction policies.

### P1 — Shared extraction
1. Extract pagination primitives and response wrappers first.
2. Extract reusable error-mapping helpers next, without collapsing bounded-context-specific errors.
3. Promote only stable, technology-neutral port contracts into `phenotype-port-interfaces`.
4. Keep adapters and workflow-specific glue local until repetition proves reuse.

### P1 — Repo structure
1. Normalize top-level repo buckets to make intent obvious: `apps/`, `libs/`, `infrastructure/`, `governance/`, `tooling/`, `templates/`.
2. Keep active worktrees under `repos/worktrees/<project>/<category>/<wtree>` and out of canonical repo roots.
3. Prefer shallow, discoverable directories over deeply nested ad hoc layouts.

### P2 — Canonical status and promotion path
1. Keep one active governance status source in `plans/` and treat it as the canonical backlog.
2. Promote ADRs when decisions change repo layout, extraction boundaries, or shared contracts.
3. Revisit the long-term cleanup backlog after each governance PR lands.
