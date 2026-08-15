# Architecture Decision Records — phenotype-shared

---

## ADR-001: Cargo Workspace with Zero Inter-Crate Dependencies

**Status**: Accepted
**Date**: 2026-03-26
**Traces to**: FR-DOM-004, FR-PORT-009

**Context**: Shared infrastructure must be consumable per-crate. A monolithic shared library forces consumers to take all transitive deps even when they need only one primitive (e.g., just the policy engine).

**Decision**: Use a Cargo workspace where workspace-level `[workspace.dependencies]` pins shared dependency versions, but each crate's `Cargo.toml` selects only what it needs. No crate in the workspace depends on another crate in the workspace.

**Alternatives considered**:
- Single flat crate with feature flags: rejected because feature flag combinatorics are hard to test and document.
- Separate repos per crate: rejected because coordinating dep version bumps across repos is operationally expensive at this stage.

**Consequences**:
- Clean, auditable dependency graphs per crate.
- Individual crate versioning and publishing is straightforward.
- Workspace-level `cargo test --workspace` validates all crates together.
- A shared type (e.g., a common `DomainError`) cannot be imported from a sibling crate; it must be defined locally or in a shared foundation crate outside this workspace.

---

## ADR-002: Hexagonal Architecture (Ports and Adapters) in Every Crate

**Status**: Accepted
**Date**: 2026-03-26
**Traces to**: FR-CACHE-008, FR-EVT-007, FR-PORT-001 to FR-PORT-009

**Context**: Infrastructure primitives (cache, event store, state machine) must be swappable without touching application logic. Services need to mock these for testing.

**Decision**: Every substantial crate follows hexagonal architecture with three layers:
- `domain/` — entities, value objects, port traits, domain services. No external deps beyond `serde`, `thiserror`, `uuid`, `chrono`.
- `application/` — use cases, commands/queries, DTOs. Depends only on domain layer ports.
- `adapters/` — concrete implementations (in-memory, HTTP, Postgres, Redis). Depends on domain ports; never imported by domain or application.

**Alternatives considered**:
- Layered architecture (Controller → Service → Repository): rejected because it still couples service layer to infrastructure via concrete types.
- No architectural constraint: rejected because it led to test-hostile, tightly coupled code in earlier Phenotype services.

**Consequences**:
- Application code is testable with pure in-memory adapters; no Docker required for unit tests.
- Adding a new adapter (e.g., SQLite event store) requires implementing a port trait only — application code is untouched.
- More boilerplate per crate (port traits + use case structs), but that cost is paid once and amortized across all consumers.

---

## ADR-003: SHA-256 Hash Chains for Event Integrity

**Status**: Accepted
**Date**: 2026-03-26
**Traces to**: FR-EVT-003, FR-EVT-004

**Context**: Event sourcing requires tamper detection. Soft audit trails (sequence numbers only) can be silently corrupted by database updates.

**Decision**: Each `EventEnvelope` contains a `hash` field: `SHA-256(serialize(previous_envelope))`. The first event in a stream uses a defined genesis constant (`"genesis"`). A verification pass traverses the full stream and recomputes hashes.

**Alternatives considered**:
- No integrity check: rejected because the crate targets governance and audit use cases.
- HMAC with a shared secret: rejected because it requires key management infrastructure; SHA-256 chaining provides tamper-evidence without secrets.
- Merkle tree: rejected as over-engineered for per-stream linear event logs.

**Consequences**:
- Appending an event is slightly more expensive (one SHA-256 hash per append).
- Verification is O(n) in event count; callers should checkpoint verification frequency.
- Any out-of-order insert or update to a past event breaks the chain and is detectable.

---

## ADR-004: DashMap for Thread-Safe Policy Registry

**Status**: Accepted
**Date**: 2026-03-26
**Traces to**: FR-POL-005

**Context**: `PolicyEngine` must support concurrent reads (policy evaluation) and occasional writes (add/remove policy) in a multi-threaded Tokio runtime without a global `RwLock`.

**Decision**: Use `DashMap<String, Policy>` from the `dashmap` crate. `DashMap` uses sharded locking internally, giving fine-grained read parallelism and non-blocking concurrent writes to different shards.

**Alternatives considered**:
- `Arc<RwLock<HashMap>>`: simple but creates a global write bottleneck; any policy add blocks all readers.
- `Arc<Mutex<HashMap>>`: worse than RwLock for read-heavy workloads.
- Immutable snapshot on write (copy-on-write): complex and wastes memory for large policy sets.

**Consequences**:
- Near-zero contention for read-dominant workloads.
- `DashMap` is a workspace dependency shared by `phenotype-cache-adapter` and `phenotype-policy-engine`.
- Iteration over all policies acquires short-lived shard locks sequentially; callers must not hold shard refs across await points.

---

## ADR-005: Regex Pattern Matching for Policy Rules

**Status**: Accepted
**Date**: 2026-03-26
**Traces to**: FR-POL-001, FR-POL-009

**Context**: Policy rules need flexible matching on fact values. Simple equality is insufficient for patterns like IP range prefixes, email domains, or role name prefixes.

**Decision**: Each `Rule` stores a `pattern: String` compiled to `Regex` at evaluation time. Compilation errors surface as `PolicyEngineError::RegexCompilationError`.

**Alternatives considered**:
- Glob patterns: simpler but cannot express character classes or anchoring precisely.
- Compiled `Regex` stored in `Rule`: preferred long-term but requires `Regex` to be `Send + Sync + Clone` or wrapped in `Arc`; deferred to avoid premature optimization.
- CEL (Common Expression Language): powerful but adds a large parser dependency.

**Consequences**:
- Regex compilation cost on every rule evaluation; acceptable for policy evaluation frequencies (not hot path).
- Invalid regex is a configuration error surfaced at runtime, not at TOML load time; callers should validate patterns on load if desired.
- `regex` crate is a workspace dependency.

---

## ADR-006: Forward-Only Ordinal Enforcement in State Machine

**Status**: Accepted
**Date**: 2026-03-26
**Traces to**: FR-SM-005

**Context**: Many entity lifecycles (task progression, order fulfillment, agent startup) are strictly forward-only. Allowing backward transitions requires consumers to add their own guards everywhere.

**Decision**: `State` trait requires `ordinal() -> u32`. `StateMachine` has an optional `forward_only` flag. When enabled, transitions where `to.ordinal() < current.ordinal()` are rejected with `StateMachineError::BackwardTransitionForbidden` before the guard is evaluated.

**Alternatives considered**:
- Encode forward-only as a property of each transition: more granular but verbose; callers repeat `from.ordinal() < to.ordinal()` in every guard.
- Derive ordinal from enum discriminant automatically: requires proc-macro; out of scope for a foundation crate.

**Consequences**:
- Consumer must assign stable ordinals to states; reordering enum variants without updating ordinals is a logic bug.
- Forward-only check is O(1) per transition.
- Consumers that need bidirectional FSMs simply leave `forward_only = false`.

---

## ADR-007: ULID-Based Prefixed IDs for TypeScript Packages

**Status**: Accepted
**Date**: 2026-03-26
**Traces to**: FR-TS-004 to FR-TS-008

**Context**: TypeScript services need globally unique, sortable entity IDs that are self-describing (the ID itself reveals entity type) and safe for URL usage.

**Decision**: IDs are `{prefix}_{ulid}` strings where:
- `prefix` is 2–3 lowercase ASCII letters from `PREFIX_MAP` keyed by `EntityType`.
- `ulid` is a 26-character Crockford Base32 ULID (sortable, monotonic, URL-safe).
- Format validated against `/^[a-z]{2,3}_[0-9A-HJKMNP-TV-Z]{26}$/`.

**Alternatives considered**:
- UUID v4: not sortable, not self-describing.
- Snowflake IDs: require a node ID coordination service.
- NanoID: no time component, not sortable.

**Consequences**:
- IDs are database-sortable by creation time within an entity type.
- Parsing an ID recovers `EntityType` without a DB lookup.
- Adding a new entity type requires updating `PREFIX_MAP` and `REVERSE_PREFIX_MAP`; must choose a unique prefix.
- Generated IDs are validated synchronously in `generateId`; a format mismatch throws immediately (debug assertion — negligible on production paths).
