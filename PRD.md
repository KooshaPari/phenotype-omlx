# PRD — phenotype-shared

## Overview

`phenotype-shared` is a Rust workspace (with companion TypeScript packages) providing domain-agnostic, reusable infrastructure primitives for the Phenotype ecosystem. Each crate is independently consumable with no inter-crate dependencies. The workspace is the canonical location for shared infrastructure so that every Phenotype service can depend on the same battle-tested primitives rather than reinventing them.

---

## Epics and User Stories

### E1: Core Domain Model (`phenotype-domain`)

**E1.1: Bounded-Context Value Objects**
As a service author, I need strongly-typed, self-validating value objects (AgentId, TaskId, WorkflowId, PolicyId, AgentStatus, TaskStatus, Priority, Timestamp, AgentName, TaskName, WorkflowName) so that I can model domain concepts without primitive obsession.

Acceptance criteria:
- All value objects are immutable structs.
- Construction validates invariants and returns a `DomainError` or `ValidationError` on failure.
- All implement `Debug`, `Clone`, `Serialize`, `Deserialize`.
- No external infrastructure dependency.

**E1.2: DDD Building Blocks**
As a service author, I need foundational DDD types — entities, aggregates, domain events, and domain services — so that I can apply consistent DDD patterns across services.

Acceptance criteria:
- `entities.rs` exposes entity base traits with stable identity.
- `aggregates.rs` exposes aggregate root traits enforcing consistency boundaries.
- `events.rs` exposes immutable domain event types.
- `services.rs` exposes stateless cross-aggregate service traits.
- Zero infrastructure dependencies in the domain layer.

---

### E2: Event Sourcing Engine (`phenotype-event-sourcing`)

**E2.1: Append-Only Event Store**
As a service author, I need an append-only event store with monotonically increasing sequence numbers so that I can build audit trails and event-sourced aggregates.

Acceptance criteria:
- `EventEnvelope` carries typed payload, sequence number, aggregate type, aggregate ID, and timestamp.
- Sequence numbers are monotonically increasing per aggregate stream.
- Append operation is atomic.

**E2.2: SHA-256 Hash Chain Integrity**
As an operator, I need each appended event to include a SHA-256 hash linking to the previous event so that I can detect tampering or corruption.

Acceptance criteria:
- Each `EventEnvelope` contains a `hash` field: SHA-256 of the serialized previous event (or genesis hash for first).
- A hash verification function traverses all events in a stream and returns `Ok(())` if the chain is intact.
- Verification failure yields a typed error with the offending sequence number.

**E2.3: Snapshot Management**
As a service author, I need snapshot creation and loading so that I can reconstruct aggregates without replaying the full event history.

Acceptance criteria:
- `SnapshotRepository` port defines `save_snapshot` and `load_snapshot`.
- In-memory snapshot store provided as default adapter.
- Snapshot stores aggregate version alongside state.

**E2.4: Pluggable Storage Backends**
As an operator, I need to swap storage backends (in-memory, SQLx, SQLite) without changing application code so that I can choose the right persistence tier.

Acceptance criteria:
- `EventRepository` and `SnapshotRepository` are Rust traits (ports).
- `InMemoryEventStore` ships in the crate.
- SQLx-based store is available as an optional feature.

---

### E3: Two-Tier Cache Adapter (`phenotype-cache-adapter`)

**E3.1: L1 LRU + L2 DashMap Cache**
As a service author, I need a two-tier cache where L1 is a bounded LRU and L2 is an unbounded concurrent map so that hot keys benefit from fast local lookup while cold keys still have cache coverage.

Acceptance criteria:
- Read: check L1 first; on miss, check L2; on L2 hit, promote to L1.
- Both tiers are thread-safe.
- L1 capacity is configurable via `CacheConfig`.

**E3.2: TTL Expiration**
As a service author, I need entries to expire after a configurable TTL so that stale data is evicted automatically.

Acceptance criteria:
- `CacheEntry` stores an `Instant` deadline.
- Reads that encounter an expired entry treat it as a miss and evict it.
- TTL is validated at `CacheConfig` construction (must be > 0).

**E3.3: Metrics Hooks**
As an operator, I need cache hit/miss/eviction counters so that I can observe cache effectiveness.

Acceptance criteria:
- `MetricsCollector` trait with `record_hit`, `record_miss`, `record_eviction`.
- `NoopMetricsCollector` (zero-overhead default) and `AtomicMetricsCollector` (lock-free counters) provided.
- Metrics are accessible via `CacheMetricsDto`.

**E3.4: Hexagonal Architecture Compliance**
As a crate consumer, I need the cache implementation to follow hexagonal architecture so that I can swap the in-memory adapter for an external cache without touching application logic.

Acceptance criteria:
- `CacheService` port in domain layer.
- `InMemoryEntryStore` in adapter layer.
- Application use cases (`GetFromCache`, `InsertIntoCache`, `RemoveFromCache`, `GetCacheMetrics`) depend only on port traits.

---

### E4: Policy Engine (`phenotype-policy-engine`)

**E4.1: Rule-Based Policy Evaluation**
As a governance author, I need to define policies as named sets of Allow/Deny/Require rules so that I can codify and enforce access and compliance constraints programmatically.

Acceptance criteria:
- `Rule` has three types: `Allow`, `Deny`, `Require`.
- Each rule targets a named fact key and a regex pattern.
- Allow: passes if fact absent or matches pattern.
- Deny: fails if fact matches pattern.
- Require: fails if fact absent or does not match pattern.

**E4.2: Policy Evaluation Against Context**
As a service author, I need to evaluate an `EvaluationContext` (a key-value map of facts) against a set of policies and receive a `PolicyResult` with all violations so that I can make governance decisions.

Acceptance criteria:
- `PolicyEngine` holds a `DashMap<String, Policy>` for thread-safe concurrent access.
- `evaluate_all` returns a `Vec<PolicyResult>` with one result per policy.
- `PolicyResult` lists `Violation` structs with fact key, pattern, severity, and rule type.
- `Severity` levels: `Low`, `Medium`, `High`, `Critical`.

**E4.3: TOML Configuration Loading**
As a governance author, I need to load policies from TOML files so that policies can be version-controlled without recompilation.

Acceptance criteria:
- `PolicyLoader` reads a TOML file and returns `Vec<Policy>`.
- Invalid TOML or schema violations yield a typed `PolicyEngineError`.
- `PolicyEngine::with_policies` accepts the loaded `Vec<Policy>` for registration.

---

### E5: Finite State Machine (`phenotype-state-machine`)

**E5.1: Generic FSM with Typed States**
As a service author, I need a generic finite state machine parameterized on any user-defined `State` type so that I can model entity lifecycles with compile-time safety.

Acceptance criteria:
- `State` trait requires `ordinal() -> u32` and `all_states() -> Vec<Self>`.
- FSM is generic: `StateMachine<S: State>`.
- Transition table is built from a list of `(from, to, guard)` tuples.

**E5.2: Transition Guards**
As a service author, I need optional guard callbacks on transitions so that I can add runtime business rule validation before a state change is applied.

Acceptance criteria:
- Guard is a callable `Fn(&Context) -> bool`.
- If a guard returns false, the transition is rejected with a typed error.
- Guard errors are distinct from invalid-transition errors.

**E5.3: Forward-Only Enforcement**
As a service author, I need the FSM to optionally enforce forward-only progression so that I can prevent illegal state regressions.

Acceptance criteria:
- When forward-only mode is enabled, any transition to a state with `ordinal() < current.ordinal()` is rejected.
- Forward-only mode is configurable at FSM construction.

**E5.4: Transition History**
As an auditor, I need the full transition history of an FSM instance so that I can audit how an entity moved through states.

Acceptance criteria:
- Each accepted transition is appended to an internal history log.
- `history()` returns an ordered slice of `(from, to, timestamp)` tuples.

---

### E6: Port Interface Library (`phenotype-port-interfaces`)

**E6.1: Inbound Ports (CQRS)**
As a service author, I need standard `Command`, `CommandHandler`, `CommandBus`, `Query`, `QueryHandler`, and `QueryBus` traits so that I can implement CQRS without redefining these abstractions per-service.

Acceptance criteria:
- `Command` trait: associated `Result` type; implements `Serialize`.
- `CommandHandler<C: Command>`: async `handle(&self, command: C) -> Result<C::Result>`.
- `Query` trait: associated `Output` type.
- `QueryHandler<Q: Query>`: async `execute(&self, query: Q) -> Result<Q::Output>`.

**E6.2: Outbound Repository Port**
As a service author, I need a generic `Repository<Entity, Id>` trait with async CRUD operations so that I can write application logic against a persistence abstraction.

Acceptance criteria:
- `find_by_id`, `find_all(page, page_size)`, `save`, `delete`, `exists` methods.
- `RepositoryExt` blanket impl adds `find_or_create` and `count`.
- All methods are `async` and return `Result<_, PortError>`.

**E6.3: Outbound Infrastructure Ports**
As a service author, I need standard ports for cache, logging, filesystem, configuration, queue, HTTP, and event bus so that I can swap infrastructure adapters without changing application code.

Acceptance criteria:
- `CachePort`, `LoggerPort`, `FilesystemPort`, `ConfigPort`, `QueuePort`, `HttpPort`, `EventBusPort` traits defined.
- All are async traits returning `Result<_, PortError>`.
- All defined in `outbound/` module; no concrete implementations in this crate.

---

### E7: Application Layer (`phenotype-application`)

**E7.1: CQRS Commands and Queries**
As a service author, I need ready-made command and query structs for Agent and Task domain operations.

Acceptance criteria:
- Commands: `CreateAgent`, `UpdateAgent`, `DeleteAgent`, `CreateTask`, `AssignTask`.
- Queries: `GetAgentById`, `ListAgents`, `SearchAgents`, `GetTaskMetrics`, `ListTasksByAgent`.
- Each has corresponding DTO types in the `dto` module.

**E7.2: Use Case Handlers**
As a service author, I need handler implementations that validate input, coordinate domain objects, call ports, and return DTOs.

Acceptance criteria:
- Each handler depends only on port traits (no concrete infrastructure).
- Validation errors produce typed `ApplicationError` variants.
- Handlers are independently testable with mock ports.

---

### E8: TypeScript Shared Packages

**E8.1: Typed Error Library (`@helios/errors`)**
As a TypeScript service author, I need canonical `ErrorCode` enums and `HeliosAppError` class so that error handling is consistent across all TypeScript services.

Acceptance criteria:
- `ErrorCode` covers generic, protocol, and domain-specific error codes.
- `HeliosAppError extends Error` with `code`, `details`, and optional `fatal` flag.
- Zero runtime dependencies.

**E8.2: Workspace and Session Types (`@helios/types`)**
As a TypeScript service author, I need shared type definitions for Workspace, Project, and Session so that the same types are used across services.

Acceptance criteria:
- `WorkspaceState`, `Workspace`, `ProjectBinding`, `WorkspaceBinding`, `Session` defined as readonly interfaces.
- Pure type definitions — no runtime code.

**E8.3: Prefixed ID Generation (`@helios/ids`)**
As a TypeScript service author, I need ULID-based prefixed ID generation for all entity types so that IDs are sortable, typesafe, and self-describing.

Acceptance criteria:
- `generateId(entityType: EntityType): string` produces `{prefix}_{ulid}`.
- ID format validated against `/^[a-z]{2,3}_[0-9A-HJKMNP-TV-Z]{26}$/`.
- `validateId`, `parseId`, `generateCorrelationId` exported.
- `PREFIX_MAP` and `REVERSE_PREFIX_MAP` exported for introspection.

---

### E9: Workspace Ergonomics and Crate Versioning

**E9.1: Independent Crate Publication**
As a platform engineer, I need each crate to be independently publishable to crates.io so that
consuming services can take only the crates they need without transitive dependencies on unused
infrastructure.

Acceptance criteria:
- Each crate has its own `[package]` block with `version`, `description`, `license`.
- No crate has a path dependency on another crate in this workspace.
- `cargo publish -p <crate>` succeeds for each crate independently.

**E9.2: Workspace Dependency Version Pinning**
As a maintainer, I need all shared dependency versions pinned at the workspace level so that
all crates agree on transitive dependency versions and avoid resolver conflicts.

Acceptance criteria:
- All shared dependencies (serde, tokio, thiserror, dashmap, etc.) are declared in
  `[workspace.dependencies]` with pinned major versions.
- Individual crates reference workspace deps using `workspace = true`.
- `cargo update` does not silently upgrade a dep for one crate while leaving another behind.

**E9.3: Backward Compatibility Policy**
As a consuming service, I need a documented semver backward compatibility policy so that I
can plan upgrades confidently.

Acceptance criteria:
- Semver is followed: patch = no API change, minor = additive only, major = breaking.
- Breaking changes in any crate must update that crate's major version.
- A `CHANGELOG.md` entry is required for every published version.
- Public API is defined as: all items reachable via `pub use` re-exports in `lib.rs`.
