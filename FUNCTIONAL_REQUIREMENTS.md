# Functional Requirements — phenotype-shared

Traces to: PRD.md epics E1–E8.
ID format: FR-{CAT}-{NNN} where CAT = DOM | EVT | CACHE | POL | SM | PORT | APP | TS.

---

## Domain Model (phenotype-domain)

**FR-DOM-001**: The system SHALL provide strongly-typed value objects for AgentId, TaskId, WorkflowId, PolicyId, AgentStatus, TaskStatus, Priority, Timestamp, AgentName, TaskName, and WorkflowName.
Traces to: E1.1

**FR-DOM-002**: Value object constructors SHALL validate invariants and return a typed `ValidationError` when input is invalid.
Traces to: E1.1

**FR-DOM-003**: All domain value objects SHALL implement `Debug`, `Clone`, `Serialize`, and `Deserialize`.
Traces to: E1.1

**FR-DOM-004**: The domain crate SHALL have zero dependencies on infrastructure or adapter crates.
Traces to: E1.1, E1.2

**FR-DOM-005**: The system SHALL provide entity base traits with stable identity semantics, aggregate root traits enforcing consistency boundaries, immutable domain event types, and stateless domain service traits.
Traces to: E1.2

---

## Event Sourcing (phenotype-event-sourcing)

**FR-EVT-001**: The system SHALL append events to a named aggregate stream with monotonically increasing sequence numbers.
Traces to: E2.1

**FR-EVT-002**: `EventEnvelope` SHALL carry: typed serializable payload, u64 sequence number, aggregate type string, aggregate ID string, UTC timestamp, and actor string.
Traces to: E2.1

**FR-EVT-003**: Each appended `EventEnvelope` SHALL include a SHA-256 hash computed over the serialized previous event envelope; the first event in a stream SHALL use a defined genesis hash.
Traces to: E2.2

**FR-EVT-004**: The system SHALL expose a hash-chain verification function that traverses all events in a stream and returns `Ok(())` if the chain is intact or a typed error identifying the first broken link.
Traces to: E2.2

**FR-EVT-005**: The system SHALL support snapshot creation and loading via `SnapshotRepository` port with `save_snapshot(aggregate_id, version, state)` and `load_snapshot(aggregate_id)` operations.
Traces to: E2.3

**FR-EVT-006**: `InMemoryEventStore` SHALL be provided as the default `EventRepository` adapter with no external dependencies.
Traces to: E2.4

**FR-EVT-007**: `EventRepository` and `SnapshotRepository` SHALL be Rust trait definitions in the domain ports module, allowing adapters to be swapped without changing application code.
Traces to: E2.4

---

## Cache Adapter (phenotype-cache-adapter)

**FR-CACHE-001**: Cache reads SHALL consult L1 (LRU) first; on an L1 miss they SHALL consult L2 (DashMap); on an L2 hit the entry SHALL be promoted to L1 before returning.
Traces to: E3.1

**FR-CACHE-002**: L1 capacity (max entries) SHALL be configurable at `CacheConfig` construction and validated to be >= 1.
Traces to: E3.1

**FR-CACHE-003**: `CacheEntry` SHALL store an expiry deadline (`Instant`); any read of an expired entry SHALL treat it as a miss and evict it from both tiers.
Traces to: E3.2

**FR-CACHE-004**: TTL SHALL be validated at `CacheConfig` construction; a zero or negative TTL SHALL be rejected with a typed error.
Traces to: E3.2

**FR-CACHE-005**: The system SHALL expose a `MetricsCollector` trait with `record_hit`, `record_miss`, and `record_eviction` methods.
Traces to: E3.3

**FR-CACHE-006**: `NoopMetricsCollector` (no-op, zero overhead) and `AtomicMetricsCollector` (lock-free atomic counters) SHALL be provided as built-in implementations.
Traces to: E3.3

**FR-CACHE-007**: Cache metrics SHALL be accessible as a `CacheMetricsDto` with hit_count, miss_count, eviction_count, and hit_rate fields.
Traces to: E3.3

**FR-CACHE-008**: Application use cases (`GetFromCache`, `InsertIntoCache`, `RemoveFromCache`, `GetCacheMetrics`) SHALL depend only on `CacheService` port, not on any concrete store type.
Traces to: E3.4

---

## Policy Engine (phenotype-policy-engine)

**FR-POL-001**: The system SHALL define three `RuleType` variants: `Allow`, `Deny`, and `Require`, each with distinct evaluation semantics.
Traces to: E4.1

**FR-POL-002**: `Allow` rules SHALL pass when the target fact is absent or matches the regex pattern; `Deny` rules SHALL fail when the fact matches the pattern; `Require` rules SHALL fail when the fact is absent or does not match the pattern.
Traces to: E4.1

**FR-POL-003**: `Rule` SHALL carry: `rule_type`, `fact` (key), `pattern` (regex string), and optional `description`.
Traces to: E4.1

**FR-POL-004**: `EvaluationContext` SHALL be a key-value map (String -> Value) supporting typed accessors `get_string`, `get_bool`, `get_int`.
Traces to: E4.2

**FR-POL-005**: `PolicyEngine` SHALL store policies in a `DashMap` to allow concurrent reads and writes without a global lock.
Traces to: E4.2

**FR-POL-006**: `PolicyEngine::evaluate_all(context)` SHALL return one `PolicyResult` per registered policy; each result SHALL list all `Violation` structs produced by that policy's rules.
Traces to: E4.2

**FR-POL-007**: `Violation` SHALL carry: fact key, pattern string, `RuleType`, and `Severity` (Low | Medium | High | Critical).
Traces to: E4.2

**FR-POL-008**: `PolicyLoader` SHALL parse a TOML file into `Vec<Policy>` and return a typed `PolicyEngineError` for malformed input.
Traces to: E4.3

**FR-POL-009**: Invalid regex patterns in rules SHALL produce a `PolicyEngineError::RegexCompilationError` at evaluation time, not at load time.
Traces to: E4.1

---

## State Machine (phenotype-state-machine)

**FR-SM-001**: The `State` trait SHALL require `ordinal() -> u32` (for ordering) and `all_states() -> Vec<Self>` (for validation).
Traces to: E5.1

**FR-SM-002**: `StateMachine<S: State>` SHALL maintain a current state and a transition table mapping `(from, to)` pairs to optional guard functions.
Traces to: E5.1

**FR-SM-003**: Attempting a transition not registered in the transition table SHALL return a typed `StateMachineError::InvalidTransition`.
Traces to: E5.1

**FR-SM-004**: When a transition guard returns false, the transition SHALL be rejected with `StateMachineError::GuardRejected`.
Traces to: E5.2

**FR-SM-005**: When forward-only mode is enabled, any transition where `to.ordinal() < current.ordinal()` SHALL be rejected with `StateMachineError::BackwardTransitionForbidden`.
Traces to: E5.3

**FR-SM-006**: Every accepted transition SHALL be appended to an internal ordered history log recording `(from_state, to_state, timestamp)`.
Traces to: E5.4

**FR-SM-007**: `history()` SHALL return an immutable slice of all recorded transitions in insertion order.
Traces to: E5.4

---

## Port Interfaces (phenotype-port-interfaces)

**FR-PORT-001**: The `Command` trait SHALL declare an associated `Result` type and require `Serialize + Send + Sync + 'static`.
Traces to: E6.1

**FR-PORT-002**: `CommandHandler<C: Command>` SHALL declare `async fn handle(&self, command: C) -> Result<C::Result>`.
Traces to: E6.1

**FR-PORT-003**: The `Query` trait SHALL declare an associated `Output` type.
Traces to: E6.1

**FR-PORT-004**: `QueryHandler<Q: Query>` SHALL declare `async fn execute(&self, query: Q) -> Result<Q::Output>`.
Traces to: E6.1

**FR-PORT-005**: `Repository<Entity, Id>` SHALL declare async `find_by_id`, `find_all(page, page_size)`, `save`, `delete`, and `exists` methods returning `Result<_, PortError>`.
Traces to: E6.2

**FR-PORT-006**: A `RepositoryExt` blanket implementation SHALL add `find_or_create` and `count` to all `Repository` implementors.
Traces to: E6.2

**FR-PORT-007**: Outbound port traits SHALL be defined for: `CachePort`, `LoggerPort`, `FilesystemPort`, `ConfigPort`, `QueuePort`, `HttpPort`, and `EventBusPort`.
Traces to: E6.3

**FR-PORT-008**: All outbound port methods SHALL be async and return `Result<_, PortError>`.
Traces to: E6.3

**FR-PORT-009**: No concrete adapter implementations SHALL exist in this crate; it is a trait-only library.
Traces to: E6.3

---

## Application Layer (phenotype-application)

**FR-APP-001**: The application crate SHALL define command structs: `CreateAgent`, `UpdateAgent`, `DeleteAgent`, `CreateTask`, `AssignTask`.
Traces to: E7.1

**FR-APP-002**: The application crate SHALL define query structs: `GetAgentById`, `ListAgents`, `SearchAgents`, `GetTaskMetrics`, `ListTasksByAgent`.
Traces to: E7.1

**FR-APP-003**: Each command and query SHALL have a corresponding DTO type in the `dto` module for request input and response output.
Traces to: E7.1

**FR-APP-004**: Handler implementations SHALL depend only on port traits from `phenotype-port-interfaces`; no concrete infrastructure types SHALL be imported.
Traces to: E7.2

**FR-APP-005**: Handlers SHALL produce typed `ApplicationError` variants (e.g., `ValidationFailed`, `NotFound`, `Conflict`) rather than opaque error strings.
Traces to: E7.2

---

## TypeScript Packages

**FR-TS-001**: `@helios/errors` SHALL export `ErrorCode` enum covering at minimum: `INTERNAL_ERROR`, `INVALID_ARGUMENT`, `NOT_FOUND`, `ALREADY_EXISTS`, `PERMISSION_DENIED`, `UNAUTHENTICATED`, `RESOURCE_EXHAUSTED`, `CANCELLED`, `UNAVAILABLE`, `NOT_IMPLEMENTED`, `TIMEOUT`, `VALIDATION_ERROR`, `METHOD_NOT_SUPPORTED`, `MISSING_CORRELATION_ID`, `TERMINAL_NOT_FOUND`, `LANE_NOT_FOUND`, `SESSION_NOT_FOUND`, `SESSION_NOT_ATTACHED`, `TERMINAL_BINDING_INVALID`.
Traces to: E8.1

**FR-TS-002**: `HeliosAppError` SHALL extend `Error` and expose `readonly code: ErrorCode`, `readonly details?: Record<string, unknown>`, and `readonly fatal?: boolean`.
Traces to: E8.1

**FR-TS-003**: `@helios/types` SHALL export readonly interfaces: `Workspace`, `WorkspaceBinding`, `ProjectBinding`, `Session` with the state union types `WorkspaceState` and session state.
Traces to: E8.2

**FR-TS-004**: `@helios/ids` SHALL export `generateId(entityType: EntityType): string` that produces a string matching `/^[a-z]{2,3}_[0-9A-HJKMNP-TV-Z]{26}$/`.
Traces to: E8.3

**FR-TS-005**: `generateId` SHALL throw synchronously if the generated ID fails format validation (debug assertion, not performance-critical).
Traces to: E8.3

**FR-TS-006**: `validateId(id: string): ValidationResult` SHALL verify format and known prefix.
Traces to: E8.3

**FR-TS-007**: `parseId(id: string): ParsedId` SHALL extract the `EntityType` and ULID portion from a valid ID string.
Traces to: E8.3

**FR-TS-008**: `generateCorrelationId()` SHALL be a convenience wrapper that calls `generateId("correlation")`.
Traces to: E8.3

---

## Workspace Ergonomics and Versioning (phenotype-shared workspace)

**FR-WS-001**: Each crate in the workspace SHALL have its own `[package]` block with `version`, `description`, and `license` fields populated so that the crate is independently publishable to crates.io.
Traces to: E9.1

**FR-WS-002**: No crate in the workspace SHALL depend on another crate in the same workspace via a path dependency. All inter-service sharing is done by the consuming service depending on individual published crates.
Traces to: E9.1

**FR-WS-003**: All shared dependencies (serde, tokio, thiserror, dashmap, lru, parking_lot, moka, uuid, chrono, regex, sha2, tracing) SHALL be declared in `[workspace.dependencies]` with pinned major versions. Individual crates SHALL reference them using `workspace = true`.
Traces to: E9.2

**FR-WS-004**: The workspace root `Cargo.toml` SHALL carry a `[workspace.package]` section defining the shared edition, license, and repository URL inherited by all member crates.
Traces to: E9.2

**FR-WS-005**: A `CHANGELOG.md` entry SHALL be authored for every published version of any crate, documenting breaking changes, new features, and bug fixes.
Traces to: E9.3

**FR-WS-006**: Breaking API changes in any crate SHALL result in a major version increment for that crate. Additive changes SHALL result in a minor increment. Bug-fix-only changes SHALL result in a patch increment.
Traces to: E9.3
