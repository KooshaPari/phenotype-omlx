# User Journeys — phenotype-shared

**Version:** 1.0.0
**Traces to:** PRD.md epics E1–E8

---

## UJ-1: Service Author Adds a New Phenotype Service Using the Domain Model

```
Service author adds phenotype-domain to Cargo.toml
         |
         v
Defines bounded-context value objects (AgentId, TaskId, Priority)
using provided types from phenotype_domain::value_objects
         |
         v
Author implements domain entities (Agent, Task) implementing
entity base traits from phenotype_domain::entities
         |
         v
Author writes aggregates that enforce consistency boundaries
using AgentAggregate or TaskAggregate patterns from phenotype_domain::aggregates
         |
         v
Domain compiles with zero infrastructure deps
         |
         v
Author plugs in adapters (phenotype-postgres-adapter, phenotype-redis-adapter)
via the ports defined in phenotype-port-interfaces
         |
         v
Service tests run with in-memory adapters; no Docker required
```

**Actors:** Service author, Rust compiler
**Goal:** Stand up a new bounded-context service conforming to DDD patterns without
re-implementing shared domain primitives.
**Success criteria:**
- `cargo check` passes with zero warnings in the new service crate.
- All domain types (AgentId, TaskId, etc.) originate from `phenotype-domain`.
- No concrete adapter type is imported in the domain or application layers.

---

## UJ-2: Operator Enables Event Integrity Audit on an Existing Aggregate

```
Operator enables phenotype-event-sourcing in service Cargo.toml
         |
         v
Application layer creates InMemoryEventStore (or swaps in SQLx adapter)
         |
         v
Every aggregate mutation appends an EventEnvelope with typed payload,
sequence number, actor, and SHA-256 hash of previous event
         |
         v
Operator triggers verify_chain(aggregate_id) on demand or on schedule
         |
         v
Chain verification returns Ok(()) — all events are intact
     OR
Chain verification returns Err(HashChainError { sequence, .. }) —
tampered event identified by sequence number
         |
         v
Operator sees audit log with full event history and can replay state
from any snapshot checkpoint
```

**Actors:** Operator, service, event store
**Goal:** Enable tamper-detection audit trails on event-sourced aggregates.
**Success criteria:**
- `verify_chain` traverses all events and returns `Ok(())` when no tampering occurred.
- A single modified event in the chain causes `Err` with the offending sequence number.
- Snapshots reduce replay time for aggregates with >100 events.

---

## UJ-3: Service Author Uses the Two-Tier Cache for an Expensive Computation

```
Author adds phenotype-cache-adapter to Cargo.toml
         |
         v
Author constructs TieredCache::new(capacity, ttl)
or uses CacheConfigBuilder for fine-grained config
         |
         v
On first request: cache.get(&key) returns None (miss)
         |
         v
Author computes expensive result and calls cache.insert(key, value)
         |
         v
On subsequent requests: L1 LRU hit returns value in O(1) without
hitting L2 or the underlying data source
         |
         v
After L1 eviction: L2 DashMap hit promotes entry back to L1
         |
         v
After TTL expiry: entry treated as miss; stale data never served
         |
         v
Author reads CacheMetricsDto to observe l1_hits, l2_hits, misses,
promotions, evictions — confirms expected cache behaviour
```

**Actors:** Service author, cache
**Goal:** Reduce redundant computation by caching results with automatic TTL expiry and
observability hooks.
**Success criteria:**
- L1 hit rate >= 80% under typical hot-key workloads.
- Expired entries are never returned.
- Metrics accurately reflect actual hit/miss/eviction counts.

---

## UJ-4: Governance Author Loads and Evaluates Security Policies from TOML

```
Governance author writes a TOML policy file:
  [[policies]]
  name = "no-public-ips"
  [[policies.rules]]
  rule_type = "Deny"
  fact = "ip_address"
  pattern = "^0\.0\.0\.0$"
  severity = "Critical"
         |
         v
PolicyLoader::from_file("policies.toml") parses the file into Vec<Policy>
         |
         v
PolicyEngine::with_policies(policies) registers all policies
         |
         v
Service builds EvaluationContext from request facts:
  ctx.insert("ip_address", "0.0.0.0");
         |
         v
PolicyEngine::evaluate_all(&ctx) returns Vec<PolicyResult>
         |
         v
Author checks results: PolicyResult for "no-public-ips" contains
  Violation { fact: "ip_address", severity: Critical, rule_type: Deny }
         |
         v
Service rejects the request and logs the violation
```

**Actors:** Governance author, service, policy engine
**Goal:** Codify and enforce governance rules without recompiling the service.
**Success criteria:**
- Policy files load and evaluate without code changes.
- `Deny` violations at `Critical` severity are returned for matching facts.
- An invalid TOML file produces a typed `PolicyEngineError` at load time.

---

## UJ-5: Service Author Models an Order Lifecycle with the State Machine

```
Author defines OrderState enum implementing State trait:
  Draft (ordinal 0), Confirmed (1), Shipped (2), Delivered (3)
         |
         v
Author constructs StateMachine::new(OrderState::Draft)
with forward_only = true
         |
         v
Author registers transitions:
  (Draft -> Confirmed, guard: inventory_available)
  (Confirmed -> Shipped, guard: payment_captured)
  (Shipped -> Delivered, guard: delivery_confirmed)
         |
         v
Valid transition: sm.transition(OrderState::Confirmed) succeeds
when inventory_available guard returns true
         |
         v
Invalid guard: transition returns Err(StateMachineError::GuardRejected)
when payment_captured guard returns false
         |
         v
Backward transition: sm.transition(OrderState::Draft) returns
Err(StateMachineError::BackwardTransitionForbidden) — forward-only enforced
         |
         v
Author calls sm.history() and sees full ordered transition log
for audit and debugging
```

**Actors:** Service author, state machine
**Goal:** Enforce entity lifecycle state progressions with guard-validated transitions
and a full history log.
**Success criteria:**
- Valid forward transitions succeed when guards pass.
- Guard failures and backward transitions return typed errors.
- `history()` contains a correct ordered record of all accepted transitions.

---

## UJ-6: TypeScript Service Author Creates a Typesafe Entity ID

```
TypeScript service imports @helios/ids
         |
         v
Author calls generateId("agent") -> "ag_01H9ZXK2..."
         |
         v
Author calls validateId(id) -> { valid: true }
         |
         v
Author calls parseId(id) -> { entityType: "agent", ulid: "01H9ZXK2..." }
         |
         v
Author stores id in database; database sorts correctly by creation time
because ULID is monotonically increasing
         |
         v
Author uses generateCorrelationId() for request tracing across services
```

**Actors:** TypeScript service author
**Goal:** Generate globally unique, sortable, self-describing entity IDs without a
coordination service.
**Success criteria:**
- `generateId` produces IDs matching `/^[a-z]{2,3}_[0-9A-HJKMNP-TV-Z]{26}$/`.
- `parseId` recovers the entity type from the prefix without a DB lookup.
- IDs are sorted correctly by their ULID component (monotonic within millisecond).
