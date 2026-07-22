# Application-Layer Ownership Specifications

## D0 Ownership Contracts

### phenotype-contracts

Owns versioned, implementation-neutral schemas and behavioral constants shared across
repositories: identifiers, request/result envelopes, task states, error taxonomy, capability
facts, retry classifications, and telemetry field names. It MUST NOT own orchestration,
routing decisions, provider credentials, process lifecycle, or hardware discovery.

### substrate

Owns application use cases, orchestration and durable task lifecycle, cancellation, domain
ports, and composition of driven adapters. It MUST depend on contracts rather than concrete
router/proxy implementations. It MUST NOT own provider-selection algorithms, provider auth,
agent terminal mechanics, or capacity discovery.

### OmniRoute

Owns provider-neutral routing policy, request/response translation, fallback and retry-budget
application, cost/quality evaluation, router replay, and MCP/A2A application ingress. It MUST
expose routing through a substrate port and MUST NOT own task truth, agent process lifecycle,
credential custody, or hardware inventory.

### agentapi++

Owns the agent-facing terminal HTTP/SSE contract, session/process adapter, event ordering,
stream resumption, and terminal cancellation mechanics. It MUST NOT select model providers,
orchestrate multi-step business tasks, or own shared contract schemas.

### cliproxyapi++

Owns provider-specific OAuth/token custody, refresh, provider transport, and normalized proxy
errors at the credential boundary. It MUST NOT choose routing policy, orchestrate tasks, or
persist hardware/capacity truth. Secrets MUST remain out of application telemetry and shared
task payloads.

### hwLedger

Owns observed hardware inventory, capacity/availability facts, evidence timestamps, source
provenance, and staleness. It exposes facts through a read port. It MUST NOT dispatch work,
select providers, host inference, or become the task-state database.

## Cross-Repository Identity Contract

Every boundary envelope MUST carry:

- `schema_version`: immutable contract version or content-addressed revision.
- `request_id`: unique per ingress request and unchanged through retries.
- `trace_id`: W3C-compatible distributed trace correlation identifier.
- `task_id`: durable orchestration identity, stable across asynchronous polling/resumption.
- `attempt_id`: unique execution attempt under a task.
- `session_id`: agent/stream session identity when applicable; absent otherwise.
- `route_decision_id`: immutable reference to the evaluated routing decision.
- `capability_snapshot_id`: hwLedger fact snapshot used by a decision, when applicable.

IDs MUST be opaque, non-secret, validated at ingress, propagated without rewriting, and logged
as structured fields. Retries MUST create a new `attempt_id` without changing `request_id` or
`task_id`. No repository may infer domain identity from a transport connection identifier.

## Contract and Conformance Requirements

1. Every consumer pins an exact `phenotype-contracts` version or content SHA.
2. Contract fixtures include success, typed failure, cancellation, timeout, retryable failure,
   unknown-field forward compatibility, and malformed-envelope rejection.
3. Each language implementation runs the same golden vectors and publishes its conformance
   result with the contract revision.
4. HTTP/SSE, MCP/A2A, Protobuf/gRPC, and FFI adapters map to the same domain states; transport
   details never create additional uncontracted task states.
5. Error mapping preserves retryability, provenance, and original safe diagnostic context.
6. Cancellation and deadlines propagate end to end; retry budgets are finite and cannot
   multiply independently at substrate, OmniRoute, and cliproxy layers.
7. Stream producers implement bounded buffering/backpressure and deterministic event ordering.
8. Telemetry follows OpenTelemetry semantic conventions where stable and redacts secrets,
   credentials, unrestricted prompt content, and derived presence data by default.

## D2 Contract-Surface Proposal (Implementation-Ready, Pending G0/G1)

This is a proposal for the future `phenotype-contracts` owner, not a claim that an unresolved
repository exists or that any current implementation conforms. It applies only to the verified
OmniRoute, cliproxyapi++, `pheno-rt-spec-probe`, and Tokn roles. `substrate`, `agentapi++`,
`hwLedger`, and all local inference hosts remain fenced until G0 records their canonical roots.

### Versioning and Envelope

Publish JSON Schema and language bindings under one immutable contract revision. Each envelope
uses `schema_version` in `MAJOR.MINOR` form: consumers reject unsupported MAJOR versions, accept
new optional MINOR fields, and preserve only namespaced `extensions` they understand. Required
top-level fields are `schema_version`, `request_id`, `trace_id`, `task_id`, `attempt_id`, and
`emitted_at`; `session_id`, `route_decision_id`, and `capability_snapshot_id` are required when
the relevant role creates or consumes them. IDs are opaque strings, timestamps are RFC 3339 UTC,
and all payloads have an explicit `kind` discriminator. No token, credential, raw prompt, or
derived presence value is permitted in an envelope or extension.

### Schema Set

| Schema | Required fields | Semantics and owner mapping |
|---|---|---|
| `request-envelope` | common identity fields, `kind`, `deadline_at`, `idempotency_key`, `payload` | OmniRoute translates ingress; cliproxy consumes only provider-scoped request facts; pheno-rt publishes vectors. `idempotency_key` remains stable across attempts. |
| `route-decision` | common identity fields, `route_decision_id`, immutable `policy_revision`, ordered `candidates`, selected candidate, `decision_reason`, `capability_snapshot_id?` | OmniRoute emits it; Tokn may ingest it as immutable accounting evidence; it contains no provider secret. |
| `failure-envelope` | common identity fields, `error_code`, `category`, `retryability`, `safe_message`, `origin`, `retry_after_ms?`, `cause_id?` | Each adapter maps transport/provider failure to this taxonomy without leaking headers, tokens, raw bodies, or stack traces. |
| `retry-budget` | `request_id`, `task_id`, `attempt_id`, `budget_id`, `max_attempts`, `attempt_number`, `remaining_attempts`, `deadline_at` | Exactly one budget is created at ingress. Adapters may consume but not reset, multiply, or independently mint it. Cancellation and expired deadlines prohibit retry. |
| `capacity-snapshot` | `capability_snapshot_id`, `observed_at`, `expires_at`, `source_revision`, `source_kind`, `availability`, `limits`, `evidence_uri?` | Read-only fact shape reserved for future hwLedger. OmniRoute can reference it but cannot create capacity truth; Tokn can retain it for provenance. |

`error_code` is a stable machine enum: `invalid_request`, `unauthenticated`, `unauthorized`,
`resource_mismatch`, `rate_limited`, `deadline_exceeded`, `cancelled`, `unavailable`,
`provider_rejected`, `internal`, and `data_loss`. `category` is one of `client`, `auth`,
`quota`, `transport`, `provider`, `deadline`, `cancellation`, or `internal`. Retryability is
`never`, `after_delay`, or `budgeted`; only `budgeted` can decrement a live retry budget.

### Conformance Vectors and Gates

The contract repository MUST publish language-neutral JSON fixtures with expected normalized
outputs, then run identical vectors in the verified implementations:

| Vector family | Required assertions | Active role coverage |
|---|---|---|
| Identity propagation | IDs remain byte-for-byte stable across ingress, translation, retry, stream event, and Tokn ingestion; only `attempt_id` changes on retry. | OmniRoute, cliproxyapi++, pheno-rt, Tokn |
| Version and extension handling | Reject unsupported MAJOR, malformed timestamps/IDs and missing discriminator; accept an additive optional MINOR field; ignore unknown namespaced extension without changing core semantics. | OmniRoute, cliproxyapi++, pheno-rt |
| Error normalization | Provider/HTTP/SSE timeout, 401/403, 429 with delay, 5xx, malformed stream, client cancellation, and deadline map to one safe `failure-envelope`. | OmniRoute, cliproxyapi++ |
| Retry budget | Nested router/proxy retries cannot exceed `max_attempts`; cancellation, non-idempotent request, or deadline exhaustion produces zero further calls. | OmniRoute, cliproxyapi++ |
| Capacity facts | Expired, malformed, or unknown-source snapshots are rejected for policy input; valid snapshots are referenced by ID only and retained unchanged in ledger records. | OmniRoute, Tokn, pheno-rt |
| Security redaction | Synthetic bearer token, OAuth code, raw prompt, tool arguments, and presence marker are absent from serialised envelopes, logs, replay, and telemetry fixtures. | OmniRoute, cliproxyapi++, Tokn |
| Stream behavior | Bounded buffering, deterministic sequence number/order, terminal event, resumption cursor, and cancellation are identical across HTTP/SSE mappings. | OmniRoute, pheno-rt; agentapi++ only after G0 |

G2 completes only when schemas, compatibility policy, generated bindings, and these vectors are
published at one immutable revision. G3 completes only when each active implementation emits a
machine-readable conformance result naming that revision. Performance work may begin only after
G3; it must benchmark the same vectors and preserve their semantics.

## Delivery Gates

| Gate | Required evidence | Blocks |
|---|---|---|
| G0 Root identity | Each named repository is an independent Git root with recorded remote, branch, and clean/dirty state. A bounded protocol source does not establish ownership for the other named repositories or for `phenotype-contracts`. | all cross-repo changes |
| G1 Ownership | Approved D0 owner/forbidden-responsibility matrix and removal plan for overlaps | adapter design |
| G2 Contract | Owner-approved canonical shared-contract URL plus immutable tag/content SHA, versioned schemas, golden vectors, compatibility policy, and generated bindings | integration code |
| G3 Conformance | All active language implementations pass identical contract vectors | end-to-end tests |
| G4 Resilience | Deadline, cancellation, idempotency, backpressure, and single retry-budget fault tests | production replay |
| G5 Security | OAuth/resource binding, secret-redaction, threat model, SBOM, provenance and signing evidence | release |
| G6 Observability | End-to-end ID propagation and trace reconstruction across every boundary | performance approval |
| G7 Benchmark | Reproducible production-shaped baseline and candidate comparison with quality/correctness guardrails | polyglot rewrite or rollout |

## Additional Normative Requirements

1. G2 MUST pin the MCP and A2A versions, then supply shared vectors for authorization,
   AgentCard discovery, task states, cancellation, stream ordering, resumption, and malformed
   envelopes. MCP HTTP authorization and stdio credential handling remain separate adapters.
2. `cliproxyapi++` MUST validate OAuth resource/audience binding and MUST NOT forward an
   upstream token unchanged to a different downstream resource. Redirect URIs require exact
   matching; sender-constrained tokens require an explicit threat-model decision.
3. Every OpenAI-compatible HTTP/SSE ingress MUST have a versioned OpenAPI description and the
   same behavioral/negative streaming vectors as its language peers. OAS schema validity alone
   is insufficient for semantic compatibility.
4. Router replay records MUST include immutable policy/version, candidate set, chosen action and
   propensity, safe outcome/quality signal, realized cost/latency, and censoring. A learned or
   exploring policy MUST pass high-confidence off-policy comparison against a baseline, canary
   limits, and an operator kill switch before production traffic.
5. G6 MUST pin an OpenTelemetry GenAI semantic-convention revision. Route/task fields are
   namespaced application attributes; tests MUST prove prompt, credential, and presence-data
   redaction.
6. G5 release evidence MUST cover every Rust native library/wheel and Go/TypeScript artifact:
   locked dependencies and toolchain, clean rebuild/hash comparison, SBOM, and provenance
   verification against an allowed signer-builder pair.
7. A Kubernetes Gateway API Inference Extension integration, if adopted, is a driven
   Endpoint-Picker/capability adapter. It MUST NOT own substrate task truth or OmniRoute policy.
8. G7 benchmarks MUST compare cold and warm boundary crossings, serialization/copy, startup,
   cancellation, and failure recovery with confidence intervals and recorded noise controls.

## Component-Specific, Benchmark-First Polyglot Policy

No language is categorically approved or prohibited. Rust, Go, TypeScript, Python, Zig, C, C++,
Mojo, Nim, Pony, Vale, or another runtime may be proposed only for a named verified application
component and a measured bottleneck. Existing implementations are baselines, not language bans.
This policy covers routers, credential proxies, protocol adapters, telemetry/cost ledgers, and
control planes only; local inference hosts and engine internals remain explicitly excluded.

| Verified role / current baseline | Allowed optimization target | Required benchmark and correctness gate | Boundary rule |
|---|---|---|---|
| OmniRoute provider-neutral router and HTTP/SSE ingress (TypeScript with Rust crates) | Translation, policy evaluation, stream framing, cache lookup, and replay ingestion. | Production-shaped replay with fixed provider fixtures: p50/p95/p99 route latency, time-to-first-event, events/s, allocation/copy, event ordering, cancellation, retry-budget behavior, and equivalent normalized response/error vectors. | Keep policy behind a substrate routing port; use generated network contracts by default. An in-process N-API/C ABI binding needs measured end-to-end gain including packaging/startup. |
| cliproxyapi++ OAuth/provider proxy (Go) | Token refresh scheduling, credential-store access, provider translation, rate-limit accounting, and stream proxying. | Concurrent refresh/expiry/revocation and stream-cutoff fault matrix; measure p99 authorization overhead, refresh contention, connection reuse, memory, and retry amplification. Prove resource/audience binding, redaction, and no token pass-through. | Credentials stay in the Go boundary. A Rust/C/C++ helper may receive only capability-scoped, non-secret data through an explicit ownership and error contract. |
| `pheno-rt-spec-probe` router protocol (JSON Schema/reference vectors) | Schema validation, envelope serialization, and conformance tooling. | Cross-language golden/negative vectors for unknown fields, malformed input, cancellation, deadline, ordering, and ID propagation; measure validation/serialization cost separately from transport. | Schema and vectors remain implementation-neutral; do not embed route policy, credential custody, or task truth in a protocol runtime. |
| Tokn usage/cost/telemetry ledger (Rust) | Ingestion normalization, aggregation, query, and durable artifact generation. | Representative trace replay with p50/p95/p99 ingest/query, throughput, storage amplification, recovery, provenance retention, and exact accounting parity. | Consume immutable route/task facts; it does not select routes or hold provider credentials. Bind FFI only if the measured crossing cost is lower than a versioned process/API boundary. |
| AgilePlus umbrella/control-plane evidence | No implementation choice until a component is recovered as an independent root. | G0 provenance first: canonical remote, commit, clean/dirty state, owner, and contract consumer list. | `agentapi++` and `hwLedger` remain unresolved names, not targets for a speculative rewrite. |

Every proposal MUST publish a reproducible baseline and candidate command, locked toolchain and
dependencies, input corpus/workload revisions, machine/OS and load state, warmup policy, repeat
count, confidence intervals, raw samples, failure/cancellation cases, and a rollback criterion.
Comparison includes p50/p95/p99 latency, throughput, allocation/copy volume, RSS, startup,
serialization, cancellation, failure recovery, build/package size, and boundary overhead. A
change is accepted only when it materially improves the stated service-level objective without
regressing conformance, OAuth/secret safety, traceability, portability, or operator cost.

Prefer versioned network/process contracts for independently deployed components. Use a narrow
C ABI, PyO3, N-API, cgo, UniFFI, WebAssembly component boundary, or generated binding only when
an in-process boundary is required and the benchmark demonstrates a net gain after crossing,
packaging, and recovery costs. ABI versioning, ownership/lifetime rules, panic/exception
containment, cancellation, and artifact distribution MUST be specified before adoption.

## Explicit Exclusions

Inference-engine and local-host internals are outside this specification: vLLM, SGLang,
TensorRT-LLM, llama.cpp, MLX/Metal serving, kernel selection, KV-cache layout, quantization,
batch scheduling, and model execution. The application layer may consume their documented
service endpoints and capabilities only through driven adapters. An inference-engine router is
not the canonical application router.

## ARUs: Assumptions, Risks, and Uncertainties

| Type | Item | Mitigation / decision gate |
|---|---|---|
| Assumption | `phenotype-contracts` remains the neutral schema SSOT. | Confirm repository ownership and current contract revision at G0/G1. |
| Assumption | substrate is the durable orchestration owner. | Inspect recovered source and approve use-case/port inventory before G1. |
| Risk | OmniRoute, substrate, and cliproxy each retry, causing retry amplification. | Define one retry budget in contracts and exercise layered fault injection at G4. |
| Risk | Provider secrets leak through traces, task payloads, or replay fixtures. | Credential-boundary threat model, redaction tests, and synthetic secrets at G5. |
| Risk | agentapi++ README/spec or implementation disagree on terminal stream ownership. | Resolve a single normative protocol and add golden SSE vectors at G2/G3. |
| Risk | hwLedger evolves from fact ledger into dispatcher, duplicating substrate. | Enforce read-only capacity port and forbidden-responsibility test at G1. |
| Risk | FFI rewrite improves microbenchmarks but worsens end-to-end latency/operations. | Require G7 boundary-inclusive benchmark and rollback criteria. |
| Uncertainty | Independent local roots for substrate, phenotype-contracts, agentapi++, and hwLedger were unavailable in the recovery audit; Tokn currently has only a local Airlock remote. | Keep these identities ABSENT/UNRESOLVED until an owner supplies canonical URL and commit; record existing roots/preservation evidence without altering dirty work. |
| Uncertainty | Local `CHATGPT-*.md` corpus and remote desktop evidence were unavailable. | Re-run evidence ingestion when access exists; keep decisions provenance-tagged. |
| Uncertainty | Some candidate research names lack verified authoritative upstreams. | Exclude them until exact official repository/paper identity is established. |
