# Application-Layer Forward DAG Session

## Goal

Define an evidence-backed forward dependency graph and ownership model for the
Phenotype application layer. The result must make cross-repository contracts,
control-plane responsibilities, routing policy, protocol edges, hardware facts,
and delivery gates explicit before implementation work proceeds.

## Scope

Included repositories and concerns:

- `phenotype-contracts`: neutral, versioned schema and behavioral-contract SSOT.
- `substrate`: orchestration use cases, task lifecycle, domain ports, and adapters.
- `OmniRoute`: provider-neutral routing policy, translation, fallback, evaluation,
  MCP/A2A ingress, and router telemetry.
- `agentapi++`: agent-facing terminal HTTP/SSE protocol and session/process adapter.
- `cliproxyapi++`: OAuth/provider credential proxy and provider-specific transport.
- `hwLedger`: observed capacity, hardware inventory, telemetry, and provenance facts.
- Qualifying application-layer routers, CLIs, registries, ledgers, and control-plane
  adapters whose responsibilities cross one or more of the boundaries above.

Excluded from this DAG are inference-engine internals and local model hosts, including
vLLM-, SGLang-, TensorRT-, llama.cpp-, and MLX-serving implementation details. They may
appear only as external driven adapters behind an application-layer port.

## D0 Ownership Summary

```text
phenotype-contracts
  -> contract versions, IDs, envelopes, error and resilience semantics
  -> substrate
       -> orchestration, task lifecycle, policy-independent domain ports
       -> agentapi++ adapter (agent sessions and terminal event streams)
       -> OmniRoute adapter (routing decisions and provider-neutral execution)
            -> cliproxyapi++ adapter (OAuth and provider-specific transport)
       -> hwLedger adapter (read-only capacity/provenance facts)

All runtime components -> conformance tests against phenotype-contracts
All execution edges    -> shared trace/task/request identity and telemetry
```

Ownership is exclusive at the domain level: contracts do not orchestrate; substrate
does not own provider-routing policy; OmniRoute does not own agent process lifecycle or
credential custody; `agentapi++` does not choose providers; `cliproxyapi++` does not
orchestrate tasks; and `hwLedger` records facts rather than dispatching work.

## Success Criteria

1. Every included repository has one primary domain owner and narrow inbound/outbound
   ports; overlapping responsibilities are resolved rather than duplicated.
2. Contract schemas are versioned, pinned, and tested for conformance across language
   boundaries before dependent integration work begins.
3. Request, task, session, and trace identities propagate end to end across every edge.
4. Routing, retry, timeout, cancellation, backpressure, and error semantics are explicit
   and covered by cross-repository contract and replay tests.
5. Security boundaries cover OAuth/token custody, least privilege, auditability, and
   signed/reproducible artifacts without leaking credentials into telemetry.
6. Benchmarks and production-shaped replay gates establish current baselines before any
   hot-path rewrite; changes must improve agreed latency, throughput, memory, or quality
   metrics without reducing correctness or operability.
7. The final DAG exposes dependencies, critical path, acceptance gates, and independently
   deliverable work packages while keeping inference engines outside the application layer.

## Polyglot Principle

There is no blanket language ban. Rust, Go, C, C++, Zig, Mojo, Python, TypeScript, and
other languages remain eligible when repository constraints and measured workloads justify
them. Selection is made per component using production-shaped profiles and includes total
cost at the boundary: serialization, copying, scheduler crossings, startup, packaging,
debuggability, and operational risk. Prefer stable protocol boundaries (versioned Protobuf,
HTTP/SSE, MCP/A2A) between processes; use a narrow C ABI or generated bindings inside a
process only when benchmarks prove that the FFI gain exceeds its safety and maintenance
cost. No rewrite is approved from language preference alone.

## Evidence Limitation

Local `CHATGPT-*.md` conversation-corpus evidence and the Tailscale/OpenSSH desktop source
were not available during the bounded audit. No architectural decision should claim that
corpus as reviewed until access is restored and the evidence is incorporated explicitly.
