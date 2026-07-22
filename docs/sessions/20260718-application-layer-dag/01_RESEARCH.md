# Application-Layer DAG Research

## Evidence Rules

- `verified-local` means the path and relevant text were inspected in the 2026-07-18 checkout.
- `verified-primary` means an official specification, project documentation, upstream GitHub
  repository, or original paper was identified; it is not evidence that local code conforms.
- `unverified` means the expected repository, corpus, remote machine, or upstream identity was
  unavailable or ambiguous. Unverified material is excluded from normative decisions.

## Local Evidence Matrix

| Subject | Evidence | Status | Finding |
|---|---|---|---|
| contracts SSOT | `OmniRoute/docs/contracts/README.md` | verified-local | Names `KooshaPari/phenotype-contracts` as the shared behavioral-contract SSOT and documents pinned conformance plus known resilience deltas. |
| OmniRoute scope | `OmniRoute/AGENTS.md`, `OmniRoute/CLAUDE.md`, `OmniRoute/package.json` | verified-local | Provider-neutral proxy/router with translation, fallback, MCP/A2A, evaluation, and telemetry surfaces. The root is `OmniRoute`, `main` at `30b93d46e3ad108068b47c8922bf594622231ff8`, with three modified routing files; preservation evidence is recorded below. |
| substrate delegation | `phenoAI/Cargo.toml`, `phenoAI/crates/llm-router/Cargo.toml` | verified-local | The local manifest says routing logic lives in `KooshaPari/substrate` and the local crate is an adapter shim. No independent top-level `substrate/` checkout was present, so substrate implementation/state is unverified. |
| agentapi++ | `AgilePlus/docs/pilot/agentapi-plusplus.md`, `AgilePlus/kitty-specs/002-phenotype-modular-arch/tasks/WP14-agentapi-domain-shared.md`, `AgilePlus/agentapi-plusplus-wtrees/` | verified-local, incomplete | Local planning evidence exists, but the named paths resolve to the `AgilePlus` Git root rather than an independent `agentapi++` root. Runtime ownership claims require inspection of its recovered independent repository. |
| hwLedger | `AgilePlus/hwLedger/`, `AgilePlus/hwLedger-wtrees/`, `AgilePlus/kitty-specs/008-temporal-deployment-workflow-migration/tasks/WP12-capacity.md` | verified-local, incomplete | The paths resolve to the dirty `AgilePlus` root rather than an independent `hwLedger` repository. Capacity/provenance ownership is therefore a proposed boundary, not verified implementation. |
| cliproxyapi++ | `cliproxyapi-plusplus/AGENTS.md`, `cliproxyapi-plusplus/README.md`, `cliproxyapi-plusplus/go.mod` | verified-local | Go provider/OAuth proxy surface. Independent `main` root at `e06aa192e517219b3e7d0111db350e9da0a43b83`, with GitHub origin and no short-status changes. |
| shared umbrella state | `AgilePlus/.git`, `AgilePlus/AGENTS.md`, `AgilePlus/Cargo.toml` | verified-local | `AgilePlus` is a dirty `main` root at `a086185b586835a342395c422ffe1bbc71e30e2e`. Its `hwLedger`, `phenotype-omlx`, and `agentapi-plusplus-wtrees` paths are not independent Git roots and must not be treated as final repository identities. |
| recovered target | `phenotype-omlx-recovered/.git` | verified-local | Independent recovered repository used only to hold this session evidence; application-layer ownership is not assigned to it. |

An exact top-level local checkout named `phenotype-contracts`, `substrate`, `agentapi++`, or
`hwLedger` was not found during the bounded scan. Their current branch, dirty state, and full
implementation cannot be claimed as verified from the umbrella paths above.

## 2026-07-19 G0 Recovery Ledger

The former `/Users/kooshapari/CodeProjects/Phenotype/repos` directory is not a Git worktree.
The following are the only application-layer source roots verified in the recovery audit:

| Identity | Root / revision | Remote or preservation evidence | Current state |
|---|---|---|---|
| OmniRoute | `OmniRoute`, `main`, `30b93d46e3ad108068b47c8922bf594622231ff8` | `git@github.com:KooshaPari/OmniRoute.git`; `.preserve/omniroute-20260719T045112Z/all-refs.bundle` verifies and records `refs/phenotype-preserve/20260719T045112Z`. | Three modified files: `open-sse/rpc/polyglotEdges.ts`, `open-sse/rpc/tierResolver.ts`, and `tests/unit/polyglot-tier-resolver.test.ts`. |
| cliproxyapi++ | `cliproxyapi-plusplus`, `main`, `e06aa192e517219b3e7d0111db350e9da0a43b83` | `git@github.com:KooshaPari/cliproxyapi-plusplus.git` | Clean in the inspected status. |
| Tokn | `Tokn`, `main`, `e07124df486bc518f58fdc262e075ef09eee8c30` | Origin is local Airlock bare repository, not a verified external canonical remote. | Clean, but external source provenance remains unresolved. |
| router protocol | `pheno-rt-spec-probe`, `main`, `5b043a1f4b73a5f639dc1288d873b0a17b427736` | `git@github.com:KooshaPari/phenotype-router-spec.git` | Clean, independently verified bounded router-wire source; it does not clear the shared-contract gate. |
| phenotype-omlx (nested) | `phenotype-registry/registry/absorbed-crates/phenotype-omlx`, `chore/archive-no-simd-lib-rs-2026-07-18`, `7aff55c573b889f3a0d17ba7b4a23dfb3b1abe47` | GitHub origin; prior state bundle, patches, and untracked archive checksum-verify under `.preserve/phenotype-registry__phenotype-omlx-20260719T051206Z/`. | Dirty; not an application-layer owner. |
| phenotype-omlx (nested, drift snapshot) | `phenotype-registry/registry/absorbed-crates/phenotype-omlx`, `chore/archive-no-simd-lib-rs-2026-07-18`, `d0bf653832fa20b1ac1b5660ecac31142ed4d311` | GitHub origin; a fresh all-ref bundle, binary worktree/index patches, exact untracked archive, and verified SHA-256 manifest are under `.preserve/phenotype-registry__phenotype-omlx-20260719T105140Z/`; local ref `refs/phenotype-preserve/20260719T105140Z`. | Captured without reset, clean, staging, commit, merge, or remote mutation; status before and after match, including nine modified kernel-registry files and ten untracked paths. |
| phenotype-omlx (nested, current drift snapshot) | `phenotype-registry/registry/absorbed-crates/phenotype-omlx`, `chore/archive-no-simd-lib-rs-2026-07-18`, `4a9dd8345729bc3e967b387124a29f95604cbe33` | GitHub origin; a separate all-ref bundle, binary worktree/index patches, exact untracked archive, and verified SHA-256 manifest are under `.preserve/phenotype-registry__phenotype-omlx-20260719T113644Z/`; local ref `refs/phenotype-preserve/20260719T113644Z`. | Captured after the earlier snapshot’s committed drift, without reset, clean, staging, commit, merge, or remote mutation; status before and after match with `.agileplus/` and `rust_out` untracked. |
| phenotype-omlx (nested, clean current snapshot) | `phenotype-registry/registry/absorbed-crates/phenotype-omlx`, `chore/archive-no-simd-lib-rs-2026-07-18`, `d13280d3744401d1dc70202e4cfa5665002dc98c` | GitHub origin; a separate all-ref bundle, binary worktree/index patches, empty untracked archive, and verified SHA-256 manifest are under `.preserve/phenotype-registry__phenotype-omlx-20260719T125036Z/`; local ref `refs/phenotype-preserve/20260719T125036Z`. | Captured after the prior drift advanced and its untracked files were removed; status before and after are clean, with no reset, clean, staging, commit, merge, or remote mutation by preservation. |
| phenotype-omlx (nested, conflicted current snapshot) | `phenotype-registry/registry/absorbed-crates/phenotype-omlx`, `chore/archive-no-simd-lib-rs-2026-07-18`, `d0020b856e8be14ecf0a806569ff05f09497ba96` | GitHub origin; a separate all-ref bundle, binary worktree/index patches, empty untracked archive, and verified SHA-256 manifest are under `.preserve/phenotype-registry__phenotype-omlx-20260719T190632Z/`; local ref `refs/phenotype-preserve/20260719T190632Z`. | The target advanced after the clean-head request; preservation captured its exact staged/conflicted state without resolving it: `artifact.rs` added, `error.rs`/`lib.rs` staged, and `compile.rs`/`pipeline.rs` unresolved (`UU`). |
| phenotype-omlx (nested, eval-harness conflict snapshot) | `phenotype-registry/registry/absorbed-crates/phenotype-omlx`, `chore/archive-no-simd-lib-rs-2026-07-18`, `ba80d415e6d7a78462f4609b4ccb1f387aa2c4a5` | GitHub origin; a separate all-ref bundle, binary worktree/index patches, empty untracked archive, and verified SHA-256 manifest are under `.preserve/phenotype-registry__phenotype-omlx-20260719T204400Z/`; local ref `refs/phenotype-preserve/20260719T204400Z`. | Captured the exact later operation state without resolution: `eval-harness/src/backend.rs` is `AA`, `eval-harness/src/lib.rs` is `UU`, and `eval-harness/tests/backend_execution.rs` is staged added. |
| phenotype-omlx (nested, clean post-eval snapshot) | `phenotype-registry/registry/absorbed-crates/phenotype-omlx`, `chore/archive-no-simd-lib-rs-2026-07-18`, `bb330734565bc54a4455b01c94a6ce3cde1f7637` | GitHub origin; a separate all-ref bundle, binary worktree/index patches, empty untracked archive, and verified SHA-256 manifest are under `.preserve/phenotype-registry__phenotype-omlx-20260719T205749Z/`; local ref `refs/phenotype-preserve/20260719T205749Z`. | Captured the later clean state after the prior eval-harness conflict/index state was no longer present; preservation did not resolve, stage, reset, merge, or alter remotes. |
| phenotype-omlx (nested, clean current snapshot) | `phenotype-registry/registry/absorbed-crates/phenotype-omlx`, `chore/archive-no-simd-lib-rs-2026-07-18`, `fd98ebdee2999f4da64b6b57962bd5a83a058c84` | GitHub origin; a separate all-ref bundle, binary worktree/index patches, empty untracked archive, and verified SHA-256 manifest are under `.preserve/phenotype-registry__phenotype-omlx-20260719T213107Z/`; local ref `refs/phenotype-preserve/20260719T213107Z`. | Captured the subsequent clean committed advance without reset, clean, staging, merge, or remote mutation by preservation. |
| phenotype-omlx (nested, clean current snapshot) | `phenotype-registry/registry/absorbed-crates/phenotype-omlx`, `chore/archive-no-simd-lib-rs-2026-07-18`, `2cb0f392908fec5db46db65f9b358d72e3905181` | GitHub origin; a separate all-ref bundle, binary worktree/index patches, empty untracked archive, and verified SHA-256 manifest are under `.preserve/phenotype-registry__phenotype-omlx-20260719T213754Z/`; local ref `refs/phenotype-preserve/20260719T213754Z`. | Captured the next clean committed advance without reset, clean, staging, merge, or remote mutation by preservation. |
| AgilePlus umbrella | `AgilePlus`, `main`, `a086185b586835a342395c422ffe1bbc71e30e2e` | GitHub origin; `.preserve/agileplus-20260719T051141Z/` records bundle, patches, and SQLite backup. | Dirty database/generated artifacts and staged specification; not evidence of separate child roots. |

`substrate`, `phenotype-contracts`, `agentapi++`, and `hwLedger` remain **ABSENT/UNRESOLVED**:
no local independent root, canonical remote, or revision was verified. The safe G0 action is to
record this ledger as the current boundary, obtain owner-supplied canonical URL and commit for
each unresolved identity, and only then create a separate clone or worktree. Do not reset,
merge, rename, repoint remotes, or absorb dirty paths as part of recovery.

### Bounded Router-Protocol Finding

`pheno-rt-spec-probe` is a verified independent source only for the phenotype router A2A wire
protocol. Its tracked `README.md` calls the repository the canonical protocol specification;
`CODEOWNERS` assigns `schema/*.json` and `docs/*.md` to `@KooshaPari`. The observed commit is
`5b043a1f4b73a5f639dc1288d873b0a17b427736` on clean `main`, matching `origin/main` at
`git@github.com:KooshaPari/phenotype-router-spec.git`.

This is **not** a global G0 or G2 clearance: the source declares protocol `0.1.0` as Draft,
has no local release/tag ref, and no evidence approves that commit as the immutable shared
`phenotype-contracts` revision. `schema/router-dispatch.json` currently requires only
`version`, `decision_id`, `request_id`, `engine`, `model`, and `created_at`; it does not publish
the D2 shared envelope/schema set, compatibility policy, golden vectors, generated bindings, or
consumer conformance records. G0 remains open for the unresolved repository identities, and G2
remains open for all integration work.

Unblocking evidence is: (1) owner-approved canonical shared-contract URL plus immutable tag or
content SHA; (2) versioned schemas and compatibility policy for the full agreed surface;
(3) language-neutral golden and negative vectors with generated bindings; and (4) each active
consumer's machine-readable conformance result naming that exact immutable revision.

## Primary-Source Matrix

| Area | Primary source | Status | Architectural implication |
|---|---|---|---|
| inference-aware application routing | [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/) and its [specification](https://gateway-api-inference-extension.sigs.k8s.io/reference/spec/) | verified-primary | Keep workload facts and routing policy separable; represent model-serving backends through stable application-layer endpoints rather than importing engine internals. |
| MCP authorization | [MCP Authorization specification](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) | verified-primary | Use OAuth-based authorization, explicit resource/audience binding, least privilege, and no token pass-through across trust boundaries. |
| MCP long-running work | [MCP Tasks](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks) and [SEP-1686 Tasks](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks) | partially verified-primary | Tasks support durable asynchronous lifecycle semantics. Exact SEP number/link identity must be rechecked before an ADR cites it normatively. |
| MCP transport | [MCP Transports](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) | verified-primary | Treat stdio and Streamable HTTP as adapters; isolate transport from domain orchestration and validate origin/auth/session handling. |
| agent interoperability | [A2A specification](https://a2a-protocol.org/latest/specification/), [A2A and MCP](https://a2a-protocol.org/latest/topics/a2a-and-mcp/) | verified-primary | A2A owns agent-to-agent task exchange while MCP exposes tools/context; do not collapse their lifecycle and authorization models. |
| observability | [OpenTelemetry semantic conventions](https://opentelemetry.io/docs/specs/semconv/) and [GenAI conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) | verified-primary | Propagate common trace/task/request IDs and use stable semantic attributes; never record secrets or unrestricted prompt content by default. |
| backpressure | [gRPC flow control](https://grpc.io/docs/guides/flow-control/) and [retry](https://grpc.io/docs/guides/retry/) | verified-primary, content recheck required | Bound streams and centralize retry budgets/idempotency. Page contents should be re-fetched before copying exact normative defaults. |
| router evaluation | [RouteLLM paper](https://arxiv.org/abs/2406.18665) and [lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM) | verified-primary | Evaluate routing with production-shaped quality/cost tradeoffs and replay, not provider count or synthetic throughput alone. |
| secure development | [NIST SP 800-218 SSDF](https://csrc.nist.gov/pubs/sp/800/218/final) | verified-primary | Make threat analysis, dependency review, provenance, and verification explicit release gates. |
| supply-chain integrity | [SLSA specification](https://slsa.dev/spec/) and [Sigstore documentation](https://docs.sigstore.dev/) | verified-primary | Produce attestations/SBOMs and sign artifacts at release boundaries, including native FFI packages. |

## Polyglot and Boundary Finding

No inspected evidence justifies a blanket ban on Mojo, Zig, Nim, Pony, Vale, or any other
language. Existing repo language statements describe current implementation, not a universal
architecture constraint. Select languages only after profiling a production-shaped hot path.
Measure end-to-end p50/p95/p99 latency, throughput, allocation/copy volume, RSS, startup,
serialization, cancellation, and failure behavior. Prefer process boundaries with versioned
Protobuf or HTTP/SSE/MCP/A2A. Approve an in-process C ABI, PyO3, N-API, cgo, UniFFI, or other
binding only when the benchmark includes crossing and packaging costs and demonstrates a
material net gain.

## 2026-07-18 Primary-Source Gap Review

| Priority | Gap in the current plan | Required follow-up | Primary evidence |
|---|---|---|---|
| P0 | MCP/A2A are named but no pinned transport, authorization, AgentCard, task-state, or stream-resumption vectors exist. | Publish a protocol-version matrix and cross-language positive/negative vectors before G2. | [MCP authorization](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization), [A2A v1](https://a2a-protocol.org/dev/specification/) |
| P0 | Resource binding is stated, but OAuth controls are not made testable at the credential boundary. | Require audience/resource validation, no downstream token pass-through, exact redirect URI handling, and a risk review for sender-constrained tokens. | [OAuth BCP 9700](https://www.rfc-editor.org/info/rfc9700/), [RFC 8707](https://www.rfc-editor.org/rfc/rfc8707.pdf) |
| P1 | OpenAI-compatible HTTP/SSE behavior has no canonical artifact. | Version each public HTTP surface as OAS plus behavioral and negative SSE vectors. | [OpenAPI 3.1.1](https://spec.openapis.org/oas/v3.1.1.html) |
| P1 | Replay does not support safe policy learning or counterfactual evaluation. | Log policy/version, candidate set, action propensity, cost/outcome/censoring; gate exploration on baseline/canary, high-confidence OPE, and a kill switch. | [RouteLLM](https://github.com/lm-sys/routellm), [safe exploration](https://arxiv.org/abs/2002.00467), [OPE](https://arxiv.org/abs/1612.01205) |
| P1 | Telemetry is not pinned to a GenAI convention revision. | Version the convention and test namespaced route/task fields plus prompt/credential redaction. | [OTel GenAI](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/) |
| P1 | Provenance and reproducibility lack artifact-level evidence. | Add locked toolchains, clean rebuild/hash comparison, SBOM, and signer-builder provenance verification for each released Rust/Python/Go/TS artifact. | [SLSA provenance](https://slsa.dev/spec/v1.1/provenance), [Rust reproducibility](https://reproducible-builds.org/docs/rust/) |
| P2 | Kubernetes inference routing is not bounded as an adapter. | If Kubernetes is targeted, specify an optional Endpoint-Picker/capability adapter only; do not move application policy into the gateway. | [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/guides/implementers/) |
| P2 | G7 does not require statistical controls for boundary-inclusive benchmarks. | Record cold/warm crossing, copy/serialization, cancellation, package startup, confidence intervals, and noise controls. | [Criterion.rs statistics](https://criterion-rs.github.io/book/user_guide/command_line_output.html) |

## Explicitly Unavailable or Unverified

- A filename-only search of `/Users/kooshapari/Desktop` and `/Users/kooshapari/CodeProjects`
  found no `CHATGPT-*.md` files; no sensitive content was read.
- The desktop source over Tailscale/OpenSSH was unavailable; no remote corpus was scraped.
- No July-2026 future publication beyond the current date was accepted as evidence.
- `RouteLMT`, `R2-Router`, `AgentRFC`, and similarly named items were not assigned an exact
  authoritative upstream identity and must not support decisions until verified.
- Inference engines and local LLM hosts were intentionally excluded; their internal design is
  outside this application-layer research.
