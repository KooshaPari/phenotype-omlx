# Forward Execution DAG — perf-core Polyglot Workspace

> Generated 2026-07-23 from session context.
> Worktree: `repos/worktrees/phenotype-omlx/langfuse-live-judge/perf-core`

---

## Phase Inventory

| Phase | ID  | Title | Items | Done | Remaining | Status |
|-------|-----|-------|-------|------|-----------|--------|
| G0 | Ground-truth snapshot | 3 | 3 | 0 | COMPLETE |
| A | Auditing | 3 | 0 | 3 | PENDING |
| B | Core domain + ABI | 3 | 3 | 0 | COMPLETE |
| C | Polyglot bindings | 4 | 2 | 2 | PARTIAL |
| D | Frontends | 2 | 0 | 2 | PENDING |
| E | Evaluation | 6 | 0 | 6 | PENDING |
| F | Validation | 3 | 0 | 3 | PENDING |
| G | Governance | 5 | 0 | 5 | PENDING |

**Overall: 12 / 29 items complete**

---

## Dependency Graph (ASCII)

```
G0 ──────────────────────────────────────────────────────────────┐
                                                                 │
A1 ──┐                                                          │
A2 ──┼── A3                                                     │
     │   │                                                       │
     │   ├── B1 ─── B2 ─── B3                                   │
     │   │                                                        │
     │   │   ┌──────────────────────────────────────────────────┐ │
     │   │   │            Polyglot Binding Layer                │ │
     │   │   │                                                  │ │
     │   │   ├── C1 (Zig)    ── DONE                           │ │
     │   │   ├── C2 (Mojo)   ── NEEDS WORK ──┐                │ │
     │   │   ├── C3 (Nim)    ── DONE         │                │ │
     │   │   ├── C4 (Go)     ── NEEDS WORK ──┤                │ │
     │   │   │                                │                │ │
     │   │   └────────────────────────────────┘                │ │
     │   │        │                                             │ │
     │   │        ├── D1 (Python/PyO3)                         │ │
     │   │        │    └── E1 (eval schema)                    │ │
     │   │        │         ├── E2 (MMLU)                      │ │
     │   │        │         ├── E3 (GPQA)                      │ │
     │   │        │         ├── E4 (Terminal-Bench)            │ │
     │   │        │         ├── E5 (perplexity/logprob)        │ │
     │   │        │         └── E6 (frontier comparison)       │ │
     │   │        │                                             │ │
     │   │        ├── D2 (Go control-plane)                    │ │
     │   │        │    └── F1 (microbench harness)             │ │
     │   │        │         └── F2 (cross-lang conformance)    │ │
     │   │        │              └── F3 (NIAH + correlation)   │ │
     │   │        │                                             │ │
     │   │        └── G1 ── G2 ── G3 ── G4 ── G5             │ │
     │   │                                                      │ │
     │   └──────────────────────────────────────────────────────┘ │
     │                                                            │
     └────────────────────────────────────────────────────────────┘
```

### Simplified Forward Edges

```
G0 ──→ A1,A2,A3 ──→ B1 ──→ B2,B3 ──→ C1,C2,C3,C4
                                                 │
                         ┌───────────────────────┘
                         ▼
                    D1,D2 ──→ E1 ──→ E2,E3,E4,E5,E6
                                     F1 ──→ F2 ──→ F3
                         │
                         ▼
                    G1 ──→ G2 ──→ G3 ──→ G4 ──→ G5
```

---

## Detailed Phase Definitions

### G0: Ground-truth Snapshot

**Status: COMPLETE**

| Item | Task | Status |
|------|------|--------|
| G0.1 | `cargo check --workspace` passes clean | DONE |
| G0.2 | Verify toolchain versions (Rust nightly, Zig 0.16, Nim 2.2.10, Mojo 1.0, Go 1.26.5) | DONE |
| G0.3 | Capture baseline test counts per crate | DONE |

**Dependencies:** None (entry point).
**Blocked by:** Nothing.

---

### A: Auditing

**Status: PENDING**

| Item | Task | Depends On | Status |
|------|------|------------|--------|
| A1 | Toolchain/reproducibility audit | G0 | PENDING |
| A2 | Existing architecture/API audit | G0 | PENDING |
| A3 | Benchmark/eval-suite selection (≤10 benchmarks) | A1, A2 | PENDING |

**Dependencies:** Requires G0 baseline to be established. A3 requires both A1 and A2 outputs to inform benchmark selection.

---

### B: Core Domain + ABI

**Status: COMPLETE**

| Item | Task | Depends On | Status |
|------|------|------------|--------|
| B1 | Canonical Rust domain + ABI contract (native-abi v1) | G0 | DONE |
| B2 | C ABI reference implementation (turbo-quant-c) | B1 | DONE |
| B3 | Rust SIMD/nightly implementation (turbo-quant with portable_simd) | B1 | DONE |

**Dependencies:** B2 and B3 both require B1 (the canonical ABI contract). B2 and B3 are parallelizable.

---

### C: Polyglot Bindings

**Status: PARTIAL (2/4 done)**

| Item | Task | Depends On | Status |
|------|------|------------|--------|
| C1 | Zig binding + roundtrip tests | B2 | DONE |
| C2 | Mojo binding + native kernel | B2, B3 | NEEDS WORK — shared lib not built |
| C3 | Nim binding + roundtrip tests | B2 | DONE |
| C4 | Go binding + ownership-safe tests | B2 | NEEDS WORK — memory safety issue |

**Dependencies:** All bindings depend on B2 (C ABI reference). C2 additionally depends on B3 (Rust SIMD kernels). C1–C4 are parallelizable once B2 lands.

**Remaining Work:**
- **C2:** Build the shared library (`libturbo_quant_mojo.so`/`.dylib`) so Mojo can load it. Verify kernel dispatch.
- **C4:** Resolve memory safety issue in Go FFI bindings. Likely involves fixing cgo pointer-passing rules or introducing a Rust-managed allocator.

---

### D: Frontends

**Status: PENDING**

| Item | Task | Depends On | Status |
|------|------|------------|--------|
| D1 | Python/PyO3 frontmatter | C1–C4 | PENDING |
| D2 | Go control-plane/benchmark runner | C4 | PENDING |

**Dependencies:** D1 requires all bindings (C phase) to be stable. D2 requires C4 (Go binding) to be functional.

---

### E: Evaluation

**Status: PENDING**

| Item | Task | Depends On | Status |
|------|------|------------|--------|
| E1 | Unified eval schema | D1 | PENDING |
| E2 | MMLU loader/scorer | E1 | PENDING |
| E3 | GPQA loader/scorer | E1 | PENDING |
| E4 | Terminal-Bench task adapter | E1 | PENDING |
| E5 | Perplexity/logprob evaluator | E1 | PENDING |
| E6 | Stock/frontier model comparison matrix | E1 | PENDING |

**Dependencies:** All eval items require E1 (unified schema). E2–E6 are parallelizable after E1. Requires D1 (Python frontend) for data loading and scoring.

---

### F: Validation

**Status: PENDING**

| Item | Task | Depends On | Status |
|------|------|------------|--------|
| F1 | Microbench harness (encode/decode latency, energy, memory) | D2 | PENDING |
| F2 | Cross-language conformance suite (byte-parity across all backends) | F1 | PENDING |
| F3 | NIAH + quality/perf correlation | F2, E1–E6 | PENDING |

**Dependencies:** F1 requires D2 (Go benchmark runner). F2 requires F1. F3 requires both F2 (conformance) and E outputs (eval results) for correlation.

---

### G: Governance

**Status: PENDING**

| Item | Task | Depends On | Status |
|------|------|------------|--------|
| G1 | Regression baseline artifacts | F1 | PENDING |
| G2 | Full workspace verification | G1 | PENDING |
| G3 | Security/FFI/memory audit | G2 | PENDING |
| G4 | Architecture/docs review | G3 | PENDING |
| G5 | Commit + airlock snapshot | G4 | PENDING |

**Dependencies:** Linear chain G1 → G2 → G3 → G4 → G5. G1 starts once microbench harness (F1) produces baselines. G5 is the final gate.

---

## Subagent Ownership

| Phase | Agent Type | Rationale |
|-------|------------|-----------|
| G0 | **explore** | Read-only verification of toolchain state and cargo check output. No implementation. |
| A | **explore** | Audit and reconnaissance — reading source, cataloguing APIs, surveying benchmarks. |
| B | **general** | Core implementation — writing Rust ABI contract and SIMD kernels. Requires synthesis. |
| C1 | **general** | Zig binding + roundtrip tests. Known-good path, bounded scope. |
| C2 | **general** | Mojo binding — needs debugging shared-lib build. Requires iterative problem-solving. |
| C3 | **general** | Nim binding + roundtrip tests. Known-good path, bounded scope. |
| C4 | **general** | Go binding — memory safety issue needs diagnosis and fix. Requires iterative debugging. |
| D1 | **general** | Python/PyO3 frontmatter — integration work across binding layer. |
| D2 | **general** | Go control-plane + benchmark runner — orchestration code. |
| E1 | **general** | Schema design — defines the contract for all eval items. |
| E2–E6 | **explore** | Each is a self-contained loader/scorer — reads data, applies schema, produces results. Parallelizable. |
| F1 | **general** | Microbench harness — requires instrumentation and measurement infrastructure. |
| F2 | **general** | Cross-language conformance — coordinates across all backends, needs holistic view. |
| F3 | **explore** | Correlation analysis — data processing and statistical analysis over E+F outputs. |
| G1 | **explore** | Baseline artifact capture — read-only, snapshot-oriented. |
| G2 | **explore** | Full workspace verification — `cargo check`, test runs, lint passes. |
| G3 | **general** | Security audit — requires reasoning about FFI boundaries and memory safety. |
| G4 | **explore** | Architecture/docs review — reading and assessing, not writing new code. |
| G5 | **general** | Commit and airlock — requires git operations and snapshot management. |

### Swarm Deployment Notes

- **A2–A3, E2–E6, G1–G2, G4** can be dispatched as a parallel swarm (all explore-type, no shared state).
- **C1–C4** can run in parallel once B2 is confirmed stable.
- **E2–E6** can run in parallel once E1 is defined.
- **G3** should run after G2 passes — it reasons over the verified state.

---

## Critical Path

```
G0 → A1,A2 → A3 → B1 → B2,B3 → C1–C4 → D1,D2 → E1 → F1 → F2 → F3 → G1 → G2 → G3 → G4 → G5
```

**Estimated bottleneck:** The C2 (Mojo) and C4 (Go safety) items are the only partially-complete items and gate D1/D2. Resolving these unblocks the entire downstream pipeline.

---

## Completion Criteria

The DAG is fully resolved when:

1. All 29 items across G0–G are marked DONE.
2. `cargo check --workspace` passes clean.
3. All six backends (Rust, C, Zig, Mojo, Nim, Go) produce byte-parity output on reference inputs.
4. Eval schema loads MMLU/GPQA/Terminal-Bench data and produces scores.
5. Microbench harness reports latency, energy, and memory per backend.
6. Regression baselines are committed and airlocked.
7. Security audit passes with zero high/critical findings.

---

## File Locations

All implementation artifacts live under the perf-core workspace root:

```
perf-core/
├── native-abi/           # B1: Canonical ABI contract
├── turbo-quant-c/        # B2: C ABI reference
├── turbo-quant/          # B3: Rust SIMD/nightly
├── turbo-quant-zig/      # C1
├── turbo-quant-mojo/     # C2
├── turbo-quant-nim/      # C3
├── turbo-quant-go/       # C4
├── eval-harness/         # D1, E1–E6
├── fleet-proto/          # D2: Go control-plane
├── regress-baseline/     # G1: Baseline artifacts
└── forward_execution_dag.md  # This file
```
