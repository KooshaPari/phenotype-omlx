# Specifications

## Domain Contract

The pure Rust core defines ModelPlan, OperatorPlan, StatePlan, KernelKey, KernelCandidate,
TuningRecord, ExecutionTrace, and QualityGate. Infrastructure implements compilation, device
inspection, persistence, model loading, and native invocation behind narrow ports.

## Execution Plan Requirements

ModelPlan must describe ordered or dependency-linked operators, persistent state, precision and
quantization policy, shape constraints, and scheduler semantics. OperatorPlan must represent:

- dense and grouped matrix multiplication;
- GQA, MLA, CCA, tree attention, and paged attention;
- sparse MoE routing, expert execution, shared experts, and reduction;
- DeltaNet, convolution, scan, and recurrent state updates;
- speculative and multi-token proposal and verification;
- masked diffusion denoise and remask steps;
- sub-byte and ternary encode, decode, and fused compute.

## Kernel Selection Contract

KernelKey includes operator kind, dimensions, strides, dtype, quantization format, state layout,
device fingerprint, and policy version. A candidate is eligible only after reference-oracle tests,
determinism checks where required, memory-safety checks, and tolerance validation. TuningRecord
stores candidate identity, measurements, variance, environment, compiler versions, and expiry.

## Model Acceptance Matrix

| Model family | Mandatory acceptance path |
|---|---|
| Qwen agentic | Long-context decode, tool-use traces, GQA or DeltaNet state, sparse MoE |
| DeepSeek | MLA, routed experts, proposal or MTP path |
| LFM | Convolution-attention schedule and sparse experts |
| ZAYA | CCA and compact nonlinear expert path |
| Bonsai | Exact ternary block layout and round-trip oracle |
| Diffusion | Parallel denoise, confidence/remask scheduling, variable active set |
| Recurrent hybrids | State continuity, scan equivalence, bounded memory |

## Native ABI v1

The ABI uses versioned descriptors, caller-owned input and output buffers where possible, explicit
capacity and written lengths, status codes, backend-owned opaque handles only when unavoidable,
and a matching release function for every allocation owner. It forbids allocator crossing,
sentinel-only error signaling, and partially initialized outputs after failure.

## Quality and Governance

Selection records and benchmark results are immutable artifacts keyed by source revision and
environment. Quality regressions override speedups. Experimental kernels remain selectable only
through explicit policy and never become production defaults without evidence promotion.

## Assumptions, Risks, Uncertainties

| Item | Mitigation |
|---|---|
| Metal compilation variance | Warmups, repeated samples, confidence bounds, cache fingerprints |
| Sparse workloads vary sharply | Shape-bucketed candidate sets and trace-derived benchmarks |
| Cross-language allocator mismatch | Caller-owned buffers and ABI conformance tests |
| Model configs drift | Parse checkpoint config into validated ModelPlan; reject unknown contracts |
| Eval contamination or weak judging | Dataset provenance, deterministic scoring, multiple judge modes |
