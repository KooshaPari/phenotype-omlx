# DAG and Work Breakdown

## Dependency Graph

    R0 evidence and red-test inventory
      -> R1 execution-plan domain
      -> R2 kernel registry and tuning store
      -> R3 native ABI v1
    R1 -> K1 attention and state kernels
    R1 -> K2 sparse MoE kernels
    R1 -> K3 recurrent and convolution kernels
    R1 -> K4 diffusion scheduler and kernels
    R1 -> K5 ternary and sub-byte kernels
    R2 + K1..K5 -> I1 runtime selection and observability
    R3 + K1..K5 -> I2 Zig, Mojo, C, Nim, and Go integration
    I1 + I2 -> V1 model-family conformance
    V1 -> V2 quality, performance, energy, and stability gates
    V2 -> G1 promotion governance and release evidence

## Critical Path

1. Repair the workspace test baseline and eval-harness public contract.
2. Add execution-plan types and reference interpreter with contract tests.
3. Add kernel registry, deterministic selector, tuning records, and trace schema.
4. Correct tree-attention and speculative-state semantics against scalar oracles.
5. Establish Native ABI v1 and migrate C/Zig first.
6. Implement and benchmark model-family kernel packages.
7. Run real model, agentic trace, NIAH, quality, and stability acceptance.

## Parallel Work Packages

| Lane | Scope | Dependency | Exit evidence |
|---|---|---|---|
| A | Eval harness correctness | R0 | Workspace green; deterministic loaders and scoring |
| B | Domain and registry | A baseline | Contract tests and serialized plans |
| C | Attention and speculation | B | Oracle parity and memory bounds |
| D | MoE, recurrent, diffusion | B | Family-specific conformance and benchmarks |
| E | ABI and polyglot | B | Sanitized cross-language round trips |
| F | AX, DX, UX, governance | B and registry | CLI reports, traces, promotion controls |
| G | Full acceptance | C through F | Reproducible regression bundle |

## Review Gates

Each implementation package receives specification review, code-quality review, targeted tests,
workspace tests, benchmark comparison, and Airlock snapshot before the next dependent package.
