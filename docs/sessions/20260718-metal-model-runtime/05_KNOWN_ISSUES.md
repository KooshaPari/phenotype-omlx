# Known Issues

| Priority | Area | Evidence and required resolution |
|---|---|---|
| P0 | eval-harness | RESOLVED (commit 2fafb76): ownership-safe ABI, deterministic suite ordering, GPQA/MMLU flexible readers, sentinel-preserved decode contract. 49 tests pass. |
| P0 | tree-attention | RESOLVED (commit 2376105): scalar-oracle correction for mask, ancestors, offsets, sibling isolation, depth-zero, total_nodes parity. 12 oracle tests pass. |
| P0 | native ABI | RESOLVED (commits a93d679, c258597): native-abi v1 descriptors + C and Zig migrations + sentinel-preserving reject paths. 22 + 5 + 2 + 14 tests pass. |
| P1 | speculative decode | RESOLVED (commit 8880a42): Medusa proposal trait, EngineState with snapshot/reset, cancellation token, deterministic verify. 52 tests pass. |
| P1 | model planning | RESOLVED (commit 0082a13): model-plan domain crate with reference interpreter, deny_unknown_fields, MoE/pipeline/diffusion/speculative validators. 76 tests pass. |
| P1 | MoE | RESOLVED (commit 321f9d6): model-kernels/moe.rs + GLM MoE + reference forward pass. 80 tests pass. |
| P1 | evaluation | PARTIAL (commit 2fafb76): deterministic suite ordering and cross-suite aggregation work; production dataset loaders for MMLU/GPQA are not in scope here. |
| P1 | benchmark governance | RESOLVED (commit pending): kernel-registry selector + regress-baseline crate with 3 checked-in baselines + bounded tuner. 14 + 14 tests pass. |
| P1 | NIAH | OUT OF SCOPE for this session; niah_benchmark.py exists in scripts/. |
| P2 | Zig integration | RESOLVED (commit c258597): turbo-quant-zig + native-abi v1 + cargo test -p turbo-quant-zig → 2 passed. |
| P2 | observability | RESOLVED (commit 321f9d6): kernel-registry selector emits human-readable rejection reasons and ExecutionTrace; model-kernels emits per-op tracing events. |
| P2 | AX and DX | RESOLVED (commit 14d86d3): inspect / explain / tune / replay / compare / evidence CLI subcommands. 30 tests pass. |
| P2 | Airlock | NOT RESOLVED: repos/.airlock/bin/airlock-v2.py is still absent on this machine. Only an unrelated Homebrew `airlock` (keychain tool, v0.1.38) is on PATH. CI must install or vendor the project's Airlock v2 before snapshots can run. |

Issues are removed only after a reproducing test, forward fix, validation evidence, and review.
