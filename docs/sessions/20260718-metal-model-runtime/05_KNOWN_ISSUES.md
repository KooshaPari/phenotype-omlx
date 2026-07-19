# Known Issues

| Priority | Area | Evidence and required resolution |
|---|---|---|
| P0 | eval-harness | Workspace compilation fails from missing public scoring, report, and loader contracts |
| P0 | tree-attention | Mask orientation, sibling isolation, offsets, and sizing need reference-oracle correction |
| P0 | native ABI | C and Zig ownership/free contracts and partial-allocation cleanup are inconsistent |
| P1 | speculative decode | Medusa proposal path returns no candidates; state ownership is incomplete |
| P1 | model planning | Runtime lacks a first-class model execution plan and per-layer state description |
| P1 | MoE | No fused routing/expert/reduction runtime or model-family conformance path |
| P1 | evaluation | MMLU and GPQA loaders are not production dataset loaders; exact scoring is incomplete |
| P1 | benchmark governance | No registered perf-bench target or locked regression baseline |
| P1 | NIAH | Existing output is empty or does not prove compressed-layer execution |
| P2 | Zig integration | Zig is excluded or version-sensitive and must pass workspace feature tests |
| P2 | observability | Kernel choice, tuning provenance, and rejection reasons are not exposed end to end |
| P2 | AX and DX | No unified plan-inspect, tune, replay, compare, and evidence-export CLI |

Issues are removed only after a reproducing test, forward fix, validation evidence, and review.
