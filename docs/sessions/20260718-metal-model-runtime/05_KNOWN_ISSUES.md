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
| P1 | benchmark governance | RESOLVED (commits b13bfe7, 321f9d6): kernel-registry selector + regress-baseline crate with 3 + 3 checked-in baselines + bounded tuner. 14 + 17 tests pass. |
| P1 | NIAH | OUT OF SCOPE for this session; niah_benchmark.py exists in scripts/. |
| P2 | Zig integration | RESOLVED (commit c258597): turbo-quant-zig + native-abi v1 + cargo test -p turbo-quant-zig → 2 passed. |
| P2 | observability | RESOLVED (commit 321f9d6): kernel-registry selector emits human-readable rejection reasons and ExecutionTrace; model-kernels emits per-op tracing events. |
| P2 | AX and DX | RESOLVED (commit 14d86d3): inspect / explain / tune / replay / compare / evidence CLI subcommands. 30 tests pass. |
| P2 | Airlock | NOT RESOLVED: repos/.airlock/bin/airlock-v2.py is still absent on this machine. Only an unrelated Homebrew `airlock` (keychain tool, v0.1.38) is on PATH. CI must install or vendor the project's Airlock v2 before snapshots can run. |
| P2 | clippy governance drift | RESOLVED (in this session, parent `phenotype-registry/clippy.toml`): the obsolete `imports-ignores` field was migrated to `allowed-wildcard-imports = []` to be compatible with the newer toolchain; workspace clippy now compiles cleanly. |
| P2 | Production policy gate enforcement | RESOLVED (this commit, next): `SelectionPolicy::Production { gates, metric }` now actually rejects candidates missing a [`kernel_registry::QualityAttachment`] (rejection reason `MissingQualityEvidence`) and rejects candidates whose attachment fails a gate (`QualityGateFailed { gate, observed, threshold }`); `Metric::EnergyPerOp` and `Metric::Dispatches` rank candidates by their measured joules/dispatches while falling back to `u64::MAX` when not measured. 12 new tests in `tests/governance.rs` lock this in. |
| P2 | PromotionRecord audit trail | RESOLVED (this commit, next): `PromotionRecord::content_hash` + `PromotionRecord::verify_content_hash` + `PromotionRecord::sign_with` + `verify_signature` give a content-addressed, key-symmetric audit trail. Tests cover round-trip and tamper detection. |
| P2 | Airlock | NOT RESOLVED: `repos/.airlock/bin/airlock-v2.py` is still absent on this machine. Only an unrelated Homebrew `airlock` (keychain tool, v0.1.38) is on PATH. CI must install or vendor the project's Airlock v2 before snapshots can run. Snapshots and contract gating remain blocked on this. |
| P1 | Qwen3-Coder-Next acceptance | RESOLVED (commits fc4474c, ceebda5, d563fdf, d85ac8c): DeltaNet + sparse MoE + shared expert end-to-end trace is exercised by `model-kernels/tests/qwen_bonsai.rs` and locked in `regress-baseline` under `qwen_deltanet_moe_end_to_end`; selector coverage in `kernel-registry/tests/sota_operators.rs`. |
| P1 | DeepSeek MLA + MTP | RESOLVED (commits c05e3fa, d563fdf, d85ac8c): `mla_cache_append`, `mla_cache_attend` parity vs `mla_attention`; `mtp_propose` and `mtp_verify` structural kernel; trace baseline `mla_cache_attend`; selector coverage. |
| P1 | LFM2 + ZAYA CCA acceptance | RESOLVED (commits 98939af, 0a7541a, d563fdf, d85ac8c): `cca_block_attend` (ZAYA) and `gated_short_conv1d_step` (LFM2) shipped with oracles; trace baseline `cca_block_attend`; selector coverage. |
| P1 | Mamba/Jamba/RWKV acceptance | RESOLVED (commits fc4474c, d85ac8c): `mamba_selective_scan`, `mamba_selective_scan_chunk`, `rwkv7_time_mix` (4-channel state) shipped with oracles; hybrid integration test `tests/recurrent_hybrid.rs`; selector coverage. |
| P1 | Bonsai ternary + Qwen acceptance | RESOLVED (commits ceebda5, d563fdf, d85ac8c): `ternary_matmul` parity against unpack-then-matmul; Qwen end-to-end DeltaNet + sparse-MoE trace in `qwen_bonsai.rs`; selector coverage. |
| P1 | LLaDA / Dream diffusion | RESOLVED (commits c7c4b52, d85ac8c): `DiffusionDecoder` + `DiffusionStepReport` with `LowConfidence`, `EntropyBased`, `RandomFraction` strategies; `tests/diffusion.rs` exercises both LLaDA and Dream-style acceptance traces; `tests/model_family_conformance.rs` ties every kernel to a model-family row in `02_SPECIFICATIONS.md`; selector coverage. |
| P1 | kernel-registry SOTA coverage | RESOLVED (commit d85ac8c): every new operator family has a selector test in `kernel-registry/tests/sota_operators.rs` (22 tests, 9 families + cross-cutting determinism and dtype-mismatch rejection). |
| P1 | regress-baseline model-family traces | RESOLVED (commit d563fdf): ZAYA / DeepSeek / Qwen end-to-end traces appended under `cca_block_attend`, `mla_cache_attend`, `qwen_deltanet_moe_end_to_end` with reproducible input hashes. |

Issues are removed only after a reproducing test, forward fix, validation evidence, and review.
