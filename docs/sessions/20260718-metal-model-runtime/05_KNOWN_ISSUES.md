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
| P2 | Production policy gate enforcement | RESOLVED (this commit): `SelectionPolicy::Production { gates, metric }` now actually rejects candidates missing a `QualityAttachment` (rejection reason `MissingQualityEvidence`) and rejects candidates whose attachment fails a gate (`QualityGateFailed { gate, observed, threshold }`); `Metric::EnergyPerOp` and `Metric::Dispatches` rank candidates by their measured joules/dispatches while falling back to `u64::MAX` when not measured. 12 tests in `tests/governance.rs`. |
| P2 | PromotionRecord audit trail | RESOLVED (this commit): `PromotionRecord::content_hash` + `verify_content_hash` + `sign_with` + `verify_signature` give a content-addressed, key-symmetric audit trail. `PromotionValidator::validate` chains gate evaluation + signing + content hash into one safe-construction path. 18 tests in `tests/governance.rs`. |
| P2 | AX promote / quarantine / gates | RESOLVED (commit 61de8a9): `omlx-research promote\|quarantine\|gates` subcommands mirror `kernel_registry::PromotionValidator` in pure Python (stdlib only — hashlib + hmac). 59 new CLI tests cover gate eval, content-hash determinism (sort_keys=True invariant), HMAC signing round-trip, append-only audit ledger, gates CRUD. Python CLI test count: 30 -> 89. |
| P2 | SOTA coverage: Mixture-of-Depths | RESOLVED (commit ae45996): `perf-core/model-kernels/src/mod_routing.rs` ships `mod_route` / `mod_apply` / `mod_scatter_back` for sparse depth routing (deterministic sigmoid(top-k) on capacity_factor in (0,1]). 4 MoD tests + 4 selector tests under OperatorKind::Moe lock the contract. |
| P2 | Perf envelope: dispatch + energy | RESOLVED (commit ae45996): `perf-core/regress-baseline/tests/dispatch_buckets.rs` shape-bucketed regression test across 6 buckets (512x2048^2 to 16384x4096^2). First-run measurements: dispatches 256..16384, energy_per_op_j 1.109e-7..1.190e-7. Ceilings leave ~50% headroom; tighten in follow-up commit. |
| P2 | Perf envelope: shared_expert inner math | RESOLVED (this commit): `shared_expert` (perf-core/model-kernels/src/moe/shared.rs) had a 1..=total divisor-scan preamble plus a strided scalar matmul; dispatch_buckets took ~40 minutes on the 8192×8192×8192 bucket in debug mode on commit ae45996. Cap the divisor scan at `min(x.len(), w.len())`, reorder the matmul to `t → kk → j` with `x[t*k + kk]` hoisted, and manually unroll the inner j-loop by 4. Evidence: dispatch_buckets went from >40 min on ae45996 to **0.64 s end-to-end** in debug on this Apple-silicon machine; sq8k bucket tile time 238 ms → 75 ms; all six buckets now report energy_per_op_j ≈ 3.4e-8 (under the 1.75e-7..1.95e-7 ceilings with ~5× headroom). New regression test `model-kernels/tests/shared_expert_perf.rs` pins a 512×512×4096 single-threaded call to <5 s in debug mode (measured 2.22 s). Public signature of `shared_expert` and the 6-bucket shape table in dispatch_buckets.rs are unchanged. |
| P2 | Airlock | NOT RESOLVED: `repos/.airlock/bin/airlock-v2.py` is still absent on this machine. Only an unrelated Homebrew `airlock` (keychain tool, v0.1.38) is on PATH. CI must install or vendor the project's Airlock v2 before snapshots can run. Snapshots and contract gating remain blocked on this. |
| P1 | Qwen3-Coder-Next acceptance | RESOLVED (commits fc4474c, ceebda5, d563fdf, d85ac8c): DeltaNet + sparse MoE + shared expert end-to-end trace is exercised by `model-kernels/tests/qwen_bonsai.rs` and locked in `regress-baseline` under `qwen_deltanet_moe_end_to_end`; selector coverage in `kernel-registry/tests/sota_operators.rs`. |
| P1 | DeepSeek MLA + MTP | RESOLVED (commits c05e3fa, d563fdf, d85ac8c): `mla_cache_append`, `mla_cache_attend` parity vs `mla_attention`; `mtp_propose` and `mtp_verify` structural kernel; trace baseline `mla_cache_attend`; selector coverage. |
| P1 | LFM2 + ZAYA CCA acceptance | RESOLVED (commits 98939af, 0a7541a, d563fdf, d85ac8c): `cca_block_attend` (ZAYA) and `gated_short_conv1d_step` (LFM2) shipped with oracles; trace baseline `cca_block_attend`; selector coverage. |
| P1 | Mamba/Jamba/RWKV acceptance | RESOLVED (commits fc4474c, d85ac8c): `mamba_selective_scan`, `mamba_selective_scan_chunk`, `rwkv7_time_mix` (4-channel state) shipped with oracles; hybrid integration test `tests/recurrent_hybrid.rs`; selector coverage. |
| P1 | Bonsai ternary + Qwen acceptance | RESOLVED (commits ceebda5, d563fdf, d85ac8c): `ternary_matmul` parity against unpack-then-matmul; Qwen end-to-end DeltaNet + sparse-MoE trace in `qwen_bonsai.rs`; selector coverage. |
| P1 | LLaDA / Dream diffusion | RESOLVED (commits c7c4b52, d85ac8c): `DiffusionDecoder` + `DiffusionStepReport` with `LowConfidence`, `EntropyBased`, `RandomFraction` strategies; `tests/diffusion.rs` exercises both LLaDA and Dream-style acceptance traces; `tests/model_family_conformance.rs` ties every kernel to a model-family row in `02_SPECIFICATIONS.md`; selector coverage. |
| P1 | kernel-registry SOTA coverage | RESOLVED (commit d85ac8c): every new operator family has a selector test in `kernel-registry/tests/sota_operators.rs` (22 tests, 9 families + cross-cutting determinism and dtype-mismatch rejection). |
| P1 | regress-baseline model-family traces | RESOLVED (commit d563fdf): ZAYA / DeepSeek / Qwen end-to-end traces appended under `cca_block_attend`, `mla_cache_attend`, `qwen_deltanet_moe_end_to_end` with reproducible input hashes. |
| P2 | Oversized modules (350/500-line policy) | RESOLVED (commits 058313b, f8ab955, cadea43, bb99d0c, b20f19c, a4a2ffc, 71d6945, d6481a4, 6a0a3bd, adf94af, 87b8abb, d648ac2, e5ecb3f): 14 oversized files split into per-topic submodules — kernel-registry tests/{contracts,governance,sota_operators} (1699→3 dirs, max 368L), model-kernels/tests/contracts.rs (1130→8 files, max 326L), regress-baseline/tests/contracts.rs (716→6 files, max 240L), model-kernels/src/{mla_cache,cca_block,decoder,quantized} (max 224L), and tests/{diffusion,qwen_bonsai,recurrent_hybrid,dispatch_buckets} plus regress-baseline/src/lib.rs. Every public symbol re-exported; callers unchanged. 402 tests across 4 crates all pass deterministically: model-kernels 223, regress-baseline 22, kernel-registry 67, eval-harness 90. |
| P2 | Airlock v2 | NOT RESOLVED: `repos/.airlock/bin/airlock-v2.py` is still absent on this machine. Only an unrelated Homebrew `airlock` (keychain tool, v0.1.38) is on PATH. CI must install or vendor the project's Airlock v2 before snapshots can run. Snapshots and contract gating remain blocked on this. |
| P2 | PromotionRecord canonical-bytes stability | RESOLVED (commit 7dadcef): `PromotionRecord::canonical_bytes` now sorts gates and evidence by `id` before serialisation so the content hash is stable across serde round-trips even if callers mutate the lists in place. Top-level keys remain lexicographically sorted. The proptest fuzz test `content_hash_is_stable_across_serde_round_trip` deterministically exercises 0-3 gate/evidence pairs and confirms the recomputed hash matches the stored hash post-round-trip. |
| P2 | eval-harness module cohesion | RESOLVED (commit 7dadcef): `lib.rs` was 396L (>350 target). Split into `lib.rs` (302L, public surface + re-exports), `backend.rs` (Backend + BackendCompletion), `error.rs` (EvalError), `runner.rs` (run_suite + run_multiple_choice_suite). Caller-side tests unchanged; 50 unit + 6 integration + 16 contracts + 18 backend tests all pass. |
| P2 | Orphaned flat test files | RESOLVED (commit abbbb7c): previous split commits (6a0a3bd diffusion, adf94af qwen_bonsai, 87b8abb recurrent_hybrid, d648ac2 dispatch_buckets) added the new `tests/<topic>/` submodules but left the old flat `tests/<topic>.rs` files in the index while the working tree no longer held them — so `git status` reported them as `deleted`. This commit `git rm`'d those index entries and tightened the unused index binding in `subbyte_pack`. `cargo test --workspace --all-targets` is green: 635 passed, 0 failed, 1 ignored (pre-existing turbo-quant `minmax` microbench). |
| P2 | regress-baseline perf envelope stubbed | RESOLVED (commit 867b1b8): `dispatch_budget` / `energy_budget_j` are no longer `u64::MAX` / `f64::INFINITY` sentinels. They now return per-shape ceilings from `regress_baseline::budget::BUCKETS`, derived from the first observed run on 2026-07-18 with 1.2× (dispatch) and 1.5× (energy) headroom. `tests/dispatch_buckets/` no longer mirrors ceilings locally — it pulls them from the library, so a single edit in `budget.rs` updates every consumer. regress-baseline: 22 → 27 tests; workspace: 635 → 640 passing, 0 failed, 1 ignored. |
| P2 | eval-harness mmlu.rs oversized | RESOLVED (commit 78cb9df): `eval-harness/src/mmlu.rs` (398L) was the last source file in the crate over the 350L soft target. Split into `mmlu/mod.rs` (246L, loaders + their public-API tests) and `mmlu/parser.rs` (394L, `parse_csv` + 19 parser-internal unit tests) along the natural loader/parser seam. The parser-internal tests exercise `parse_csv` directly so the row-level logic is covered independently from the I/O wrapping the loaders perform. One defensive `subject_idx == question_idx` belt-and-suspenders check inside `parse_csv` is intentionally not tested — distinct header names map to distinct positions by construction; the user-visible duplicate-column path is already covered. Public API unchanged: `mmlu::load_csv` / `load_csv_with_provenance` / `load_csv_bytes`. eval-harness lib: 50 → 69 tests; eval-harness full crate: 90 → 109 tests; workspace: 640 → 659 passing, 0 failed, 1 ignored. |
| P2 | mod_routing.rs oversized (408L) | RESOLVED (this turn): split `perf-core/model-kernels/src/mod_routing.rs` (408L, the last file over the 350L target after the previous split commits) into `mod_routing/{mod, route, apply, tests}.rs` (53/114/87/167 lines respectively, all well under 350L). Public API preserved via re-exports in `mod.rs`; `KernelOp::ModRouting` / `tag() == "mod_routing"` unaffected. The 11 inline tests were moved verbatim into `mod_routing/tests.rs` and re-discovered via `#[cfg(test)] mod tests;`. All 11 still pass. |
| P2 | mlx_backend.py test regression on hosts without `mlx_lm` | RESOLVED (this turn): `python/omlx_research/tests/test_mlx_backend.py` was failing 3 production-path tests with `ModuleNotFoundError: No module named 'mlx_lm'` on environments that have `mlx.core` but no `mlx_lm` (e.g. CI runners that only need structural coverage). Added a `_require_mlx_lm()` helper that raises `unittest.SkipTest` when `mlx_lm` cannot be imported, and called it from `TestMlxBackendTurboQuantProduction.setUpClass` BEFORE the model download. Also added two new structural tests for the helper itself (`TestRequireMlxLmHelper`): one verifies the helper degrades to `SkipTest` when `mlx_lm` is hidden via `sys.modules`, the other verifies it returns `None` cleanly when `mlx_lm` is importable. Python suite: 92 passed + 4 skipped → 92 passed + 5 skipped + 2 new passed = 94 passed + 5 skipped (the +1 skip is the happy-path helper test that also requires `mlx_lm` to be importable). |
| P2 | AX: runtime diagnostics missing | RESOLVED (this turn): added `python -m omlx_research.cli doctor` (`python/omlx_research/cli/doctor.py` + `_doctor_shared.py` + `_doctor_checks.py`). 10 checks: python_version (must be ≥ 3.14), mlx_core_available (fail on Apple Silicon if missing), mlx_lm_available (warn if missing), turboquant_rust_extension_available (warn if missing), kernel_registry_version, regress_baseline_version, model_kernels_operator_coverage (≥ 22 tags), native_abi_v1, airlock_v2_installed (warn — the unresolved P2 from this session is documented explicitly in the check details), tests_runnable. Human-readable text mode + `--json` mode. Exit code 0 if all pass, 1 if any warn or fail. 27 new tests in `test_doctor.py`. doctor.py = 178L, _doctor_shared.py = 201L, _doctor_checks.py = 289L, test_doctor.py = 339L — all under the 500L hard cap and three of four under the 350L target. |
| P1 | Qwen3-Coder-Next batched DeltaNet acceptance | RESOLVED (this turn): added `deltanet_batched_chunk` + `deltanet_batched_chunk_stepwise` in `perf-core/model-kernels/src/recurrent/deltanet_batched.rs` (208L). Layout: q/k/v `[batch, num_heads, chunk, head_dim]` row-major; initial_state `[batch, num_heads, head_dim, head_dim]`; outputs and final_state same shape. Implementation reuses `deltanet_step` per (batch, head, chunk) so the oracle is byte-for-byte identical to running `deltanet_chunk` N=batch_size*num_heads times. 12 new tests in `deltanet_batched_tests.rs` (260L) cover 1×1 oracle parity, 2×2 distinct data, 4×3 scale, uniform chunk_size, zero-dim rejection (×4 dims), buffer-length rejection (×2 buffers), random-inputs tolerance, and stepwise-wrapper agreement. New `KernelOp::DeltaNetBatched` variant with tag `"deltanet_batched"`. New kernel-registry selector test `deltanet_batched_selects_metal_for_b2_h2_c4_d8`. model-kernels: 149 → 161 lib tests; recurrent submodule: 29 → 41 tests; workspace: 659 → 673 passing. |
| P2 | Airlock v2 | NOT RESOLVED: `repos/.airlock/bin/airlock-v2.py` is still absent on this machine. Only an unrelated Homebrew `airlock` (keychain tool, v0.1.38) is on PATH. CI must install or vendor the project's Airlock v2 before snapshots can run. Snapshots and contract gating remain blocked on this. The new `omlx-research doctor` subcommand (this turn) surfaces this gap explicitly via the `airlock_v2_installed` check so users see it on every doctor run. |
| P1 | Qwen3-Next sliding-window attention | RESOLVED (commits eeb9d55, 6f7e80c): added `sliding_window_attention` in `perf-core/model-kernels/src/attention/sliding_window.rs` — the canonical Mistral sliding-window causal pattern, `Q at position s attends to K positions in [max(0, s - window_size + 1), min(seq_k, s + 1))`. When `window_size >= seq_k` the output is byte-identical to `gqa_attention` (locked by `sliding_window_matches_gqa_when_window_is_full`). `KernelOp::SlidingWindowAttention` (tag `"sliding_window_attention"`) registered. `kernel-registry/tests/sota_operators/attention_sliding_window.rs` confirms the selector returns a Metal-side candidate for the (seq_q=8, q_heads=8, kv_heads=2, head_dim=64, group_size=4, window_size=4) shape signature. Plan-side: `AttentionKind::SlidingWindow { window_size }` added in `model-plan/src/attention.rs` so a ModelPlan can describe Qwen3-Next long-context layers; serializes with tag `"sliding_window"`. 11 new tests in `sliding_window.rs` cover GQA parity, future-token masking, window-only attendance, all five rejection paths, the `window_size > seq_k` non-panic degenerate case, and prefill/decode shape variants. |
| P2 | clippy -D warnings in kernel-registry + model-plan | RESOLVED (commit 4f06629): five pre-existing clippy lint errors cleared — `needless_borrows_for_generic_args` and `vec_init_then_push` in `kernel-registry/src/quality.rs`, `too_many_arguments` allow on `TuningRecord::from_samples` in `kernel-registry/src/record.rs`, `large_enum_variant` on `SelectionDecision::Chosen` (boxed `TuningRecord` reduces enum size from 544 bytes to ~16 bytes; all read sites use field-access by reference so auto-deref is a drop-in change), `collapsible_match` in `model-plan/src/plan.rs`'s `check_operator_dtype` (collapsed into match guards; behavior identical). kernel-registry and model-plan now pass `cargo clippy --all-targets -- -D warnings`. Remaining clippy debt (~63 errors across 5 other crates — eval-harness upper-case acronyms GPQA/MMLU/HELM; tree-attention manual indexed loops; turbo-quant `div_ceil` + default-then-assign; native-abi build-script empty `writeln!`; spec-decode tests default-then-assign; model-kernels ~30 loop-var-only-used-to-index; fleet-proto-zeromq unused import) is filed for the next lint-clear sweep. |
| P2 | Workspace clippy `-D warnings` lint debt (all crates) | RESOLVED (this turn): cleared every remaining clippy `-D warnings` error across the entire workspace in 9 commits (ac82341, fbbb031, 5cf961f, 8426850, 84de702, 04d6f61, 79eb7b4, bfdb0e4, 438b5dc, 4d011bb). Totals cleared: 34 model-kernels lib errors, 30 native-abi (build.rs + headers.rs mirror + descriptor + dispatch + tests), 7 eval-harness (MMLU/GPQA renamed to Mmlu/Gpqa inside-crate, manual_range_contains, needless_range_loop, unused TaskSpec import), 13 tree-attention (lib + oracle tests), 5 spec-decode tests (field_reassign_with_default), 2 turbo-quant (manual_div_ceil), 1 fleet-proto-zeromq (unused super::*) + 2 turbo-quant-c + 5 metal-runtime. Final 15 model-kernels lib-test errors (needless_range_loop, identity_op, useless_vec, type_complexity, too_many_arguments, hex-digit grouping) cleared in commit 4d011bb. `cargo clippy --workspace --all-targets -- -D warnings` now exits 0 across all 9 published crates. Workspace tests unchanged: 686 passing, 0 failed, 1 ignored. |
| P1 | selector plumbing for new operator families (SlidingWindow, DeltaNetBatched) | RESOLVED (commit 31e250e): added `perf-core/kernel-registry/src/builders.rs` with three pure builder functions that bridge `model-plan::OperatorPlan` into `KernelKey` for the runtime selector: `sliding_window_key(q_heads, kv_heads, head_dim, batch_size, seq_len, group_size, window_size, dtype, device_fingerprint, policy_version)` (encodes `window_size` in `group`; clamps `[1, seq_len]`), `deltanet_batched_key(batch, num_heads, chunk_size, head_dim, dtype, fingerprint, policy_version)`, `deltanet_key(head_dim, chunk_size, dtype, fingerprint, policy_version)`. 16 unit tests cover valid round-trip, clamping, dtype/device_fingerprint/policy_version forwarding, and `operator_kind` discriminant. End-to-end tests in `kernel-registry/tests/sota_operators/builders_integration.rs` (3 tests) assert the selector picks the `DeltaNetBatchedMetal` / `SlidingWindowMetal` candidate when the runtime constructs a `KernelKey` from the builder. Public API re-exported from `kernel-registry::builders`. kernel-registry: 71 → 89 tests (+18); workspace: 686 → 704 passing. |
| P2 | DX: bare `ImportError: No module named 'mlx_lm'` for users | RESOLVED (commit 89d1cac): added `python/omlx_research/cli/_missing_dep.py` with `require_mlx_lm(where)` helper that raises a structured `RuntimeError` (caches result so subsequent calls don't re-pay the import cost) explaining how to install `mlx-lm` and `mlx-core` on Apple Silicon. `omlx-research cmd-inference` now gates on `require_mlx_lm(__name__)` instead of bare `import mlx_lm`. New `omlx-research doctor` check `mlx_lm_required_by_command(cmd)` upgrades `warn` → `fail` when the active command is `run` / `serve` / `eval` and `mlx_lm` is missing. 6 helper tests + 3 doctor tests added. Python suite: 119 passed + 4 skipped → 128 passed + 4 skipped (+9). |
| P2 | Airlock v2 | NOT RESOLVED: `repos/.airlock/bin/airlock-v2.py` is still absent on this machine. Only an unrelated Homebrew `airlock` (keychain tool, v0.1.38) is on PATH. CI must install or vendor the project's Airlock v2 before snapshots can run. Snapshots and contract gating remain blocked on this. The new `omlx-research doctor` subcommand (commit 9f5384d) surfaces this gap explicitly via the `airlock_v2_installed` check so users see it on every doctor run. The `mlx_lm_required_by_command` check (commit 89d1cac) similarly surfaces missing-required-dependency gaps per active subcommand. |

Issues are removed only after a reproducing test, forward fix, validation evidence, and review.
### Qwen3.5 custom gated-delta promotion (open)

An opt-in custom `mx.fast.metal_kernel` replacement is wired for Qwen3.5 gated-delta and has
compiled/executed in a clean MLX runtime. The current environment has incompatible package
metadata in one system interpreter (`huggingface_hub` is incomplete), so native-vs-custom parity
must be rerun in the isolated runtime before promotion. Until then, candidate manifests remain
reference-only and fail closed.

The isolated parity run completed and rejected promotion: native output was
`\\n\\n<think>\\n\\n</think>`, while the custom path produced `110口的` for the same prompt and
token budget. A follow-up run dispatched the custom kernel for all 108 gated-delta calls (no
fallbacks) and still diverged. The experimental path is now disabled by default and requires
`PHENOTYPE_OMLX_ENABLE_CUSTOM_QWEN_KERNEL=1`; see
`research/baselines/qwen35-custom-gated-delta-parity-20260723.json`.

Update 2026-07-25: isolated Qwen3.5-0.8B-OptiQ-4bit validation was rerun after the gating
fix. Short generation and a 128-token run matched native MLX exactly, with custom dispatches
and zero fallbacks; see `research/baselines/qwen35-custom-gated-delta-parity-20260723.json`
and `research/baselines/qwen35-custom-gated-delta-long-parity-20260723.json`. Promotion is
still reference-only until the 8192-token Harbor/Portage evidence envelope and a refreshed
candidate manifest are available.

Harbor execution update 2026-07-26: the cached Qwen3.5 model was served locally from
`mlx_lm.server` and `/v1/models` exposed `mlx-community/Qwen3.5-0.8B-OptiQ-4bit`. Apple
Container initially failed with an XPC service error and succeeded after `container system
start`. A localhost URL was unreachable from the container; the host LAN URL was reachable,
but the request timed out during MLX batched generation. The server logged
`There is no Stream(gpu, 0) in current thread` from `BatchGenerator.prompt`, so Harbor produced
a real trial with reward `0.0`, not a pass. Langfuse credentials were accepted by the runner;
the remaining blocker is the MLX server stream/thread failure under the container workload.

Update 2026-07-26: Apple Container lifecycle handling is now explicit in the Harbor operator.
`scripts/evals/apple_container_preflight.sh` checks `container system status`, starts the
service when stopped, rechecks that it is `running`, and fails with the official
`container system logs` diagnostic when startup does not converge. The focused contract test
`scripts/tests/test_apple_container_preflight.sh` passes with a fake stopped-then-running
service. This addresses the initial XPC prerequisite only; it does not mask or resolve the
separate MLX `Stream(gpu, 0)` generation failure.

Update 2026-07-26 (forward fix): the Python package floor is now `mlx-lm>=0.31.3`, the
first release verified to include the server's thread-local stream initialization. The
one-request NIAH oracle also sends `seed=0`, which selects mlx-lm's deterministic sequential
path instead of `BatchGenerator`; this is a containment measure for older installed runtimes,
not Harbor evidence. `scripts/tests/test_niah_openai_smoke.py` asserts the request contract and
passes. A fresh Harbor run is still required before promotion.

Audit 2026-07-26 (artifact correction): the currently retained Harbor runs are
`harbor-eval`, `harbor-eval-retry`, `harbor-eval-lan`, and `harbor-eval-patched`; each task
artifact has `verifier/reward.txt` equal to `0`, and the agent oracle records connection refusal
or timeout. No `harbor-eval-final` artifact or reward-`1` evidence is present in this checkout.
Promotion and candidate-manifest refresh therefore remain blocked until a new run emits a
task-level result with reward `1` and a successful oracle transcript.

Follow-up audit 2026-07-26: the final artifact is present in the Portage worktree (the prior
correction was scoped to this checkout and is superseded for evidence discovery). The run is
`worktrees/portage/fix-langsmith-importerror/.runs/harbor-eval-final/2026-07-25__19-32-53`, job
`c8e0d681-4754-4f94-8b00-7e82c92ee653`, trial `omlx-niah-api-smoke__ooX9Kjs`. It records one
Apple Container trial using Qwen3.5-0.8B-OptiQ-4bit over the host LAN endpoint, reward `1.0`,
zero errors, zero retries, and no fallback. `run_full_pipeline.sh` accepts the resulting
EvalReport with `pass_at_1=1.0` and one `W-EVIDENCE` warning: the cockpit converter omits
`evidence_label` and consequently defaults the synthesized suite to `reported`. This warning
must be corrected in the new evidence envelope (explicit `live_verified` provenance); it does
not invalidate the task-level live Harbor result. The stale `candidate-manifest.json` remains
unchanged with `evidence_complete=false`; promotion additionally needs a new manifest tied to
the current HEAD, independent FFI evidence, and candidate review.

Update 2026-07-27: the authorized exact Harbor gate is now live-verified with Qwen3.5
(`prompt_tokens=8192`, `context_tokens_exact=true`, thinking disabled, reward 1.0). The host
NIAH matrix also runs baseline/asymmetric/symmetric TurboKV modes at 4096 and 8192 tokens with
6/24 compressed full-attention layers and effective byte reduction. A paired 16384-token
baseline/TurboKV run then completed with 6/24 compressed layers and effective byte reduction
(ratio 0.6317); the earlier interrupted attempt is retained only as historical stability
evidence. The first
matrix attempt without explicit `sitecustomize` loading is retained as `live_failed`; the
benchmark now imports the audited layer explicitly.

Update 2026-07-29 (safety hardening): no model, Harbor, NIAH, or evaluation workload was
launched in this turn because the operator reported system overload/crashes. The
`concurrent-exec` scheduler now uses bounded admission instead of an unbounded queue, rejects
fan-out above a configured cap, and applies per-job deadlines. Focused deterministic unit tests
cover queue overflow, timeout, and fan-out rejection. Native Metal/device parity remains a
separate gate and was intentionally not exercised here.

Update 2026-07-29 (source catalog): `metal-runtime` now assembles checked-in Metal shader
sources for registry-mapped operators in reference mode and tests that each catalog entry is
non-empty and kernel-shaped. This removes the previous source-free stub as the only reference
artifact, but does not claim device compilation or execution; production mode remains
fail-closed until a real Metal compiler/artifact path is wired and verified.

Update 2026-07-29 (toolchain verification): the installed Xcode-beta Metal toolchain compiled
all 17 checked-in shaders and linked a combined `metal-runtime.metallib` through
`scripts/build_metal_runtime_bundle.sh`. This proves source/toolchain compilation only; the
artifact is not yet allowlisted or dispatched on a live model, and its temporary hash is not a
promotion baseline.

Update 2026-07-29 (artifact contract): `scripts/manifest_metal_runtime_artifacts.py` now emits
sorted, compact JSON containing every compiled `.metallib` filename and SHA-256 digest. The
Rust `ArtifactAllowlist::from_manifest_json` parser validates the strict basename/extension and
64-hex digest contract before handing bytes to `MetallibLoader`. Focused artifact tests pass;
the generated manifest is still build output, not promotion evidence, until it is stored in an
immutable candidate envelope tied to a current commit and verified on-device.

Update 2026-07-29 (selector reachability): the dispatch bridge now routes canonical Bonsai
two-bit ternary operators to the checked-in `ternary_pack` Metal source and grouped matmuls to
the MoE dispatch source. Selector coverage is deterministic and tested; this is routing proof,
not device execution or model-quality evidence.

Update 2026-07-29 (native function catalog): `native_catalog` now binds each routed tag to its
concrete Metal `kernel void` symbol and asserts that every symbol exists in checked-in MSL.
Unknown tags fail closed. This closes selector-to-function-name drift, but does not claim that a
device loaded or executed any function.

Update 2026-07-29 (verified bundle binding): `NativeKernelBundle` now loads a manifest-approved
`.metallib` and resolves only known tag/function pairs. Invalid manifests, unallowlisted files,
and unknown tags fail before any Metal device call; native dispatch remains the next gate.

Update 2026-07-29 (cache integration): the Bonsai ternary, MoE router/grouped GEMM, and
diffusion-confidence wrappers now resolve Metal function names through the native catalog before
entering the shared pipeline cache. `cargo check -p metal-runtime --features metal` passes;
device command encoding remains unexercised.

Update 2026-07-29 (diffusion scheduler leaves): active-position compaction and confidence remask
are now represented by deterministic Rust contracts and catalogued Metal kernels. The combined
Xcode-beta bundle compiles 19/19 sources. Device dispatch, trajectory-state persistence, and
Qwen3.5 acceptance remain open; no live workload was run under the overload/crash guard.

Update 2026-07-29b (trajectory state): confidence/entropy/momentum/convergence state is now
implemented and tested in Rust; `diffusion_trajectory_update_f32` is included in the 20/20
Xcode-beta source bundle. This still does not prove device execution, parity, or model quality.

Update 2026-07-29c (plan integration): `StateKind::DiffusionTrajectory` is now available for
serialized model plans. Isolated Cargo validation passes (1/1); runtime allocation and device
dispatch remain open and are not implied by this plan-level test.

Update 2026-07-29d (runtime layout): `DiffusionStateLayout` now makes the mixed `f32`/`uchar`
trajectory allocation explicit and rejects zero-token and `usize` overflow cases. Focused tests
pass 2/2; this is an allocation contract, not device execution evidence.

Update 2026-07-29e (dispatch plan): `DiffusionDispatchPlan` now fixes stage ordering and the
token-sized grid before any Metal command encoder is touched. Focused tests pass 2/2; actual
buffer binding and device dispatch remain open.

Update 2026-07-29f (Metal bindings): feature-gated bindings for active compaction, remask, and
trajectory now compile with the Metal feature and fail closed on shape or command-buffer errors.
They have not been invoked against a device; parity and Qwen3.5 evidence remain open.
