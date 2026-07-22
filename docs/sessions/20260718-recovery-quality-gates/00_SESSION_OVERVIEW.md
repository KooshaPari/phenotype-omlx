# Recovery and quality-gate session overview

## Goal

Recover the verified PyO3, readiness, and real-model validation work that was not
captured by the prior Airlock branch, then replace exact generated-text comparison
with production quality gates appropriate for lossy KV-cache quantization.

## Repository state

- Recovery checkout: `/Users/kooshapari/CodeProjects/Phenotype/repos/phenotype-omlx-recovered`.
- Branch: `wip/20260717T0516-64412e90`.
- Baseline commit: `37af3d5759405833faf84fdb11ee66e9d2c20e11`.
- The checkout was clean when this session began.
- The remote Airlock ref contains the baseline FFI, E2E runner, and readiness shell script,
  but not the later split FFI module, correctness tests, readiness helper/tests, E2E
  validator/tests, or the prior session documents.
- The missing batch remains reconstructable from local Codex rollout JSONL records under
  `~/.codex/sessions/2026/07/14/` and `~/.codex/sessions/2026/07/16/`; restoration must be
  reviewed and tested rather than treated as an authoritative Git snapshot.

## Approved quality-oracle decision

Exact generated-text equality is not a valid release claim for a lossy quantizer. The
authoritative gate will compare compacted-cache inference with an FP16-cache baseline on
the same model, tokenizer, token sequence, and deterministic corpus using teacher-forced
next-token negative log-likelihood and perplexity. A deterministic semantic suite will
independently test task behavior, including long-context retrieval. Top-k agreement and
KL divergence are diagnostic signals; they do not replace perplexity or semantic
acceptance.

The local TurboQuant benchmark evidence supports this decision: apparently coherent text
coexisted with catastrophic perplexity (`165.6` versus `6.121` FP16), while the repaired
implementation reached `6.194` (`+1.19%` versus FP16). See
`../turboquant_plus/docs/quality-benchmarks.md:128-152` and
`../turboquant_plus/docs/quality-benchmarks.md:248-259` from the repositories root.

## Success criteria

- Restore only reviewed artifacts whose behavior is verified against the Rust and Python
  APIs; do not replay repository-policy changes.
- The compiled extension is importable as `omlx_research._perf` and rejects malformed,
  non-finite, or inconsistent quantization payloads without panic.
- Teacher-forced baseline and compacted runs score identical target tokens and emit
  aggregate NLL, mean token loss, perplexity, and baseline-relative deltas.
- A deterministic semantic suite passes calibrated acceptance thresholds and never
  executes model-generated code in the host process.
- CI uses a versioned mini corpus for deterministic regression coverage; release validation
  uses a pinned full corpus with recorded identity, model revision, tokenizer revision,
  configuration, and hardware metadata.
- Thresholds are calibrated from recorded baseline distributions before enforcement; no
  unmeasured threshold is presented as validated.
- Missing metrics, non-finite values, corpus drift, model drift, evaluator errors, or zero
  compacted layers fail closed.
- Results are written to a temporary file, flushed, validated as a complete schema, and
  atomically renamed only after every required gate passes; failed runs preserve the last
  known-good result.
- Exact text may be retained as debugging evidence but is never reported as the lossy
  quantization quality oracle.
- Full readiness remains non-green until the external SSD processed-dataset requirement is
  satisfied or a separately specified release fixture policy is approved; the five expected
  processed datasets must not be fabricated silently.

## Scope boundary

This session specifies recovery, quantization correctness, packaging/readiness, and
quality validation. It does not redesign inference-host internals, weaken external dataset
requirements, or claim production quality before calibrated real-model evidence exists.
