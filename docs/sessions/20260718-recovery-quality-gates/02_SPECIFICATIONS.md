# Recovery quality-gate specifications

## Evaluation input contract

Each evaluation run receives an immutable manifest containing:

- model identifier, immutable model revision, model-content digest, and architecture;
- tokenizer identifier, immutable revision, digest, and special-token configuration;
- corpus identifier, version, content digest, ordered sample identifiers, and license metadata;
- evaluation tier (`ci` or `release`), random seed, context length, stride, and sample count;
- baseline cache configuration and compacted cache configuration, including key/value bit
  widths, group size, protected boundary layers, and requested compaction policy;
- runtime versions, device identity, operating system, and source commit;
- semantic-suite identifier, version, evaluator digest, and sandbox policy.

The evaluator rejects mutable or missing release revisions, a corpus digest mismatch,
duplicate sample identifiers, an empty corpus, an unsupported model/cache combination, or
a manifest whose baseline and compacted arms do not describe the same model and tokenizer.

## Teacher-forced metric contract

For each sample, tokenize once and use the identical ordered token sequence for both arms.
At token position `i`, both arms receive tokens before `i` and score the observed token at
`i`; generated continuations are never substituted for target tokens. The two arms use the
same context windows, stride, masking, boundary-token policy, and count of scored tokens.
The compacted arm must execute the declared same-cache sequence and report a positive
number of compacted layers when compaction is required.

Per-arm outputs are:

- `scored_tokens`: positive integer;
- `negative_log_likelihood`: finite sum of target-token negative log probabilities;
- `mean_token_loss`: finite `negative_log_likelihood / scored_tokens`;
- `perplexity`: finite `exp(mean_token_loss)`;
- per-sample values for the same fields, keyed by stable sample identifier.

Comparison outputs are:

- `nll_delta` and `mean_token_loss_delta`: compacted minus baseline;
- `perplexity_delta`: compacted minus baseline;
- `perplexity_ratio`: compacted divided by baseline;
- `perplexity_delta_pct`: baseline-relative percentage change;
- identical scored-token and sample counts for both arms;
- the calibrated policy identifier and its provenance record.

An empty score set, non-finite logits or metrics, a zero/negative baseline perplexity,
token-count mismatch, sample-set mismatch, or missing comparison field is an evaluation
error and fails the gate.

## Semantic acceptance contract

The semantic suite is deterministic and versioned. It includes fixed-answer tasks and
long-context retrieval cases whose expected answers and normalizers are committed or
content-addressed. Each evaluator is a narrow pure parser/comparator where possible.
Generated Python, shell, SQL, native code, model tool calls, and network requests are never
executed in the host evaluation process.

If a task genuinely requires execution, it runs in a separately provisioned sandbox with
no inherited secrets, no network by default, a read-only fixture mount, a disposable
working directory, explicit CPU/memory/time/process limits, and captured stdout/stderr.
Sandbox startup failure, timeout, resource-limit breach, malformed output, or evaluator
exception is a failed task rather than a skipped pass.

The result records task totals, pass/fail counts, per-task outcomes, evaluator versions,
and the calibrated semantic policy identifier. Free-form coherence and exact generated
text are debugging artifacts, not acceptance metrics.

## Diagnostic distribution metrics

On the same teacher-forced positions, the evaluator may record mean KL divergence,
same-top-token rate, and top-k overlap between compacted and baseline distributions. The
manifest defines vocabulary masking, numerical precision, KL direction, and `k`. These
diagnostics support regression localization but cannot independently pass a run that
fails perplexity or semantic acceptance.

## Calibration and threshold policy

No production threshold is valid until calibration is complete. Calibration runs repeated
baseline-versus-baseline controls and baseline-versus-compacted candidates over the pinned
release corpus, supported model classes, and declared hardware/runtime matrix. It records
measurement variation, task-level failure distribution, and sensitivity to known-bad
fixtures. Reviewers select limits that separate controls from known regressions with a
documented safety margin. The resulting versioned policy contains the evidence artifact
digests, applicable matrix, approval record, and expiry/recalibration triggers.

Until an applicable policy exists, measurement may produce a calibration artifact but the
release gate reports `uncalibrated` and fails. A policy is inapplicable when the model,
tokenizer, corpus, cache mode, evaluator, or material runtime assumptions differ.

### Offline calibration evidence contract

`scripts/calibration_evidence.py` validates an offline evidence document; it does not load a
model, acquire data, contact a network, publish a result, or authorize a release. The document
records immutable `model_revision`, `tokenizer_revision`, and `runtime_revision`, together with
the canonical token/loss/perplexity/timing metrics and a corpus matrix of dataset IDs, immutable
corpus revisions, and processed-data digests.

`uncalibrated` is the only valid status without a qualified corpus matrix and serializes with
`release_eligible: false`. `acceptance` requires exactly five distinct, pinned corpus entries and
a SHA-256 matrix digest over their canonicalized identifiers, revisions, and processed-data
digests. It may be `review_eligible`, but pure evidence always serializes with
`release_eligible: false`.
An immutable matrix manifest and approved policy are separate external requirements. Even a
schema-valid acceptance record remains calibration evidence for later reviewer policy approval,
not a release claim.

## Corpus tiers and provenance

The CI mini corpus is small, deterministic, versioned, license-compatible, and committed
or content-addressed. It catches contract, numerical, and obvious semantic regressions but
does not authorize release. The release corpus is larger, pinned by immutable revision and
digest, covers representative text and long-context cases, and records acquisition and
normalization provenance. CI and release manifests use the same schema and scoring code.

Unavailable external corpora or SSD processed datasets are explicit environmental gate
failures. Test data must not be fabricated, silently downloaded, or replaced with an
easier corpus while retaining the original gate name.

## Result schema and publication

The result document contains schema version, run identity, source commit, complete input
manifest, baseline metrics, compacted metrics, comparisons, semantic outcomes, diagnostics,
calibration-policy identity, gate decisions, timings, and structured errors. Every required
gate has an explicit `pass` or `fail`; absence is not equivalent to pass.

Publication is fail closed:

1. write the complete candidate to a temporary file on the destination filesystem;
2. flush and close it, parse it back, and validate its schema and finite-value invariants;
3. verify all mandatory gates passed and required provenance is present;
4. atomically replace the destination with the candidate;
5. on any failure, retain the last known-good result and preserve diagnostics separately.

Partial JSON, exception-path output, or an uncalibrated result is never published as the
canonical success artifact.

## Acceptance criteria

- Baseline and compacted arms score identical target tokens with identical masks and counts.
- All required loss/perplexity fields and deltas are finite and schema-valid.
- Required compaction produces positive compacted-layer and byte-accounting evidence.
- The applicable calibrated perplexity policy passes.
- Every mandatory deterministic semantic task satisfies the applicable calibrated policy.
- KL/top-k diagnostics are clearly labeled non-authoritative.
- Model, tokenizer, corpus, evaluator, runtime, and policy provenance is immutable and
  complete.
- Failure injection proves the prior canonical result survives every error path.
- Exact-text equality is not used to pass or fail lossy quantization quality.

## Assumptions, risks, and uncertainties

| Type | Statement | Required disposition |
|---|---|---|
| Assumption | The backend exposes target-token logits for teacher-forced scoring. | Verify the MLX call contract before implementation; otherwise add a typed backend port rather than scraping generated text. |
| Assumption | Baseline and compacted runs can use the same tokenization and masking. | Assert token IDs, masks, sample IDs, and counts before comparing metrics. |
| Risk | Numeric variation differs by hardware/runtime. | Calibrate across the declared support matrix and scope each policy to its applicable environment. |
| Risk | A mini corpus overfits implementation errors. | Treat CI as regression coverage only; require the pinned full release corpus for release authorization. |
| Risk | Semantic evaluators introduce nondeterminism or unsafe execution. | Prefer pure comparators; enforce the sandbox boundary and deterministic resource limits for executable tasks. |
| Risk | Plausible text conceals distributional collapse. | Make teacher-forced loss/perplexity mandatory and retain known-bad sensitivity fixtures. |
| Risk | Atomic rename is not durable across filesystems. | Create the temporary file beside the destination and validate platform durability behavior in integration tests. |
| Uncertainty | Production numeric limits are not yet calibrated for this recovered MLX path. | Fail release as `uncalibrated` until reviewed calibration artifacts produce an applicable policy. |
| Uncertainty | Full SSD processed datasets are absent locally. | Keep readiness non-green and record the exact missing dataset contract; do not weaken or counterfeit the gate. |
| Uncertainty | Rollout-derived source may omit state not printed in logs. | Reconstruct in isolation, diff against baseline, and require complete tests and independent review before adoption. |
