# Recovery quality-gate DAG and work breakdown

## Forward DAG

```text
WP0 forensic inventory
 └─> WP1 artifact reconstruction
      └─> WP2 Rust quantization contract
           └─> WP3 PyO3 package contract
                └─> WP4 focused native and wheel tests
                     ├─> WP5 readiness CI/release split
                     └─> WP6 teacher-forced metrics evaluator
                          └─> WP7 deterministic semantic evaluator
                               └─> WP8 fail-closed atomic publication
                                    └─> WP9 real-model calibration
                                         └─> WP10 full verification and snapshot
```

WP5 and WP6 may proceed in parallel after WP4. WP7 depends on the shared manifest and
result types established by WP6. WP8 integrates the completed metric, semantic, and
readiness decisions. WP9 cannot establish policy until all evaluators and publication
invariants are stable.

## Work packages

| WP | Deliverable | Depends on | Completion evidence |
|---|---|---|---|
| 0 | Immutable inventory of Git baseline, rollout JSONL evidence, missing artifacts, and files explicitly excluded from replay | None | Baseline SHA, rollout paths/records, and restore manifest agree; no repository-policy file is selected |
| 1 | Reviewed reconstruction of split FFI, Rust correctness changes, readiness helper/tests, E2E validator/tests, and canonical session context in an isolated branch | WP0 | Reconstructed diff is attributable to evidence; `git diff --check` passes; unrelated absorbed-tree changes are absent |
| 2 | Fallible, self-describing Rust quantized-tensor contract with validated bit widths, group sizes, lengths, metadata, finite values, and decode invariants | WP1 | Unit, adversarial, and property tests pass without panic; Cargo check, test, clippy, and format gates pass |
| 3 | Mixed maturin package exposing the Rust extension as `omlx_research._perf`, with typed Python exceptions and one canonical tensor schema | WP2 | Release wheel installs into a fresh environment; canonical import and encode/decode round trip pass; top-level `_perf` is not the supported contract |
| 4 | Focused regression suite covering Rust/Python round trips, malformed payloads, wheel provenance, same-cache compaction truthfulness, and zero-compaction rejection | WP3 | Focused test suites pass from clean processes; mutation of each required invariant produces the expected failure |
| 5 | Readiness orchestration split into deterministic CI checks and explicit release checks, including external SSD dataset provenance | WP4 | CI readiness passes only its declared fixture tier; release readiness fails clearly when any processed SSD dataset or release corpus is absent |
| 6 | Teacher-forced scorer producing baseline and compacted NLL, mean loss, perplexity, deltas, ratios, token/sample counts, plus optional KL/top-k diagnostics | WP4 | Identical target-token assertions pass; empty/non-finite/mismatched results fail; known-bad logits are detected |
| 7 | Versioned deterministic semantic suite with pure comparators and a no-secret/no-network sandbox boundary for executable tasks | WP6 | Fixed-answer and long-context retrieval tests pass; timeout, malformed output, network attempt, and sandbox startup failure fail closed |
| 8 | Versioned result schema and same-filesystem temporary-write, reparse, validate, flush, and atomic-replace publisher | WP5, WP6, WP7 | Failure injection at every publication stage preserves the previous canonical artifact; only complete passing results replace it |
| 9 | Repeated same-run FP16 controls and compacted candidates over pinned release corpus/model/runtime matrix; reviewed calibration policy | WP8 | Signed/reviewed policy records evidence digests, applicable matrix, limits, variation, known-bad sensitivity, and recalibration triggers |
| 10 | Full native, Python, wheel, readiness, semantic, real-model, security, file-size, and result-provenance verification followed by Airlock snapshot | WP9 | Every required command is green, release gate uses an applicable policy, source tree is reviewed, and remote snapshot contains the verified commit |

## Critical path

```text
WP0 -> WP1 -> WP2 -> WP3 -> WP4 -> WP6 -> WP7 -> WP8 -> WP9 -> WP10
```

The critical path is quality-policy constrained: implementation can produce measurements
before WP9, but release authorization cannot proceed while the policy is `uncalibrated`.
The external SSD processed-dataset gap may independently hold the WP5 release lane and
therefore WP8/WP10 even when evaluator development is complete.

## Verification gates

| Gate | Required evidence | Failure disposition |
|---|---|---|
| Recovery provenance | Exact baseline, rollout record identifiers, reviewed reconstructed diff | Stop integration; preserve forensic inputs |
| Rust safety | Fallible APIs, adversarial/property tests, no panic on external payloads | Reject PyO3 integration |
| Wheel identity | Fresh environment installs built wheel and imports `omlx_research._perf` | Reject readiness success |
| Same-cache truth | Prefill, materialize, compact, and decode occur on the same cache; positive transitions and bytes are observed | Reject real-model result |
| Metric integrity | Same tokens/counts; finite NLL/loss/PPL; complete baseline-relative fields | Fail evaluation |
| Semantic safety | Deterministic fixture/evaluator identity and enforced sandbox policy | Fail affected task and evaluation |
| Calibration | Applicable reviewed policy and evidence digests | Report `uncalibrated`; fail release |
| Dataset provenance | CI fixture digest or pinned release corpus plus required SSD processed datasets | Fail the corresponding tier |
| Publication | Complete schema, all mandatory gates pass, atomic replacement succeeds | Preserve last known-good artifact |
| Final integration | Full commands green and Airlock snapshot points to verified commit | Do not claim completion |

## Execution ownership

- Recovery worker owns WP0-WP1 and hands an evidence manifest to reviewers.
- Rust/FFI worker owns WP2-WP4; an independent reviewer validates API and panic safety.
- Readiness worker owns WP5 without weakening release requirements.
- Evaluation worker owns WP6-WP8; security review owns the sandbox boundary.
- Benchmark worker owns WP9 on the pinned matrix; reviewers approve the policy.
- Manager integrates WP10, verifies evidence, and publishes the cockpit/snapshot status.
