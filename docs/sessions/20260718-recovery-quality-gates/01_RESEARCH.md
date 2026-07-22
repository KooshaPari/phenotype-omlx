# Recovery and quality-gate research

## Evidence standard

Repository source and executable tests govern implementation behavior. External sources
inform packaging, evaluation, security, and interoperability choices but do not override
the checked-in API. Sources below were inspected or previously verified during this
session; future implementation must pin exact model, tokenizer, corpus, and dependency
revisions in each result artifact.

## Evidence matrix

| Evidence | Finding | Design consequence |
|---|---|---|
| `python/ffi/src/lib.rs:41-81` at baseline `37af3d5` | The recovered binding calls infallible `QuantizedTensor::encode_uniform`, reconstructs a tensor without complete shape/configuration metadata, and exposes raw lists. | Restore the reviewed fallible, self-describing quantization contract and test malformed metadata, unsupported bit widths, non-finite inputs, and decode-length mismatches. |
| `perf-core/turbo-quant/src/lib.rs` | The Rust crate is the quantization API source of truth; Python must bind its real public types and error behavior rather than inventing wrapper APIs. | Validate all PyO3 conversions at the boundary and convert Rust errors to typed Python exceptions without panic. |
| `scripts/e2e_real_model.py:101-112` and `scripts/e2e_real_model.py:177-234` | The recovered runner measures a separately constructed cache and generates through another cache; its comments and `key_bits` configuration do not establish same-cache compacted-prefill behavior. | Recovery must restore the reviewed single-cache sequence: prefill all but the final token, materialize, compact the same cache, and decode the unprocessed final token through that cache. |
| `scripts/e2e_real_model.py:344-347` | Baseline results are written directly to a hard-coded former checkout path without a fail-closed validation transaction. | Results must be schema-validated and atomically published only after every required gate passes. |
| `scripts/phenotype-omlx-ready:29-36` and `scripts/phenotype-omlx-ready:113-122` | Baseline readiness may skip Cargo checking when a target directory exists and imports `_perf` as a top-level module. | Readiness must run the native gates unconditionally and validate the installed wheel's canonical `omlx_research._perf` module in a fresh environment. |
| `../turboquant_plus/docs/quality-benchmarks.md:5-15` | Coherent text and cosine similarity were explicitly judged insufficient; perplexity, KL divergence, performance, and long-context retrieval were selected as quality evidence. | Exact generated-text equality is debugging evidence only. Teacher-forced loss/perplexity plus deterministic semantic acceptance form the release oracle; KL/top-k remain diagnostics. |
| `../turboquant_plus/docs/quality-benchmarks.md:128-152` | A broken implementation produced plausible text while perplexity rose from `6.121` FP16 to `165.6`. | Human-readable output cannot pass a lossy quantizer; distributional scoring is mandatory and missing/non-finite metrics fail closed. |
| `../turboquant_plus/docs/quality-benchmarks.md:248-259` | After repairing the coordinate-space error, TurboQuant perplexity was `6.194`, `+1.19%` versus FP16. | Compare compacted results to a same-run FP16 baseline and calibrate limits from repeated observed distributions rather than importing an unrelated fixed threshold. |
| Absorbed tree `perf-core/eval-harness/src/perplexity.rs:3-7` | The surviving harness computes perplexity as `exp(-mean(log probability))` and returns infinity for an empty sequence. | The recovered scorer must use teacher-forced target-token log probabilities, report token count and aggregate loss, and reject empty/non-finite results. |
| Local Codex rollout JSONL under `~/.codex/sessions/2026/07/14/` and `~/.codex/sessions/2026/07/16/` | Tool-call inputs, diffs, file dumps, and test outputs preserve reconstructable evidence for the missing uncommitted batch. | Treat rollout records as forensic inputs: reconstruct into an isolated branch, review diffs against `37af3d5`, and rerun every gate. Do not treat narrative summaries as source code. |
| Remote `wip/20260717T0516-64412e90` at `37af3d5759405833faf84fdb11ee66e9d2c20e11` | The named Airlock snapshot points to the baseline and lacks the later split FFI, readiness, E2E validation, tests, and prior session documents. | Recovery provenance must distinguish the Git baseline from reconstructed rollout content and produce a new verified snapshot after restoration. |

Paths beginning with `../turboquant_plus` are relative to the repositories root, not this
checkout. The absorbed eval harness is at
`../phenotype-registry/registry/absorbed-crates/phenotype-omlx/` from that same root.

## Official implementation references

| Primary source | Relevance |
|---|---|
| [PyO3 user guide](https://pyo3.rs/) | Canonical Rust-to-Python types, exceptions, GIL boundaries, and extension-module behavior. The implementation must use the documentation matching its pinned PyO3 version. |
| [PyO3 migration guide](https://pyo3.rs/latest/migration.html) | Required review when changing PyO3 versions; prevents silently mixing APIs from incompatible releases. |
| [maturin tutorial](https://www.maturin.rs/tutorial.html) | Canonical mixed Rust/Python packaging layout and local wheel-development workflow. |
| [maturin configuration](https://www.maturin.rs/config.html) | Canonical `pyproject.toml` module-name and Python-source configuration for installing `omlx_research._perf`. |
| [Python `os.replace`](https://docs.python.org/3/library/os.html#os.replace) | Atomic replacement primitive used only after a complete temporary result has been flushed, closed, and validated. |
| [Python `tempfile`](https://docs.python.org/3/library/tempfile.html) | Safe creation of a result temporary file in the destination filesystem before atomic replacement. |

## Previously verified application-layer references

These sources govern the broader application-layer roadmap but are not quality-oracle
substitutes:

- Kubernetes Gateway API Inference Extension specification:
  <https://gateway-api-inference-extension.sigs.k8s.io/reference/spec/>.
- Model Context Protocol authorization specification:
  <https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization>.
- Model Context Protocol task utilities:
  <https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks>.
- Agent2Agent protocol specification: <https://a2a-protocol.org/latest/specification/>.
- OpenTelemetry generative-AI semantic conventions:
  <https://opentelemetry.io/docs/specs/semconv/gen-ai/>.
- NIST Secure Software Development Framework:
  <https://csrc.nist.gov/pubs/sp/800/218/final>.
- SLSA specification: <https://slsa.dev/spec/>.
- Sigstore documentation: <https://docs.sigstore.dev/>.
- RouteLLM paper and implementation: <https://arxiv.org/abs/2406.18665> and
  <https://github.com/lm-sys/RouteLLM>.

No decision in this session relies on unverified RouteLMT, R2-Router, or AgentRFC
references.

## Unavailable evidence

No local `CHATGPT-*.md` conversation corpus is currently available in the inspected
workspace. The remote desktop corpus was not inspected because Tailscale was stopped and
this session did not authorize starting remote-access services. Neither absence is treated
as evidence for or against the approved design.
