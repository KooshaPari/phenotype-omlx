# Recovery quality gates implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Reconstruct a production-correct Rust/PyO3 quantization interface, readiness verifier, and loss-aware real-model quality gate.

**Architecture:** Rust validates and encodes self-describing tensors; PyO3 exposes that contract through the canonical wheel module; Python owns readiness, teacher-forced scoring, semantic acceptance, and atomic result publication. Release eligibility comes only from a calibrated policy.

**Tech Stack:** Rust 2021, `thiserror`, serde, PyO3, maturin, Python, pytest, MLX/MLX-LM.

---

### Task 1: Fallible self-describing Rust tensors

**Files:** Modify `perf-core/turbo-quant/src/lib.rs`; create `perf-core/turbo-quant/tests/correctness.rs`.

- [ ] **Step 1: Write the failing tests.**

```rust
assert!(QuantizedTensor::encode_uniform(&[1.0], 0, 64).is_err());
assert!(QuantizedTensor::try_from_parts(vec![2], 4, 64, vec![], vec![], vec![]).is_err());
```

- [ ] **Step 2: Verify failure.** Run `cargo test -p turbo-quant --test correctness`; expect missing fallible APIs.

- [ ] **Step 3: Implement.** Add `QuantError`, stored `bits` and `group_size`, `try_from_parts`, and `Result`-returning encode/decode. Reject non-finite values, invalid bits/group size, packed-length mismatches, scale/zero mismatches, and wrong output length. Remove the inferred-bits heuristic.

- [ ] **Step 4: Verify pass.** Run `cargo test -p turbo-quant --test correctness && cargo test -p turbo-quant`; expect PASS for 2/3/4-bit, constants, malformed metadata, and deterministic round trips.

- [ ] **Step 5: Commit.** `git add perf-core/turbo-quant && git commit -m "feat(turbo-quant): validate self-describing tensors"`.

### Task 2: Canonical PyO3 contract

**Files:** Modify `python/ffi/src/lib.rs`; create `python/ffi/src/quantization.rs`; create `python/tests/test_turbo_quant_ffi.py`; modify `python/pyproject.toml`.

- [ ] **Step 1: Write failing tests.**

```python
value = perf.turbo_quant_encode([0.25] * 128, group_size=32, bits=4)
assert set(value) == {"shape", "bits", "group_size", "packed", "scales", "zeros"}
assert isinstance(value["packed"], bytes)
with pytest.raises(ValueError):
    perf.turbo_quant_decode([2], 4, 64, b"", [], [])
```

- [ ] **Step 2: Verify failure.** Run `python -m pytest python/tests/test_turbo_quant_ffi.py -q`; expect baseline metadata inference/list-payload failure.

- [ ] **Step 3: Implement.** Map `QuantError` to `PyValueError`; release the GIL for Rust work; use `bytes` for packed payload; expose only complete metadata; register `omlx_research._perf` and remove top-level compatibility import.

- [ ] **Step 4: Verify pass.** Run `cargo check --manifest-path python/ffi/Cargo.toml && python -m pytest python/tests/test_turbo_quant_ffi.py -q`; expect PASS and no Rust panic.

- [ ] **Step 5: Commit.** `git add python/ffi python/tests/test_turbo_quant_ffi.py python/pyproject.toml && git commit -m "feat(ffi): validate quantization payloads"`.

### Task 3: Verify the installed release wheel

**Files:** Modify `python/pyproject.toml`; create `python/tests/test_wheel_install.py`.

- [ ] **Step 1: Write the failing test.**

```python
wheel = build_release_wheel(tmp_path)
assert install_and_run_isolated(wheel, "from omlx_research import _perf").returncode == 0
```

- [ ] **Step 2: Verify failure.** Run `python -m pytest python/tests/test_wheel_install.py -q`; expect failure before module-name/package alignment.

- [ ] **Step 3: Implement.** Configure maturin for `omlx_research._perf`, package Python sources once, and test in a fresh venv without source-tree `PYTHONPATH`.

- [ ] **Step 4: Verify pass.** Run `maturin build --release --manifest-path python/ffi/Cargo.toml && python -m pytest python/tests/test_wheel_install.py -q`; expect PASS.

- [ ] **Step 5: Commit.** `git add python/pyproject.toml python/tests/test_wheel_install.py && git commit -m "test(packaging): verify FFI wheel"`.

### Task 4: Replace readiness shortcuts

**Files:** Create `scripts/readiness_check.py`; modify `scripts/phenotype-omlx-ready`; create `tests/test_readiness.py`.

- [ ] **Step 1: Write failing tests.**

```python
assert run_readiness(target_exists=True).cargo_checked is True
assert run_readiness(module_name="_perf").exit_code == 3
```

- [ ] **Step 2: Verify failure.** Run `python -m pytest tests/test_readiness.py -q`; expect target-directory skip and top-level import failures.

- [ ] **Step 3: Implement.** Always run `cargo check --workspace`; build a release wheel; install it freshly; import `omlx_research._perf`; run one validated round trip; report compiler, wheel, environment, and dataset failures with distinct stable exit codes.

- [ ] **Step 4: Verify pass.** Run `python -m pytest tests/test_readiness.py -q && bash scripts/phenotype-omlx-ready`; unit tests pass and live readiness reports factual external failures.

- [ ] **Step 5: Commit.** `git add scripts/readiness_check.py scripts/phenotype-omlx-ready tests/test_readiness.py && git commit -m "feat(readiness): validate fresh wheel"`.

### Task 5: Teacher-forced loss and semantic gates

**Files:** Create `scripts/e2e_validation.py`; create `tests/test_e2e_validation.py`; modify `scripts/e2e_real_model.py`.

- [ ] **Step 1: Write failing tests.**

```python
with pytest.raises(ValidationError, match="token"):
    compare_scores(baseline=[1], compacted=[2])
assert publish_release_result(tmp_path, calibration=None).status == "uncalibrated"
```

- [ ] **Step 2: Verify failure.** Run `python -m pytest tests/test_e2e_validation.py -q`; expect absent teacher-forced evaluator.

- [ ] **Step 3: Implement.** Score identical observed token IDs for FP16 and compacted caches; record finite NLL, mean loss, PPL, and deltas. Add pure deterministic fixed-answer and retrieval comparators. Treat KL/top-k as diagnostics only. Forbid host execution, network, secrets, and model tools; executable evaluators require a separate no-network sandbox adapter.

- [ ] **Step 4: Verify pass.** Run `python -m pytest tests/test_e2e_validation.py -q`; expect exact text not to decide lossy quality and uncalibrated release to fail closed.

- [ ] **Step 5: Commit.** `git add scripts/e2e_validation.py scripts/e2e_real_model.py tests/test_e2e_validation.py && git commit -m "feat(e2e): add loss-aware quality gates"`.

### Task 6: Atomic results and calibration evidence

**Files:** Modify `scripts/e2e_real_model.py`; create `tests/test_e2e_results.py`; create `research/calibration/README.md`.

- [ ] **Step 1: Write failing test.**

```python
result.write_text('{"status":"known-good"}')
with pytest.raises(GateFailure): publish_candidate(result, invalid_candidate())
assert json.loads(result.read_text())["status"] == "known-good"
```

- [ ] **Step 2: Verify failure.** Run `python -m pytest tests/test_e2e_results.py -q`; expect direct canonical-file writes.

- [ ] **Step 3: Implement.** Write beside destination, flush/close, parse/schema-validate, require all gates and provenance, then `os.replace`. Keep failed diagnostics separate. Calibration records evidence but cannot yield a release pass without an applicable approved policy.

- [ ] **Step 4: Verify pass.** Run `python -m pytest tests/test_e2e_results.py tests/test_e2e_validation.py -q && python scripts/e2e_real_model.py --tier calibration`; expect tests PASS and a factual calibration outcome.

- [ ] **Step 5: Commit.** `git add scripts/e2e_real_model.py tests/test_e2e_results.py research/calibration/README.md && git commit -m "feat(results): publish evidence atomically"`.

### Task 7: Final verification and provenance

**Files:** Create `docs/sessions/20260718-recovery-quality-gates/05_KNOWN_ISSUES.md`; create `docs/sessions/20260718-recovery-quality-gates/06_TESTING_STRATEGY.md`.

- [ ] **Step 1: Record external gates.** State that missing SSD corpora and absent calibrated release policy remain explicit release failures.

- [ ] **Step 2: Verify all local gates.** Run `cargo test --workspace && cargo check --manifest-path python/ffi/Cargo.toml && python -m pytest python/tests tests/test_readiness.py tests/test_e2e_validation.py tests/test_e2e_results.py -q && git diff --check && airlock status`; expect executable local checks PASS and external release gates reported honestly.

- [ ] **Step 3: Check provenance.** Verify the child repository tracks `upstream/*`, Airlock retains human push approval, and `/Users/kooshapari/CodeProjects/Phenotype/repos` is non-Git.

- [ ] **Step 4: Commit.** `git add docs/sessions/20260718-recovery-quality-gates && git commit -m "docs: record quality gate verification"`.
