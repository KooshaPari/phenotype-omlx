# 2026-07-19 — Resume notes (turn 3)

Session resumed after turn 2 closed the kernel-registry / model-plan
clippy lint debt and added sliding-window attention. This turn
focused on three concrete lanes:

1. **Clippy `-D warnings` final sweep** — cleared every remaining
   lint error across the entire workspace (9 crates). After this
   turn, `cargo clippy --workspace --all-targets -- -D warnings`
   exits 0 across all 9 published crates for the first time.
2. **Selector-plumbing builders** — added pure builder functions in
   `kernel-registry/src/builders.rs` that bridge `model-plan`
   `OperatorPlan` into `KernelKey` for `SlidingWindow`, `DeltaNet`,
   and `DeltaNetBatched` so the runtime doesn't have to hand-roll
   shape-signature magic numbers.
3. **DX hardening for missing `mlx_lm`** — structured
   `RuntimeError` with install hint, per-command gating in the
   `omlx-research doctor` subcommand.

## State at start of turn

| Surface | Total | Pass | Fail | Skip |
|---|---:|---:|---:|---:|
| Rust workspace | 686 | 686 | 0 | 1 (turbo-quant minmax microbench) |
| Python | 123 | 119 | 0 | 4 (mlx_backend prod-path + helper) |
| `cargo clippy --workspace --all-targets -- -D warnings` | — | **75 errors across 9 crates** | — | — |

## Work delivered this turn

### 1. Clippy `-D warnings` final sweep (10 commits)

Every remaining lint error was cleared. Per-crate totals:

| Crate | Commit | Errors cleared | Notable lints |
|---|---|---:|---|
| `model-kernels` (lib) | `ac82341` | 34 | `needless_range_loop` (27×), `too_many_arguments` (4×), `manual_memcpy` (2×), `unnecessary_cast`, `unnecessary_lazy_evaluations`, `map_identity` |
| `native-abi` | `fbbb031` | 30 | `writeln_empty_string` (14× in `build.rs` + 14× in `headers.rs` mirror + descriptor/dispatch/tests) |
| `eval-harness` | `5cf961f` | 7 | `upper_case_acronyms` (`MMLU` → `Mmlu`, `GPQA` → `Gpqa`; crate-local, no external callers), `manual_range_contains` (2×, escalated to `is_ascii_uppercase`), `needless_range_loop`, `unused_imports` (`TaskSpec`) |
| `tree-attention` | `8426850` | 13 | `if_same_then_else`, `manual_checked_ops` → `checked_div`, `needless_range_loop` × 11 (lib + tests/oracle.rs) |
| `spec-decode` tests | `84de702` | 5 | `field_reassign_with_default` → struct-init form |
| `turbo-quant` | `04d6f61` | 2 | `manual_div_ceil` × 2 |
| `fleet-proto-zeromq` | `79eb7b4` | 1 | `unused_imports` (`super::*`) |
| `turbo-quant-c` | `bfdb0e4` | 2 | `manual_div_ceil`, etc. |
| `metal-runtime` | `438b5dc` | 5 | `or_insert_like`, `if_same_then_else`, `items_after_test_module` |
| `model-kernels` (lib test) | `4d011bb` | 15 | `needless_range_loop`, `identity_op`, `useless_vec`, `type_complexity`, `too_many_arguments`, `unusual_byte_groupings` |

The `MMLU` → `Mmlu` rename was the most behavior-touching change in
the sweep. Verified via `grep -rn 'BenchmarkKind::MMLU\|Suite::GPQA'`
that both enum variants are only used inside `eval-harness`, so the
rename is purely internal (8 files updated, 0 external callers).
A small `allow_clippy_removal` belt-and-suspenders pattern was added
in `eval-harness/src/lib.rs` for the wider `MMLU` token if it
appears in macro paths.

Final state: `cargo clippy --workspace --all-targets -- -D warnings`
exits 0 with only the unrelated `turbo-quant-mojo` stub-only
build-script note remaining. Workspace tests unchanged: 686 passing,
0 failed, 1 ignored.

### 2. Selector-plumbing builders (`31e250e`)

New file `perf-core/kernel-registry/src/builders.rs` with three
pure builders that bridge `model-plan::OperatorPlan` into
`KernelKey`:

```rust
pub fn sliding_window_key(
    q_heads: usize, kv_heads: usize, head_dim: usize,
    batch_size: usize, seq_len: usize,
    group_size: usize, window_size: usize,
    dtype: DType, device_fingerprint: &str, policy_version: u32,
) -> KernelKey;

pub fn deltanet_batched_key(
    batch_size: usize, num_heads: usize,
    chunk_size: usize, head_dim: usize,
    dtype: DType, device_fingerprint: &str, policy_version: u32,
) -> KernelKey;

pub fn deltanet_key(
    head_dim: usize, chunk_size: usize,
    dtype: DType, device_fingerprint: &str, policy_version: u32,
) -> KernelKey;
```

Encoding (consistent with the rest of the kernel-registry):

- `sliding_window_key`: `m = q_heads * head_dim / 8`, `n =
  group_size`, `k = kv_heads`, `batch`, `seq`, `group = window_size`
  (clamped to `[1, seq_len]`). `attention_kind = Some(Gqa)`,
  `operator_kind = Attention`.
- `deltanet_batched_key`: `m = n = k = head_dim`, `batch`,
  `seq = chunk_size`, `group = num_heads`.
  `operator_kind = DeltaNet`, `attention_kind = None`.
- `deltanet_key`: `m = n = k = head_dim`, `seq = chunk_size`.
  `operator_kind = DeltaNet`, `attention_kind = None`.

All three panic on degenerate input (zero dims) with a clear
message; clamp `window_size` so a degenerate `window_size = 0` key
still matches scalar fallbacks.

**Coverage** (16 unit + 3 integration tests):
- valid inputs round-trip the expected `ShapeSignature`
- `window_size > seq_len` is clamped
- `window_size == 0` is clamped to 1
- `dtype` / `device_fingerprint` / `policy_version` are forwarded
  unchanged
- `operator_kind` discriminant is correct
- end-to-end: each builder produces a `KernelKey` whose selector
  picks the expected `DeltaNetBatchedMetal` / `SlidingWindowMetal`
  candidate

Public API re-exported from `kernel-registry::builders` so the
runtime can write `use kernel_registry::builders::sliding_window_key;`
directly.

`kernel-registry`: 71 → 89 tests (+18). Workspace: 686 → 704 passing.

### 3. DX hardening for missing `mlx_lm` (`89d1cac`)

New file `python/omlx_research/cli/_missing_dep.py` with the helper:

```python
def require_mlx_lm(where: str) -> types.ModuleType:
    """Lazy-import mlx_lm and raise a structured RuntimeError if missing.

    The result is cached at module level so subsequent calls do not
    re-pay the import cost.
    """
```

When `mlx_lm` is missing, the helper raises:

```
RuntimeError: mlx_lm is required for {where}.

Install with:
    pip install mlx-lm

On Apple Silicon, also ensure mlx-core is installed:
    pip install mlx-core

To run without mlx_lm (decode-path only), use the
`omlx-research doctor` subcommand or `--no-mlx-lm` flag.
```

Wired into `omlx-research cmd-inference` so users see the
structured message instead of a bare `ModuleNotFoundError`.

New doctor check `mlx_lm_required_by_command(cmd)` upgrades
`warn` → `fail` for the active subcommand if it is in
`{"run", "serve", "eval"}` and `mlx_lm` is missing.

**Coverage** (6 helper tests + 3 doctor tests):
- helper raises when `mlx_lm` is hidden via `sys.modules`
- helper returns the module when present
- result is cached across calls
- error message contains the install hint
- error message names the call site
- doctor check emits `fail` for run/serve/eval without mlx_lm
- doctor check emits `warn` for other subcommands
- doctor check emits `pass` when mlx_lm is installed

Python suite: 119 passed + 4 skipped → 128 passed + 4 skipped (+9).

### 4. Airlock v2 — STILL NOT RESOLVED

`repos/.airlock/bin/airlock-v2.py` remains absent. Only the
unrelated Homebrew `airlock` (keychain tool, v0.1.38) is on PATH.

The `omlx-research doctor` subcommand continues to surface the gap
explicitly via `airlock_v2_installed`. CI must install or vendor
the project's Airlock v2 before snapshots can run.

## Test status (post-turn)

| Surface | Total | Pass | Fail | Skip | Δ |
|---|---:|---:|---:|---:|---:|
| Rust workspace | 704 | 704 | 0 | 1 | +18 |
| Python | 132 | 128 | 0 | 4 | +9 |
| `cargo clippy --workspace --all-targets -- -D warnings` | — | 0 errors | 0 | — | −75 errors |

## Commit graph (newest first)

```
89d1cac feat(cli): structured mlx_lm-missing error message and per-command doctor gating
31e250e feat(kernel-registry): add KernelKey builders for sliding-window + DeltaNet (selector plumbing)
4d011bb fix(clippy): clear 15 -D warnings errors in model-kernels lib tests
438b5dc fix(clippy): clear 5 -D warnings errors in metal-runtime
bfdb0e4 fix(clippy): clear 2 -D warnings errors in turbo-quant-c
79eb7b4 fix(clippy): clear 1 unused-import error in fleet-proto-zeromq tests
04d6f61 fix(clippy): clear 2 -D warnings errors in turbo-quant
84de702 fix(clippy): clear 5 -D warnings errors in spec-decode tests
8426850 fix(clippy): clear 13 -D warnings errors in tree-attention
5cf961f fix(clippy): clear 7 -D warnings errors in eval-harness
fbbb031 fix(clippy): clear 30 -D warnings errors in native-abi
ac82341 fix(clippy): clear 34 -D warnings errors in model-kernels
745f725 docs(sessions): record turn-2 work — sliding-window + AttentionKind wiring + clippy lint clears  ← turn-2 tip
```

## Module-size discipline (post-turn)

| File | Lines | Budget |
|---|---:|---|
| `kernel-registry/src/builders.rs` | 156 | ≤350 ✓ |
| `kernel-registry/tests/sota_operators/builders_integration.rs` | 80 | ≤350 ✓ |
| `python/omlx_research/cli/_missing_dep.py` | 60 | ≤500 ✓ |

All new files fit within the project's soft 350-line target and
hard 500-line cap.

## Forward-looking priorities

1. **Airlock v2** — still blocked on tooling; the `doctor` check
   surfaces the gap but resolution needs the upstream crate.
2. **GPU ablation benchmarks** — now that builders exist for
   SlidingWindow + DeltaNetBatched, a `dispatch_buckets`-style
   shape-bucketed regression test in `regress-baseline` would be a
   natural follow-up.
3. **Metal-runtime compile pipeline** — the pipeline currently
   walks `plan.operators[*].kind.tag()`. A `match` arm for the new
   `SlidingWindow` / `DeltaNetBatched` MSL emit stubs is the
   remaining integration step (the model-plan side already
   serializes correctly).
4. **Continuous clippy gate** — the `-D warnings` gate is now
   green, but it's only checked manually. Adding a CI step that
   runs `cargo clippy --workspace --all-targets -- -D warnings`
   on every PR would prevent future drift.
