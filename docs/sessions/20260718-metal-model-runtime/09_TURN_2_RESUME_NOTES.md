# 2026-07-19 — Resume notes (turn 2)

Session resumed after all P0/P1 issues and most P2 issues from the
2026-07-18 turn were resolved. This turn focused on:

1. Verifying the workspace test baseline remained green.
2. Closing remaining clippy lint debt (kernel-registry, model-plan).
3. Adding sliding-window causal attention for Qwen3-Next / Mistral.
4. Adding `AttentionKind::SlidingWindow` on the plan side so the
   model-runtime pipeline can describe Qwen3-Next long-context layers.
5. Verifying Airlock v2 is still absent and that the new doctor check
   surfaces it on every run.

## State at start of turn

| Surface | Total | Pass | Fail | Skip |
|---|---:|---:|---:|---:|
| Rust workspace | 673 | 673 | 0 | 1 (turbo-quant minmax microbench) |
| Python | 119 | 119 | 0 | 4 (3 mlx_backend prod-path + 1 happy-path helper test) |

## Work delivered this turn

### 1. clippy -D warnings clears (kernel-registry + model-plan)

Five pre-existing clippy lint errors that had been failing
`cargo clippy --workspace --all-targets -- -D warnings` for several
commits were addressed in commit `4f06629`:

- `kernel-registry/src/quality.rs`
  - `clippy::needless_borrows_for_generic_args`: drop `&` on
    `h.update(&self.canonical_bytes())`.
  - `clippy::vec_init_then_push`: replace `Vec::with_capacity(8)` +
    8× `push` with a single `vec![]` literal.
- `kernel-registry/src/record.rs`
  - `clippy::too_many_arguments`: add `#[allow(...)]` on
    `TuningRecord::from_samples` (the 9-arg signature is part of
    the public API and tests rely on positional args).
- `kernel-registry/src/selector.rs`
  - `clippy::large_enum_variant`: box the `TuningRecord` inside
    `SelectionDecision::Chosen`. The Chosen variant was 544 bytes
    vs 48 for Rejected; boxing reduces enum size to ~16 bytes.
    Auto-deref means all existing read sites (trace.rs,
    sota_operators tests) continue to work without source change.
- `model-plan/src/plan.rs`
  - `clippy::collapsible_match`: collapse the `if`-inside-`match`
    arm in `check_operator_dtype` into match guards. Behavior is
    identical for the Add/Mul and DenseMatmul/GroupedMatmul arms
    (validation only runs when the relevant condition holds;
    otherwise falls through to `_ => {}`).

### 2. Sliding-window attention (Qwen3-Next long-context)

Added `perf-core/model-kernels/src/attention/sliding_window.rs`
with `sliding_window_attention` — the canonical Mistral
sliding-window causal pattern:

  Q at position s attends to K positions in
    [max(0, s - window_size + 1), min(seq_k, s + 1))

When `window_size >= seq_k`, the output is byte-identical to
`gqa_attention` (locked by
`sliding_window_matches_gqa_when_window_is_full`).

`KernelOp::SlidingWindowAttention` (tag `"sliding_window_attention"`)
registered; total `KernelOp` tag coverage is now 24.

`kernel-registry/tests/sota_operators/attention_sliding_window.rs`
confirms the selector returns a Metal-side candidate tagged
`sliding_window_attention` for the
`(seq_q=8, q_heads=8, kv_heads=2, head_dim=64, group_size=4,
window_size=4)` shape signature.

`omlx-research doctor` `model_kernels_operator_coverage` threshold
bumped from 22 to 24 (covers DeltaNetBatched + SlidingWindowAttention).

### 3. Plan-side wiring: `AttentionKind::SlidingWindow`

Added the new `AttentionKind::SlidingWindow { window_size }`
variant in `model-plan/src/attention.rs` so a `ModelPlan` can
describe Qwen3-Next long-context layers. Serializes with tag
`"sliding_window"` and round-trips through serde. The
`tag_for_each_variant` and `serde_round_trip` tests were updated
to cover the new variant.

The metal-runtime compile pipeline (`metal-runtime/src/compile.rs`)
already iterates over `plan.operators` and uses `op.kind.tag()`,
so this addition is the canonical integration point: any future
selector that walks `OperatorPlan::attention` will see the
SlidingWindow variant without further plumbing.

### 4. Airlock v2 status — STILL NOT RESOLVED

`repos/.airlock/bin/airlock-v2.py` is **NOT** present on this
machine. Only an unrelated Homebrew `airlock` (keychain tool,
v0.1.38) is on PATH.

The new `omlx-research doctor` subcommand's `airlock_v2_installed`
check (added in the previous turn, commit `9f5384d`) surfaces the
gap explicitly:

```
[WARN] airlock_v2_installed
        airlock-v2 binary on PATH
        NOT INSTALLED — airlock-v2 is a known unresolved P2 from the
        session; documenting it explicitly here so doctor users see
        the gap. Install once the upstream crate ships.
```

CI must install or vendor the project's Airlock v2 before snapshots
can run. Snapshots and contract gating remain blocked on this.

## Test status (post-turn)

| Surface | Total | Pass | Fail | Skip |
|---|---:|---:|---:|---:|
| Rust workspace | 686 | 686 | 0 | 1 |
| Python | 123 | 119 | 0 | 4 |

+13 Rust tests vs prior turn (11 sliding-window + 2 model-plan).

## Commit graph (newest first)

```
4f06629 fix(clippy): clear 5 pre-existing -D warnings lint errors
6f7e80c feat(model-plan): AttentionKind::SlidingWindow — connect sliding-window attention to the plan surface
eeb9d55 feat(model-kernels): sliding-window causal GQA attention (Qwen3-Next)
4f06629 fix(clippy): clear 5 pre-existing -D warnings lint errors   ← previous tip
48e0e31 docs(sessions): record this turn's four resolutions (mod_routing split, batched DeltaNet, mlx_backend skip, doctor subcommand)
```

## Forward-looking priorities

1. **Airlock v2** — still blocked on tooling; the new `doctor`
   check surfaces the gap but resolution needs the upstream crate.
2. **Clippy `-D warnings`** — kernel-registry and model-plan are
   clean as of this turn. Remaining 5 crates still emit ~63 lint
   errors (eval-harness upper-case acronyms GPQA/MMLU/HELM;
   tree-attention manual indexed loops; turbo-quant `div_ceil` +
   default-then-assign; native-abi build-script empty `writeln!`;
   spec-decode tests default-then-assign; model-kernels ~30
   loop-var-only-used-to-index; fleet-proto-zeromq unused import).
   Filed for the next lint-clear sweep.
3. **Selector plumbing for sliding-window** — the new
   `AttentionKind::SlidingWindow` variant is wired into the plan,
   but no candidate family in `kernel-registry` dispatches on it
   yet. A follow-up should add a `KernelKey` shape that includes
   `window_size` and register Metal-side candidates for it.
4. **GPU ablation benchmarks** — once #3 lands, a
   `deltanet_batched` shape-bucketed regression test (mirroring
   `dispatch_buckets`) would be a natural follow-up.