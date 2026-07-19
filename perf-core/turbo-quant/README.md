# turbo-quant

CPU-side TurboQuant encode/decode used by `perf-core` and the Python FFI.

## FR-2: AArch64 NEON min/max (WP03)

Group-scale computation in `encode_uniform` calls `minmax::min_max`:

| Target | Implementation |
| --- | --- |
| `aarch64` | Explicit NEON intrinsics in `src/minmax.rs` (`vminq_f32` / `vmaxq_f32`) plus scalar tail |
| Other | `scalar_min_max` portable fallback |

The scalar path is the test oracle: dispatched results must match on every supported length.

### Local verification (arm64 gate)

```bash
cd perf-core
cargo test -p turbo-quant
```

On Apple Silicon / other `aarch64` hosts, `neon_handles_unaligned_slice_and_scalar_tail` exercises unaligned sub-slices and tail lengths. On x86_64 and other targets that test is cfg-gated out; parity is still covered by `scalar_and_dispatched_results_match`.

Optional microbench (ignored by default):

```bash
cargo test -p turbo-quant microbench_scalar_vs_neon_min_max -- --ignored --nocapture
```

No dedicated GitHub Actions workflow is required for WP03 — billing constraints favor this local gate on the development host.
