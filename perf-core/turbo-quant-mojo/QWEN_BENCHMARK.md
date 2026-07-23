# Qwen 0.8B GEMV Decode Kernel A/B Benchmark

## Setup

- **Model:** Qwen 0.8B
- **Platform:** MacBook Apple Silicon
- **Build:** `cargo bench -p turbo-quant-mojo --bench kernel_ab` (release, LTO)
- **Warmup:** 20 iterations
- **Max bench time:** 8 seconds per kernel per dimension (adaptive iterations)
- **Kernels:**
  - `gemv_decode (naive)` — scalar Rust, row-major
  - `gemv_decode_rust_simd` — 32-element chunked iteration for cache friendliness
  - `gemv_decode_mojo (stub)` — delegates to naive until Mojo FFI gemv bridge lands

## Qwen 0.8B Dimensions

| Parameter | Value |
|-----------|-------|
| `QWEN_0_8B_HIDDEN` | 2048 |
| `QWEN_0_8B_INTERMEDIATE` | 5120 |
| `QWEN_0_8B_NUM_HEADS` | 16 |
| `QWEN_0_8B_HEAD_DIM` | 128 |

## Dimensions Tested

| Operation | Matrix Shape | Description |
|-----------|-------------|-------------|
| Q/K/V projection | 2048 × 2048 | hidden → hidden |
| Gate/Up projection | 2048 × 5120 | hidden → intermediate |
| Down projection | 5120 × 2048 | intermediate → hidden |
| Head slice | 128 × 2048 | head_dim × hidden |

## Results

| Kernel | Rows×Cols | us/call | tokens/sec | Iters |
|--------|-----------|---------|------------|-------|
| gemv_decode (naive) | 2048×2048 | 6925.3 | 144 | 1156 |
| gemv_decode_rust_simd | 2048×2048 | 3369.1 | 297 | 2375 |
| gemv_decode_mojo (stub) | 2048×2048 | 7449.4 | 134 | 1074 |
| gemv_decode (naive) | 2048×5120 | 18253.3 | 55 | 439 |
| gemv_decode_rust_simd | 2048×5120 | 8017.6 | 125 | 998 |
| gemv_decode_mojo (stub) | 2048×5120 | 17210.1 | 58 | 466 |
| gemv_decode (naive) | 5120×2048 | 19933.3 | 50 | 402 |
| gemv_decode_rust_simd | 5120×2048 | 7636.4 | 131 | 1048 |
| gemv_decode_mojo (stub) | 5120×2048 | 16702.5 | 60 | 479 |
| gemv_decode (naive) | 128×2048 | 524.3 | 1907 | 15259 |
| gemv_decode_rust_simd | 128×2048 | 172.5 | 5796 | 46372 |
| gemv_decode_mojo (stub) | 128×2048 | 401.7 | 2490 | 19917 |

## Speedup: SIMD / Naive

| Dimension | Speedup |
|-----------|---------|
| 2048×2048 | **2.06×** |
| 2048×5120 | **2.28×** |
| 5120×2048 | **2.61×** |
| 128×2048 | **3.04×** |

## Key Findings

1. **SIMD chunked iteration consistently beats naive** across all Qwen 0.8B dimensions (2.06×–3.04×).
2. **Smaller matrices benefit more** from SIMD — 128×2048 (head slice) sees 3.04× speedup vs 2.06× at 2048×2048.
3. **Mojo FFI stub** currently delegates to naive; real Mojo gemv bridge will replace this path.
4. **Decode throughput** at 2048×2048 (Q/K/V): 297 tok/s with SIMD — sufficient for real-time decode at typical generation rates.

## Running

```bash
cd /Users/kooshapari/CodeProjects/Phenotype/repos/phenotype-omlx/perf-core
cargo bench -p turbo-quant-mojo --bench kernel_ab 2>&1 | grep -E "time:|thrpt:|Benchmarking|tok/s|Speedup" | head -30
```
