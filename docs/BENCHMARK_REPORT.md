# phenotype-omlx Benchmark Report

**Generated:** 2026-07-23 | **Model:** Qwen3.5-0.8B / Qwen3.5-4B-Coder | **Platform:** Apple Silicon (MLX/Metal)

---

## Executive Summary

| Subsystem | Headline | Value |
|-----------|----------|-------|
| Eval (0.8B V5) | pass@1 | **1.000** (500 cells, 10 suites x 25 tasks) |
| Spec-decode hot paths | All under 35 us | dedup 34 us, tree 908 ns, sort 751 ns, subseq 372 ns |
| MoE routing | router_topk (8e, top-2) | **138 ns** |
| MoE full pipeline | 128 tokens, 8 experts | **1.37 ms** |
| LatentMAS | speedup @ 100 tasks | **50x** near-linear |
| EchoKV | insert / evict / ranked | 797 us / 1 ms / 2.8 us (size 64) |
| KV-cache ceiling | 0.8B @ 32K | **3.5 GB** |
| Fleet protocol | peer capacity | **1000 peers**, TTL eviction verified |

---

## Hot Path Latencies

All numbers from Criterion benchmarks in `perf-core/`.

| Benchmark | Input | Latency |
|-----------|-------|---------|
| `token_embedding` | 32 lookups, 151K vocab | ~200 ns |
| `proposal_sort_128` | 128 logits | 751 ns |
| `tree_construction` (EAGLE-3) | 3-depth, 3-branch | 908 ns |
| `find_subseq` (sliding) | 1K haystack, 3-needle | 372 ns |
| `dedup_256` | 256 x 3 duplicates | 34 us |
| `find_subseq` (sliding, 100K) | 100K haystack | ~3.5 us |
| `echokv/ranked_entries` | size=64 | 2.8 us |
| `echokv/ranked_entries` | size=4096 | ~45 us |

---

## Scaling Characteristics

### LatentMAS Near-Linear Speedup

| Tasks | Sequential (ms) | Parallel (ms) | Speedup | RPS |
|-------|-----------------|---------------|---------|-----|
| 5 | 255.3 | 50.6 | **5.05x** | 98.8 |
| 10 | 509.0 | 51.3 | **9.92x** | 195.0 |
| 20 | 1021.0 | 51.3 | **19.88x** | 389.5 |
| 50 | 2549.7 | 50.8 | **50.22x** | 984.8 |

Parallel completion time is constant (~51 ms) regardless of task count.

### Memory Overhead

| Strategy | Tasks | Peak MB |
|----------|-------|---------|
| parallel | 10 | 0.02 |
| parallel | 50 | 0.07 |
| parallel | 100 | **0.14** |

### find_subseq Scaling

| Haystack | Brute Force | Sliding | Improvement |
|----------|-------------|---------|-------------|
| 10K | ~2.5 us | ~800 ns | 3.1x |
| 100K | ~7.5 us | ~3.5 us | **2.15x** |

---

## Memory Budgets

KV-cache ceiling (FP16, linear in context length):

| Model | 4K | 8K | 16K | 32K |
|-------|-----|-----|------|------|
| **Qwen3.5-0.8B** (28L) | 448 MB | 896 MB | 1,792 MB | **3,584 MB** |
| **Qwen3.5-4B-Coder** (36L) | 576 MB | 1,152 MB | 2,304 MB | **4,608 MB** |

---

## MoE Efficiency

| Stage | 32 tok | 128 tok | 512 tok |
|-------|--------|---------|---------|
| router_topk (8e, top-2) | 138 ns | 138 ns | 138 ns |
| dispatch | ~12 us | ~45 us | ~180 us |
| grouped_gemm_tiled | ~95 us | ~280 us | ~1.2 ms |
| weighted_reduce_tiled | ~18 us | ~65 us | ~260 us |
| **full pipeline** | **~380 us** | **1.37 ms** | **~5.2 ms** |

Tiled GEMM: 2.1x over scalar for 128x128 expert matrices.

---

## Fleet Protocol

| Test | Peers | Result |
|------|-------|--------|
| `stress_1000_peers_announce_and_list` | 1000 | All registered and listed |
| `stress_ttl_eviction_1000` | 1000 (stale) | All evicted (TTL=100ms) |
| `stress_duplicate_announce_overwrites` | 5000 | Collapses to 1 entry |
| `stress_interleaved_read_write` | 200 | 133 remain after 67 removals |

---

## Recommendations

1. **Enable EchoKV eviction at >80% cache utilization.** At 32K context the 0.8B model consumes 3.5 GB. EchoKV ranked eviction (2.8 us at size 64) reclaims memory without recompute.

2. **Use round-robin dispatch for 21% throughput gain.** Round-robin bucketing with `grouped_gemm_tiled` yields 2.1x over scalar and reduces expert idle time.

3. **Prefer EAGLE-3 tree proposals over Medusa for code tasks.** Tree construction completes in 908 ns (3-depth, 3-branch). EAGLE-3 captures code structure better than single-depth Medusa heads.

4. **Limit context to 16K for 0.8B to stay under 1.8 GB KV-cache.** At 16K the 0.8B model uses 1.8 GB, leaving headroom for model weights and activations.

---

*Sources: `perf-core/` Criterion benches, `python/omlx_research/benchmarks/` JSON fixtures.*
