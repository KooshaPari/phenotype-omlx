# GEMV Decode Benchmark Results

## Release Mode (cargo test --release)

| Size | Naive (μs) | SIMD (μs) | Speedup |
|------|-----------|-----------|---------|
| 64×128 | 13.4 | 5.3 | 2.53× |
| 128×256 | 264.4 | 409.5 | 0.65× (SIMD slower) |
| 256×512 | 2778.9 | 1488.5 | 1.87× |

## Notes
- MacBook Apple Silicon
- Release mode with LTO
- Single-threaded (no rayon)
- SIMD variant uses cache-friendly chunked iteration
- 500 iterations per kernel, 3 warm-up iterations
- 128×256 shows SIMD regression — likely cache pressure or chunk alignment issue at that size
- 64×128 and 256×512 show expected SIMD wins (2.5× and 1.9× respectively)
