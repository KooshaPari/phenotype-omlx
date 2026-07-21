"""
phenotype-omlx — Production TurboQuant+ benchmark.

Measures:
  1. Rust SIMD (`_perf.turbo_quant_encode`) vs Python reference (`turboquant.TurboQuant`)
     across shapes (128/512/2048/8192) × bits (2/3/4)
  2. End-to-end MLX inference with TurboQuant+ KV cache vs FP16 baseline
     on a real Qwen2.5 model

Usage:
    python3 scripts/perf_turboquant.py
    python3 scripts/perf_turboquant.py --lengths 1024 4096 16384
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("HF_HOME", "/Users/kooshapari/.cache/huggingface")

import argparse
import random
import time

import numpy as np


def bench_rust_vs_python(seed: int = 42):
    """Step 1 — Rust SIMD vs Python reference TurboQuant encode/decode."""
    print("=" * 64)
    print(" STEP 1 — Rust SIMD vs Python reference TurboQuant encode/decode")
    print("=" * 64)
    random.seed(seed)

    try:
        import _perf
    except ImportError:
        print("  ❌ _perf not installed — run: maturin develop --release in python/ffi/")
        return False

    from turboquant import TurboQuant as PyTQ
    from omlx_research.backends.mlx_backend import MlxBackend

    be = MlxBackend.__new__(MlxBackend)  # bypass __init__
    be._perf_module = be._rust_perf()

    shapes = [128, 512, 2048, 8192]  # batched shape sizes (n vectors of group_size dims)
    bits = [4, 3, 2]

    print(f"\n  {'shape':>6s} {'bits':>4s} | {'py encode (μs)':>16s} {'rust encode (μs)':>18s} {'speedup':>8s} | {'py max_err':>10s} {'rust max_err':>10s}")
    print("  " + "-" * 100)
    for n in shapes:
        # n = number of vectors; each vector is group_size floats
        group_size = min(64, n // 4) if n >= 16 else 8
        n_vecs = max(1, n // group_size)
        for b in bits:
            data = np.random.randn(n_vecs * group_size).astype(np.float32)
            # --- Python reference (single-vector API per call)
            py_tq = PyTQ(d=group_size, bit_width=b)
            t0 = time.perf_counter()
            for _ in range(8):
                cv = py_tq.quantize(np.random.randn(group_size).astype(np.float32))
            py_enc_us = (time.perf_counter() - t0) * 1e6 / 8
            # --- Rust FFI (batch encode)
            flat = data.tolist()
            t0 = time.perf_counter()
            rust_q = be.turbo_quant_encode_array(flat, group_size=group_size, bits=b)
            rust_enc_us = (time.perf_counter() - t0) * 1e6
            # --- decode Rust side
            t0 = time.perf_counter()
            recon = be.turbo_quant_decode_array(
                rust_q["packed"], rust_q["scales"], rust_q["zeros"],
                n=n_vecs * group_size, group_size=group_size, bits=b,
            )
            rust_dec_us = (time.perf_counter() - t0) * 1e6

            # Quality check (compare against single Python decode of the same vector)
            test_vec = np.random.randn(group_size).astype(np.float32)
            py_cv = py_tq.quantize(test_vec)
            py_recon = np.array(py_tq.dequantize(py_cv), dtype=np.float32)
            # Rust side: rebuild from a single-vector encode
            rust_qv = be.turbo_quant_encode_array(test_vec.tolist(), group_size=group_size, bits=b)
            rust_recon_v = be.turbo_quant_decode_array(
                rust_qv["packed"], rust_qv["scales"], rust_qv["zeros"],
                n=group_size, group_size=group_size, bits=b,
            )
            rust_recon = np.array(rust_recon_v, dtype=np.float32)

            py_err = float(np.max(np.abs(test_vec - py_recon)))
            rust_err = float(np.max(np.abs(test_vec - rust_recon)))

            speedup = py_enc_us / rust_enc_us if rust_enc_us > 0 else float("inf")
            print(f"  {n:>6d} {b:>4d} | {py_enc_us:>16.1f} {rust_enc_us:>18.1f} {speedup:>7.1f}x | {py_err:>10.4f} {rust_err:>10.4f}")

    return True


def bench_mlx_with_turbo(model_path: str, lengths: list[int]):
    """Step 2 — End-to-end inference: FP16 baseline vs TurboQuant+ KV cache."""
    print()
    print("=" * 64)
    print(f" STEP 2 — MLX inference: FP16 baseline vs TurboQuant+ KV cache")
    print(f" model: {model_path}")
    print("=" * 64)

    import mlx_lm

    print("  Loading model...")
    t0 = time.time()
    model, tokenizer = mlx_lm.load(model_path)
    load_s = time.time() - t0
    print(f"  Loaded in {load_s:.1f}s ({len(model.layers)} layers)")

    # Build a synthetic fill document
    filler = "The quick brown fox jumps over the lazy dog. " * 200

    for length in lengths:
        prompt = filler[:length]

        # ── Baseline FP16 ──
        t0 = time.time()
        text_fp = mlx_lm.generate(
            model, tokenizer, prompt, max_tokens=20, verbose=False,
        )
        t_fp = time.time() - t0

        # ── TurboQuant+ asymmetric (K=FP16, V=4bit) ──
        try:
            from mlx.nn.layers.turbo_kv_cache import TurboKVCache, compact_turbo_cache
            turbo_asym = [TurboKVCache(bits=4, key_bits=None) for _ in range(len(model.layers))]
            t0 = time.time()
            text_tasym = mlx_lm.generate(
                model, tokenizer, prompt, max_tokens=20,
                prompt_cache=turbo_asym, verbose=False,
            )
            t_tasym = time.time() - t0
            n_compressed_asym = compact_turbo_cache(turbo_asym)
        except Exception as e:
            text_tasym = f"err: {e}"
            t_tasym = float("nan")
            n_compressed_asym = -1

        # ── TurboQuant+ symmetric (K=4bit, V=4bit) ──
        try:
            turbo_sym = [TurboKVCache(bits=4, key_bits=4) for _ in range(len(model.layers))]
            t0 = time.time()
            text_tsym = mlx_lm.generate(
                model, tokenizer, prompt, max_tokens=20,
                prompt_cache=turbo_sym, verbose=False,
            )
            t_tsym = time.time() - t0
            n_compressed_sym = compact_turbo_cache(turbo_sym)
        except Exception as e:
            text_tsym = f"err: {e}"
            t_tsym = float("nan")
            n_compressed_sym = -1

        print(f"\n  ── prompt_len={length} ──")
        print(f"    baseline FP16:    {t_fp*1000:>7.0f}ms  {text_fp[:60]!r}")
        print(f"    turbo asymmetric: {t_tasym*1000:>7.0f}ms ({n_compressed_asym}/{len(model.layers)} compressed)  {text_tasym[:60]!r}")
        print(f"    turbo symmetric:  {t_tsym*1000:>7.0f}ms ({n_compressed_sym}/{len(model.layers)} compressed)  {text_tsym[:60]!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 16384])
    parser.add_argument("--model", type=str, default=None,
                        help="MLX model (default: smoke_models role=turboquant)")
    parser.add_argument("--rust-only", action="store_true",
                        help="Skip the MLX inference benchmark (just Rust vs Python)")
    args = parser.parse_args()
    if not args.model:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "python"))
        from omlx_research.smoke_models import default_model_for
        args.model = default_model_for("turboquant")

    if not bench_rust_vs_python():
        return 1

    if not args.rust_only:
        # Resolve model path (local cache or download)
        from huggingface_hub import snapshot_download
        try:
            model_path = snapshot_download(args.model)
        except Exception as e:
            print(f"\n  ⚠ Could not resolve model: {e}")
            print(f"    Run: env -u HF_HUB_OFFLINE HF_HOME=/Users/kooshapari/.cache/huggingface huggingface-cli download {args.model}")
            return 1

        bench_mlx_with_turbo(model_path, args.lengths)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())