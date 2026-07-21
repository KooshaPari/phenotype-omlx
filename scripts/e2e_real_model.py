#!/usr/bin/env python3
"""
Real end-to-end test: Qwen3.5 (SSOT) through the full
phenotype-omlx stack: MLX backend + TurboQuant+ KV cache.

Strategy:
  1. Build a cache list of size len(model.model.layers) — either plain KVCache or TurboKVCache.
  2. Run mlx_lm.generate(prompt_cache=cache) which uses TurboKVCache throughout.
  3. Build a separate prefilled cache to measure KV array bytes (FP16 → compact_turbo_cache).

Measures:
  - Load time
  - Decode tok/s (TurboQuant+ vs FP16 vs MLX native kv_bits=4)
  - KV memory delta (compact saves ~50% KV)
  - Generation quality (text output)
"""
from __future__ import annotations
import os, sys, time, json, gc, subprocess
from pathlib import Path

VENV = Path("/Users/kooshapari/CodeProjects/Phenotype/repos/turboquant_plus/.venv/bin")
if VENV.exists():
    os.environ["PATH"] = str(VENV) + ":" + os.environ.get("PATH", "")
os.environ["MPLBACKEND"] = "Agg"

PROMPT = "def fibonacci(n):\n    "
MAX_TOKENS = 64

def _resolve_model() -> str:
    if os.environ.get("PHENO_MODEL") or os.environ.get("OMLX_READY_MODEL"):
        return os.environ.get("PHENO_MODEL") or os.environ["OMLX_READY_MODEL"]
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "python"))
    from omlx_research.smoke_models import default_model_for
    return default_model_for("turboquant")

MODEL = _resolve_model()


def rss_mb() -> float:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).decode().strip()
    return int(out) / 1024


_peak = rss_mb()


def peak_rss_mb() -> float:
    global _peak
    cur = rss_mb()
    if cur > _peak:
        _peak = cur
    return _peak


def num_layers(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "layers"):
        return len(model.layers)
    raise RuntimeError("Cannot discover num_layers")


def import_stack():
    global mx
    import mlx.core as mx
    from mlx.nn.layers.turbo_kv_cache import (
        TurboKVCache, make_turbo_cache, compact_turbo_cache
    )
    print(f"  mlx={mx.__version__}  metal={mx.metal.is_available()}  peak_rss={peak_rss_mb():.0f}MB")
    return mx, TurboKVCache, make_turbo_cache, compact_turbo_cache


def load_model(path):
    import mlx_lm
    t0 = time.time()
    model, tok = mlx_lm.load(path)
    dt = time.time() - t0
    print(f"  model loaded in {dt:.2f}s  peak_rss={peak_rss_mb():.0f}MB")
    return model, tok


def _eval_cache(cache_list, mx):
    flat = []
    for c in cache_list:
        s = c.state
        if isinstance(s, (list, tuple)):
            flat.extend(s)
        else:
            flat.append(s)
    mx.eval(*flat)


def _kv_bytes(cache_list) -> int:
    total = 0
    for c in cache_list:
        s = c.state
        if isinstance(s, (list, tuple)):
            arrs = s
        else:
            arrs = (s,)
        for a in arrs:
            if hasattr(a, "nbytes"):
                total += a.nbytes
    return total


def measure_kv_for_mode(model, tok, prompt, cache_factory, mx):
    """
    Build a fresh cache via cache_factory(num_layers), prefill to len(prompt),
    eval, return bytes.
    """
    cache = cache_factory(num_layers(model))
    ids = tok.encode(prompt)
    x = mx.array(ids)[None]
    logits = model(x, cache=cache)[:, -1, :]
    mx.eval(logits)
    _eval_cache(cache, mx)
    return _kv_bytes(cache), cache


def gen_baseline(model, tok, prompt, max_tokens, mx):
    """Vanilla mlx_lm.generate — no quantization."""
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.0)
    gc.collect()
    rss_before = rss_mb()
    t0 = time.time()
    text = mlx_lm.generate(
        model, tok, prompt=prompt, max_tokens=max_tokens,
        verbose=False, sampler=sampler,
    )
    dt = time.time() - t0
    from mlx_lm.models.cache import KVCache
    kv_b, _ = measure_kv_for_mode(model, tok, prompt, lambda n: [KVCache() for _ in range(n)], mx)
    return {
        "text": text.strip(),
        "elapsed_s": dt,
        "tok_per_s": max_tokens / dt if dt > 0 else 0,
        "rss_delta_mb": rss_mb() - rss_before,
        "peak_rss_mb": peak_rss_mb(),
        "kv_fp16_bytes": kv_b,
        "kv_saved_mb": 0,
        "kv_saved_pct": 0,
    }


def gen_mlx_native(model, tok, prompt, max_tokens, mx, kv_bits=4):
    """MLX native kv_bits quantization (built-in to mlx_lm)."""
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.models.cache import QuantizedKVCache as _QKV
    sampler = make_sampler(temp=0.0)
    gc.collect()
    rss_before = rss_mb()
    t0 = time.time()
    text = mlx_lm.generate(
        model, tok, prompt=prompt, max_tokens=max_tokens,
        verbose=False, sampler=sampler, kv_bits=kv_bits,
    )
    dt = time.time() - t0
    # Build a representative quantized cache
    cache = [_QKV(group_size=64, bits=kv_bits) for _ in range(num_layers(model))]
    ids = tok.encode(prompt)
    x = mx.array(ids)[None]
    logits = model(x, cache=cache)[:, -1, :]
    mx.eval(logits)
    _eval_cache(cache, mx)
    kv_b = _kv_bytes(cache)
    return {
        "text": text.strip(),
        "elapsed_s": dt,
        "tok_per_s": max_tokens / dt if dt > 0 else 0,
        "rss_delta_mb": rss_mb() - rss_before,
        "peak_rss_mb": peak_rss_mb(),
        "kv_turbo_bytes": kv_b,
        "kv_saved_mb": 0,
        "kv_saved_pct": 0,
        "config": f"kv_bits={kv_bits}",
    }


def gen_turboquant(model, tok, prompt, max_tokens, mx, bits=4, key_bits=None):
    """
    TurboQuant+ end-to-end:
      1. Build TurboKVCache list
      2. mlx_lm.generate runs with prompt_cache=TurboKVCache list
      3. Measure cache array size (FP16-style internal storage during prefill)
      4. compact_turbo_cache() compresses FP16 → TurboQuant
      5. Measure after compact
    """
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.models.cache import KVCache
    from mlx.nn.layers.turbo_kv_cache import TurboKVCache, compact_turbo_cache
    sampler = make_sampler(temp=0.0)
    gc.collect()
    rss_before = rss_mb()

    n_layers = num_layers(model)
    turbo_cache = [TurboKVCache(bits=bits, key_bits=key_bits) for _ in range(n_layers)]

    t0 = time.time()
    text = mlx_lm.generate(
        model, tok, prompt=prompt, max_tokens=max_tokens,
        verbose=False, sampler=sampler, prompt_cache=turbo_cache,
    )
    dt = time.time() - t0

    # Measure KV: build a fresh TurboKVCache, prefill, measure FP16 portion + compacted
    cache = [TurboKVCache(bits=bits, key_bits=key_bits) for _ in range(n_layers)]
    ids = tok.encode(prompt)
    x = mx.array(ids)[None]
    logits = model(x, cache=cache)[:, -1, :]
    mx.eval(logits)
    _eval_cache(cache, mx)
    fp16_bytes = _kv_bytes(cache)

    n_compacted = compact_turbo_cache(cache)
    _eval_cache(cache, mx)
    turbo_bytes = _kv_bytes(cache)

    saved_bytes = max(0, fp16_bytes - turbo_bytes)
    saved_pct = saved_bytes / fp16_bytes * 100 if fp16_bytes > 0 else 0

    return {
        "text": text.strip(),
        "elapsed_s": dt,
        "tok_per_s": max_tokens / dt if dt > 0 else 0,
        "rss_delta_mb": rss_mb() - rss_before,
        "peak_rss_mb": peak_rss_mb(),
        "kv_fp16_bytes": fp16_bytes,
        "kv_turbo_bytes": turbo_bytes,
        "kv_saved_bytes": saved_bytes,
        "kv_saved_mb": saved_bytes / 1024 / 1024,
        "kv_saved_pct": saved_pct,
        "compact_layers": n_compacted,
        "num_layers": n_layers,
        "config": f"bits={bits}, key_bits={key_bits}",
    }


def main():
    print("=" * 78)
    print(" phenotype-omlx — REAL END-TO-END (4 KV-cache strategies)")
    print(f" Model:  {Path(MODEL).name}")
    print(f" Prompt: {PROMPT.strip()!r}")
    print(f" Tokens: {MAX_TOKENS}")
    print("=" * 78)

    print("\n[1/4] Importing stack…", flush=True)
    mx, TurboKVCache, make_turbo_cache, compact_turbo_cache = import_stack()

    print("\n[2/4] Loading model…", flush=True)
    model, tok = load_model(MODEL)
    n_layers = num_layers(model)
    arch = type(model).__name__
    print(f"  arch: {arch}  layers={n_layers}  vocab={tok.vocab_size if hasattr(tok, 'vocab_size') else '?'}")
    gc.collect()

    # ── Mode 1: Baseline FP16 KVCache ─────────────────────────────────
    print(f"\n[3/4] Mode 1 — Baseline FP16 KVCache…")
    base = gen_baseline(model, tok, PROMPT, MAX_TOKENS, mx)
    print(f"    text:   {base['text'][:80]}")
    print(f"    speed:  {base['tok_per_s']:.1f} tok/s  peak: {base['peak_rss_mb']:.0f}MB")
    print(f"    KV:     {base['kv_fp16_bytes']/1024:.1f} KB (FP16)")
    gc.collect()

    # ── Mode 2: MLX native kv_bits=4 ──────────────────────────────────
    print(f"\n[4/4] Mode 2 — MLX native kv_bits=4…")
    native4 = gen_mlx_native(model, tok, PROMPT, MAX_TOKENS, mx, kv_bits=4)
    print(f"    text:   {native4['text'][:80]}")
    print(f"    speed:  {native4['tok_per_s']:.1f} tok/s  peak: {native4['peak_rss_mb']:.0f}MB")
    print(f"    KV:     {native4['kv_turbo_bytes']/1024:.1f} KB (after compact internal)")
    gc.collect()

    # ── Mode 3: TurboQuant+ asymmetric ────────────────────────────────
    print(f"\n[5/4] Mode 3 — TurboQuant+ ASYMMETRIC (K=FP16, V=4bit)…")
    asym = gen_turboquant(model, tok, PROMPT, MAX_TOKENS, mx, bits=4, key_bits=None)
    print(f"    text:   {asym['text'][:80]}")
    print(f"    speed:  {asym['tok_per_s']:.1f} tok/s  peak: {asym['peak_rss_mb']:.0f}MB")
    print(f"    KV:     {asym['kv_fp16_bytes']/1024:.1f} KB → {asym['kv_turbo_bytes']/1024:.1f} KB")
    print(f"    saved:  {asym['kv_saved_bytes']/1024:.1f} KB ({asym['kv_saved_pct']:.1f}%)")
    gc.collect()

    # ── Mode 4: TurboQuant+ symmetric ─────────────────────────────────
    print(f"\n[6/4] Mode 4 — TurboQuant+ SYMMETRIC (K=4bit, V=4bit)…")
    sym = gen_turboquant(model, tok, PROMPT, MAX_TOKENS, mx, bits=4, key_bits=4)
    print(f"    text:   {sym['text'][:80]}")
    print(f"    speed:  {sym['tok_per_s']:.1f} tok/s  peak: {sym['peak_rss_mb']:.0f}MB")
    print(f"    KV:     {sym['kv_fp16_bytes']/1024:.1f} KB → {sym['kv_turbo_bytes']/1024:.1f} KB")
    print(f"    saved:  {sym['kv_saved_bytes']/1024:.1f} KB ({sym['kv_saved_pct']:.1f}%)")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f" SUMMARY — {Path(MODEL).name}  ({n_layers} layers, {MAX_TOKENS} tokens)")
    print("=" * 78)

    rows = [
        ("Mode 1: Baseline FP16 KVCache",        base,    "kv_fp16_bytes"),
        ("Mode 2: MLX native kv_bits=4",         native4, "kv_turbo_bytes"),
        ("Mode 3: TurboQuant+ asym K=FP16 V=4b", asym,    "kv_turbo_bytes"),
        ("Mode 4: TurboQuant+ sym  K=4b  V=4b",  sym,     "kv_turbo_bytes"),
    ]
    print(f"  {'Mode':<38} {'tok/s':>8} {'peak_rss':>11} {'KV bytes':>14} {'saved':>10}")
    print(f"  {'-'*86}")
    for name, r, kv_key in rows:
        kv_str = "—"
        saved_str = "—"
        if kv_key and kv_key in r:
            kv_b = r[kv_key]
            kv_str = f"{kv_b:>14}"
            if "kv_saved_pct" in r and r["kv_saved_pct"] > 0:
                saved_str = f"{r['kv_saved_pct']:>9.1f}%"
        print(f"  {name:<38} {r['tok_per_s']:>7.1f}  {r['peak_rss_mb']:>6.0f}MB   {kv_str}  {saved_str}")

    # Quality
    print()
    print(f"  Generation quality (deterministic temp=0):")
    base_text = base['text'][:60]
    for name, r, _ in rows[1:]:
        same = r['text'][:60] == base_text
        mark = "✅" if same else "⚠️"
        diff_marker = ""
        if not same:
            for i, (a, b) in enumerate(zip(r['text'], base_text)):
                if a != b:
                    diff_marker = f"  (first diff at char {i}: {base_text[i:i+20]!r} vs {r['text'][i:i+20]!r})"
                    break
        print(f"    {mark} {name:<36} → {'identical' if same else 'DIFFERS'}{diff_marker}")

    # Speed
    print()
    print(f"  Decode speed vs baseline:")
    for name, r, _ in rows[1:]:
        pct = r['tok_per_s'] / base['tok_per_s'] * 100
        delta = r['tok_per_s'] - base['tok_per_s']
        print(f"    {name:<38} → {pct:.1f}% ({delta:+.1f} tok/s)")

    # Save
    results = {
        "model": Path(MODEL).name,
        "model_path": MODEL,
        "arch": arch,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "num_layers": n_layers,
        "modes": {name: r for name, r, _ in rows},
    }
    out = Path("/Users/kooshapari/CodeProjects/Phenotype/repos/phenotype-omlx/research/e2e_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {out}")
    print("=" * 78)


if __name__ == "__main__":
    main()