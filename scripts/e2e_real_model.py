#!/usr/bin/env python3
"""
Real end-to-end test: standard Qwen2.5-0.5B transformer through the full
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
import os, time, gc
from pathlib import Path

try:
    from e2e_real_model_support import eval_cache as _eval_cache, kv_bytes as _kv_bytes, measure_kv_for_mode, num_layers, peak_rss_mb, rss_mb
except ModuleNotFoundError:
    from scripts.e2e_real_model_support import eval_cache as _eval_cache, kv_bytes as _kv_bytes, measure_kv_for_mode, num_layers, peak_rss_mb, rss_mb

try:
    from e2e_validation import ValidationError
except ModuleNotFoundError:
    from scripts.e2e_validation import ValidationError

try:
    from e2e_real_model_host import (
        BENCHMARK_WORKLOAD_PATH,
        DecodeObservation,
        TeacherForcedCapabilityError,
        benchmark_repeated_arm,
        compare_teacher_forced_nll,
        observe_synchronized_decode,
        run_teacher_forced_scorer,
        run_lite_teacher_forced_lifecycle,
        run_one_token_lite_probe,
        run_bounded_probe_process,
        default_teacher_forced_scorer_factory,
        collect_synchronized_decode_benchmark,
        publish_host_validation_evidence,
        run_host_compacted_arm,
        run_host_validation,
        load_benchmark_workload,
        validation_manifest_for_workload,
        validation_manifest_from_environment,
    )
except ModuleNotFoundError:
    from scripts.e2e_real_model_host import (
        BENCHMARK_WORKLOAD_PATH,
        DecodeObservation,
        TeacherForcedCapabilityError,
        benchmark_repeated_arm,
        compare_teacher_forced_nll,
        observe_synchronized_decode,
        run_teacher_forced_scorer,
        run_lite_teacher_forced_lifecycle,
        run_one_token_lite_probe,
        run_bounded_probe_process,
        default_teacher_forced_scorer_factory,
        collect_synchronized_decode_benchmark,
        publish_host_validation_evidence,
        run_host_compacted_arm,
        run_host_validation,
        load_benchmark_workload,
        validation_manifest_for_workload,
        validation_manifest_from_environment,
    )

VENV = Path("/Users/kooshapari/CodeProjects/Phenotype/repos/turboquant_plus/.venv/bin")
if VENV.exists():
    os.environ["PATH"] = str(VENV) + ":" + os.environ.get("PATH", "")
os.environ["MPLBACKEND"] = "Agg"

MODEL_TINY = "/Users/kooshapari/.cache/huggingface/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
MODEL_4B   = "/Users/kooshapari/.omlx/models/Rishu11277/Qwopus3.5-4B-Coder-mlx-4Bit"
MAX_TOKENS = 64
MODEL = os.environ.get("PHENO_MODEL", MODEL_TINY)
EVIDENCE_OUTPUT = Path(__file__).resolve().parents[1] / "research" / "e2e_results.json"


def _benchmark_config() -> tuple[int, int]:
    """Read an explicit warmup-plus-measurement benchmark configuration."""

    repeats = int(os.environ.get("PHENO_BENCHMARK_REPEATS", "3"))
    warmup_count = int(os.environ.get("PHENO_BENCHMARK_WARMUPS", "1"))
    if repeats <= 0 or warmup_count < 0 or warmup_count >= repeats:
        raise ValueError("benchmark requires repeats > warmups >= 0")
    return repeats, warmup_count


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


def gen_baseline(model, tok, prompt, max_tokens, mx):
    """Vanilla mlx_lm.generate — no quantization."""
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.0)
    gc.collect()
    rss_before = rss_mb()
    observation = observe_synchronized_decode(
        generate=lambda: mlx_lm.generate(
            model, tok, prompt=prompt, max_tokens=max_tokens,
            verbose=False, sampler=sampler,
        ),
        tokenize=tok.encode,
        synchronize=mx.synchronize,
        clock=time.perf_counter,
    )
    from mlx_lm.models.cache import KVCache
    kv_b, _ = measure_kv_for_mode(model, tok, prompt, lambda n: [KVCache() for _ in range(n)], mx)
    return {
        "text": observation.text.strip(),
        "actual_tokens": observation.actual_tokens,
        "elapsed_s": observation.elapsed_seconds,
        "tok_per_s": (
            observation.actual_tokens / observation.elapsed_seconds
            if observation.elapsed_seconds > 0
            else 0
        ),
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
    observation = observe_synchronized_decode(
        generate=lambda: mlx_lm.generate(
            model, tok, prompt=prompt, max_tokens=max_tokens,
            verbose=False, sampler=sampler, kv_bits=kv_bits,
        ),
        tokenize=tok.encode,
        synchronize=mx.synchronize,
        clock=time.perf_counter,
    )
    # Build a representative quantized cache
    cache = [_QKV(group_size=64, bits=kv_bits) for _ in range(num_layers(model))]
    ids = tok.encode(prompt)
    x = mx.array(ids)[None]
    logits = model(x, cache=cache)[:, -1, :]
    mx.eval(logits)
    _eval_cache(cache, mx)
    kv_b = _kv_bytes(cache)
    return {
        "text": observation.text.strip(),
        "actual_tokens": observation.actual_tokens,
        "elapsed_s": observation.elapsed_seconds,
        "tok_per_s": (
            observation.actual_tokens / observation.elapsed_seconds
            if observation.elapsed_seconds > 0
            else 0
        ),
        "rss_delta_mb": rss_mb() - rss_before,
        "peak_rss_mb": peak_rss_mb(),
        "kv_turbo_bytes": kv_b,
        "kv_saved_mb": 0,
        "kv_saved_pct": 0,
        "config": f"kv_bits={kv_bits}",
    }


def gen_turboquant(model, tok, prompt, max_tokens, mx, *, manifest, bits=4, key_bits=None):
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
    from mlx.nn.layers.turbo_kv_cache import TurboKVCache, compact_turbo_cache
    sampler = make_sampler(temp=0.0)
    gc.collect()
    rss_before = rss_mb()

    n_layers = num_layers(model)
    text = ""
    observation: DecodeObservation | None = None
    fp16_bytes = 0
    n_compacted = 0

    def cache_factory():
        return make_turbo_cache(
            model,
            bits=bits,
            key_bits=bits if key_bits is None else key_bits,
        )

    def generate(cache):
        nonlocal text, observation
        observation = observe_synchronized_decode(
            generate=lambda: mlx_lm.generate(
                model, tok, prompt=prompt, max_tokens=max_tokens,
                verbose=False, sampler=sampler, prompt_cache=cache,
            ),
            tokenize=tok.encode,
            synchronize=mx.synchronize,
            clock=time.perf_counter,
        )
        text = observation.text
        return tok.encode(text)

    def materialize(cache, _generated_tokens):
        nonlocal fp16_bytes
        _eval_cache(cache, mx)
        fp16_bytes = _kv_bytes(cache)

    def compact(cache):
        nonlocal n_compacted
        n_compacted = compact_turbo_cache(cache)
        return fp16_bytes - _kv_bytes(cache)

    def score(cache, _generated_tokens):
        _eval_cache(cache, mx)

    run = run_host_compacted_arm(
        manifest=manifest,
        cache_factory=cache_factory,
        generate=generate,
        materialize=materialize,
        compact=compact,
        score=score,
    )
    turbo_bytes = _kv_bytes(run.cache)
    if observation is None:
        raise RuntimeError("synchronized decode observation was not recorded")

    saved_bytes = max(0, fp16_bytes - turbo_bytes)
    saved_pct = saved_bytes / fp16_bytes * 100 if fp16_bytes > 0 else 0

    return {
        "text": text.strip(),
        "actual_tokens": observation.actual_tokens,
        "elapsed_s": observation.elapsed_seconds,
        "tok_per_s": (
            observation.actual_tokens / observation.elapsed_seconds
            if observation.elapsed_seconds > 0
            else 0
        ),
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


def main(*, teacher_forced_scorer=None):
    workload = load_benchmark_workload()
    manifest = validation_manifest_for_workload(workload=workload)
    prompt = workload.prompt
    repeats, warmup_count = _benchmark_config()
    print("=" * 78)
    print(" phenotype-omlx — REAL END-TO-END (4 KV-cache strategies)")
    print(f" Model:  {Path(MODEL).name}")
    print(f" Prompt: {prompt.strip()!r}")
    print(f" Tokens: {MAX_TOKENS}")
    print("=" * 78)

    print("\n[1/4] Importing stack…", flush=True)
    mx, TurboKVCache, make_turbo_cache, compact_turbo_cache = import_stack()

    if teacher_forced_scorer is None:
        teacher_forced_scorer = default_teacher_forced_scorer_factory(TurboKVCache)

    print("\n[2/4] Loading model…", flush=True)
    model, tok = load_model(MODEL)
    n_layers = num_layers(model)
    arch = type(model).__name__
    print(f"  arch: {arch}  layers={n_layers}  vocab={tok.vocab_size if hasattr(tok, 'vocab_size') else '?'}")
    gc.collect()

    # ── Mode 1: Baseline FP16 KVCache ─────────────────────────────────
    print(f"\n[3/4] Mode 1 — Baseline FP16 KVCache…")
    base = benchmark_repeated_arm(
        run_once=lambda: gen_baseline(model, tok, prompt, MAX_TOKENS, mx),
        repeats=repeats,
        warmup_count=warmup_count,
    )
    print(f"    text:   {base['text'][:80]}")
    print(f"    speed:  {base['tok_per_s']:.1f} tok/s  peak: {base['peak_rss_mb']:.0f}MB")
    print(f"    KV:     {base['kv_fp16_bytes']/1024:.1f} KB (FP16)")
    gc.collect()

    # ── Mode 2: MLX native kv_bits=4 ──────────────────────────────────
    print(f"\n[4/4] Mode 2 — MLX native kv_bits=4…")
    native4 = benchmark_repeated_arm(
        run_once=lambda: gen_mlx_native(model, tok, prompt, MAX_TOKENS, mx, kv_bits=4),
        repeats=repeats,
        warmup_count=warmup_count,
    )
    print(f"    text:   {native4['text'][:80]}")
    print(f"    speed:  {native4['tok_per_s']:.1f} tok/s  peak: {native4['peak_rss_mb']:.0f}MB")
    print(f"    KV:     {native4['kv_turbo_bytes']/1024:.1f} KB (after compact internal)")
    gc.collect()

    # ── Mode 3: TurboQuant+ asymmetric ────────────────────────────────
    print(f"\n[5/4] Mode 3 — TurboQuant+ ASYMMETRIC (K=FP16, V=4bit)…")
    asym = benchmark_repeated_arm(
        run_once=lambda: gen_turboquant(
            model, tok, prompt, MAX_TOKENS, mx, manifest=manifest, bits=4, key_bits=None
        ),
        repeats=repeats,
        warmup_count=warmup_count,
    )
    print(f"    text:   {asym['text'][:80]}")
    print(f"    speed:  {asym['tok_per_s']:.1f} tok/s  peak: {asym['peak_rss_mb']:.0f}MB")
    print(f"    KV:     {asym['kv_fp16_bytes']/1024:.1f} KB → {asym['kv_turbo_bytes']/1024:.1f} KB")
    print(f"    saved:  {asym['kv_saved_bytes']/1024:.1f} KB ({asym['kv_saved_pct']:.1f}%)")
    gc.collect()

    # ── Mode 4: TurboQuant+ symmetric ─────────────────────────────────
    print(f"\n[6/4] Mode 4 — TurboQuant+ SYMMETRIC (K=4bit, V=4bit)…")
    sym = benchmark_repeated_arm(
        run_once=lambda: gen_turboquant(
            model, tok, prompt, MAX_TOKENS, mx, manifest=manifest, bits=4, key_bits=4
        ),
        repeats=repeats,
        warmup_count=warmup_count,
    )
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
    print(f"  Generation diagnostic (non-gating, deterministic temp=0):")
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

    if teacher_forced_scorer is None:
        raise ValidationError(
            "teacher-forced NLL/PPL comparison is required before evidence publication"
        )
    teacher_forced_comparison = run_teacher_forced_scorer(
        scorer=teacher_forced_scorer,
        model=model,
        tokenizer=tok,
        workload=workload,
        mx=mx,
    )

    # Save
    results = {
        "model_revision": manifest.model_revision,
        "corpus_revision": manifest.corpus_revision,
        "tokenizer_revision": manifest.tokenizer_revision,
        "benchmark_workload": {
            "name": workload.name,
            "kind": workload.kind,
            "revision": workload.revision,
        },
        "model": Path(MODEL).name,
        "model_path": MODEL,
        "arch": arch,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "num_layers": n_layers,
        "teacher_forced": {
            "token_count": teacher_forced_comparison.baseline.token_count,
            "baseline_total_nll": teacher_forced_comparison.baseline.total_nll,
            "compacted_total_nll": teacher_forced_comparison.compacted.total_nll,
            "compacted_minus_baseline": teacher_forced_comparison.compacted_minus_baseline,
            "perplexity_ratio": teacher_forced_comparison.perplexity_ratio,
            "workload_continuation": workload.teacher_forced_continuation,
        },
        "modes": {name: r for name, r, _ in rows},
    }
    EVIDENCE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out = publish_host_validation_evidence(
        destination=EVIDENCE_OUTPUT,
        candidate=results,
        approved_output_root=EVIDENCE_OUTPUT.parent,
    )
    print(f"\n  Results saved: {out}")
    print("=" * 78)


if __name__ == "__main__":
    main()
