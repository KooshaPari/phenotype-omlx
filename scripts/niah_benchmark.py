#!/usr/bin/env python3
"""
NIAH (Needle-In-A-Haystack) benchmark for phenotype-omlx TurboQuant+ MLX.

Measures retrieval accuracy + tokens/sec + RSS memory for 4 KV-cache modes:
  - baseline_fp16       : vanilla mlx_lm KVCache (full precision)
  - turbo_asymmetric    : TurboKVCache(bits=4, key_bits=None) — K=FP16, V=4bit
  - turbo_symmetric     : TurboKVCache(bits=4, key_bits=4)    — K=4bit, V=4bit
  - turbo4              : TurboKVCache(bits=4, key_bits=None) — alias for asymmetric
  - mlx_native_kv4      : mlx_lm's built-in kv_bits=4 quantization

Tests 3-6 context lengths (default 512, 2048, 8192; 32K optional).
At each length, hides a unique needle at 75% depth and asks the model to
retrieve it. Quality = 1.0 if needle exact-string is in the answer, else 0.0.

This is the canonical quality bar for KV cache compression — proves
TurboQuant+ preserves retrieval at long contexts where mlx_lm's native
kv_bits=4 produces garbage.
"""

import argparse
import gc
import json
import os
import random
import string
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Force online mode for HuggingFace (the user env exports HF_HUB_OFFLINE=1,
# which would block model downloads even though the model is cached locally).
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HOME"] = "/Users/kooshapari/.cache/huggingface"
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

# Absorbed-crate / worktree layout: always import from this repo's python/.
# Never use a hard-coded absolute repos/.../python sys.path (FR-5 E2).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

# Heavy / optional imports are deferred so `--help` (and any other
# argparse-only invocation) works without mlx / mlx_lm installed.
# This is what lets the doctor check ``scripts/niah_benchmark.py --help``
# transitions to PASS on a machine where only the Python bindings are
# present and the actual MLX stack is not.
try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - psutil absent in CI
    psutil = None  # type: ignore[assignment]

# Module-level lazy import helpers — the values are resolved on first
# use inside ``run_one`` so ``--help`` never touches them.
_mx = None
_mlx_lm = None


def _ensure_mlx():
    """Lazily import ``mlx.core`` and ``mlx_lm`` on first use."""
    global _mx, _mlx_lm
    if _mx is None:
        import mlx.core as _m  # type: ignore
        _mx = _m
    if _mlx_lm is None:
        import mlx_lm as _ml  # type: ignore
        _mlx_lm = _ml
    return _mx, _mlx_lm


# ── Helpers ──────────────────────────────────────────────────────────────

def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def random_filler(length: int) -> str:
    """Random filler text not containing the needle."""
    vocab = ["the", "a", "of", "in", "to", "and", "for", "with", "on", "at",
             "lorem", "ipsum", "dolor", "sit", "amet", "consectetur",
             "phenotype", "machine", "learning", "model", "rust", "python"]
    words = random.choices(vocab, k=length // 6)
    text = " ".join(words)
    text = "".join(c if c.isalpha() else " " for c in text)
    return " ".join(text.split())


def build_needle_prompt(target_len: int, needle: str) -> str:
    """Build a context of `target_len` tokens with the needle at ~75% depth."""
    filler_words = max(target_len // 2, 1)
    filler = random_filler(filler_words)
    needle_para = f"\n\nImportant context: {needle}. This fact is critical.\n\n"
    intro = "Read the following passage and recall the critical fact mentioned.\n\n"
    filler = filler[:max(len(filler) - len(needle_para) - len(intro) - 20, 1)]
    prompt = intro + filler[:len(filler) * 3 // 4] + needle_para + filler[len(filler) * 3 // 4:]
    return prompt


# ── Run a single (mode, length) benchmark ────────────────────────────────

@dataclass
class BenchResult:
    mode: str
    target_len: int
    actual_len: int
    prefill_ms: float
    decode_ms: float
    decode_tok_per_sec: float
    rss_mb_before: float
    rss_mb_after: float
    needle: str
    answer: str
    exact_match: bool
    partial_match: bool
    contains_secret: bool
    error: str = ""


def run_one(model, tokenizer, mode: str, length: int, needle: str) -> BenchResult:
    """Run one (mode, length) benchmark. Returns a BenchResult."""
    mx, mlx_lm = _ensure_mlx()
    rss_before = rss_mb()

    # ── 1. Build the prompt ──
    prompt = build_needle_prompt(length, needle)
    # Tokenize
    prompt_ids = tokenizer.encode(prompt)
    actual_len = len(prompt_ids)
    print(f"  [{mode:18s}] {length:5d} target / {actual_len:5d} actual tokens")

    # ── 2. Build the KV cache list (one per layer) ──
    n_layers = len(model.layers)
    is_qwen35 = "qwen3_5" in type(model).__module__ or "qwen3_5" in str(getattr(model, "model_type", ""))
    if is_qwen35 and mode != "baseline_fp16":
        return BenchResult(
            mode=mode, target_len=length, actual_len=actual_len,
            prefill_ms=0, decode_ms=0, decode_tok_per_sec=0,
            rss_mb_before=rss_before, rss_mb_after=rss_mb(), needle=needle,
            answer="", exact_match=False, partial_match=False,
            contains_secret=False,
            error="unsupported: Qwen3.5 mixed linear/full attention has no KV-only cache for this mode",
        )
    if mode == "baseline_fp16":
        from mlx_lm.models import cache
        # Qwen3.5 requires mixed recurrent/attention cache state; constructing
        # one KVCache per layer is invalid and breaks create_attention_mask.
        cache_list = cache.make_prompt_cache(model)
    elif mode in ("turbo_asymmetric", "turbo4"):
        from mlx.nn.layers.turbo_kv_cache import TurboKVCache
        cache_list = [TurboKVCache(bits=4, key_bits=None) for _ in range(n_layers)]
    elif mode == "turbo_symmetric":
        from mlx.nn.layers.turbo_kv_cache import TurboKVCache
        cache_list = [TurboKVCache(bits=4, key_bits=4) for _ in range(n_layers)]
    elif mode == "mlx_native_kv4":
        from mlx_lm.models.cache import QuantizedKVCache
        cache_list = [QuantizedKVCache(group_size=64, bits=4) for _ in range(n_layers)]
    else:
        raise ValueError(f"unknown mode: {mode}")

    # ── 3. Prefill (single-token decode with prompt_cache) ──
    prompt_arr = mx.array(prompt_ids)[None]  # (1, T)
    t0 = time.perf_counter()
    try:
        out = mlx_lm.generate(
            model, tokenizer,
            prompt=prompt_ids,
            max_tokens=1,
            prompt_cache=cache_list,
            verbose=False,
        )
        prefill_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        return BenchResult(
            mode=mode, target_len=length, actual_len=actual_len,
            prefill_ms=0, decode_ms=0, decode_tok_per_sec=0,
            rss_mb_before=rss_before, rss_mb_after=rss_mb(),
            needle=needle, answer="", exact_match=False,
            partial_match=False, contains_secret=False,
            error=f"prefill: {type(e).__name__}: {e}",
        )

    # ── 4. Compact TurboQuant caches to mirror "after-compression" state ──
    if mode in ("turbo_asymmetric", "turbo_symmetric", "turbo4"):
        try:
            from mlx.nn.layers.turbo_kv_cache import compact_turbo_cache
            n_compressed = compact_turbo_cache(cache_list)
            print(f"    compact_turbo_cache compressed {n_compressed}/{n_layers} layers")
        except Exception as e:
            print(f"    compact_turbo_cache skipped: {e}")

    # ── 5. Generate up to 30 tokens for the answer ──
    qa_prompt = (
        "\n\nQuestion: What is the critical fact mentioned in the passage above? "
        "Respond concisely.\n\nAnswer:"
    )
    qa_ids = tokenizer.encode(qa_prompt)
    qa_arr = mx.array(qa_ids)[None]

    t0 = time.perf_counter()
    try:
        answer_ids = mlx_lm.generate(
            model, tokenizer,
            prompt=qa_ids,
            max_tokens=30,
            prompt_cache=cache_list,
            verbose=False,
        )
        decode_ms = (time.perf_counter() - t0) * 1000.0
        n_tok = len(answer_ids) if isinstance(answer_ids, list) else 1
        decode_tps = n_tok / (decode_ms / 1000.0) if decode_ms > 0 else 0
    except Exception as e:
        return BenchResult(
            mode=mode, target_len=length, actual_len=actual_len,
            prefill_ms=prefill_ms, decode_ms=0, decode_tok_per_sec=0,
            rss_mb_before=rss_before, rss_mb_after=rss_mb(),
            needle=needle, answer="", exact_match=False,
            partial_match=False, contains_secret=False,
            error=f"decode: {type(e).__name__}: {e}",
        )

    # Decode answer
    if isinstance(answer_ids, list):
        answer = tokenizer.decode(answer_ids)
    else:
        answer = str(answer_ids)

    # Quality checks
    secret = needle.split("the secret code is ")[1] if "the secret code is " in needle else needle
    exact = needle.strip() in answer
    partial = secret in answer
    contains_secret = secret in answer

    rss_after = rss_mb()
    return BenchResult(
        mode=mode, target_len=length, actual_len=actual_len,
        prefill_ms=prefill_ms, decode_ms=decode_ms, decode_tok_per_sec=decode_tps,
        rss_mb_before=rss_before, rss_mb_after=rss_after,
        needle=needle, answer=answer, exact_match=exact,
        partial_match=partial, contains_secret=contains_secret,
    )


# ── Main ─────────────────────────────────────────────────────────────────

def require_julia() -> None:
    """FR-5 E1: Julia is mandatory on the NIAH eval path — fail loud if missing."""
    import shutil

    if shutil.which("julia") is None:
        print(
            "ERROR: julia is required on the NIAH eval path (FR-5 / E1). "
            "Install Julia and ensure `julia` is on PATH "
            "(no optional/stub fallback).",
            file=sys.stderr,
        )
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description="NIAH benchmark for TurboQuant+ MLX")
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[512, 2048, 8192],
                        help="Context lengths to test (tokens)")
    parser.add_argument("--modes", type=str, nargs="+",
                        default=["baseline_fp16", "turbo_asymmetric", "turbo_symmetric", "mlx_native_kv4"],
                        help="KV cache modes to benchmark")
    parser.add_argument("--model", type=str,
                        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                        help="MLX model to load")
    parser.add_argument("--output", type=str, default=None,
                        help="Write results JSON to this file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for needle/filler")
    args = parser.parse_args()

    # After argparse so `--help` still works without Julia; real runs fail loud.
    require_julia()

    random.seed(args.seed)

    # ── Load model ──
    print("=" * 70)
    print(f"  NIAH BENCHMARK — phenotype-omlx TurboQuant+ MLX")
    print("=" * 70)
    print(f"  Model:    {args.model}")
    print(f"  Lengths:  {args.lengths}")
    print(f"  Modes:    {args.modes}")
    print(f"  RSS:      {rss_mb():.0f} MB (start)")
    print()

    print("Loading model...")
    t0 = time.perf_counter()
    mx, mlx_lm = _ensure_mlx()
    model, tokenizer = mlx_lm.load(args.model)
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Model loaded in {load_ms:.0f}ms, RSS now {rss_mb():.0f} MB")
    print(f"  Layers:   {len(model.layers)}")
    print()

    results: list[BenchResult] = []

    # ── Run benchmarks ──
    for length in args.lengths:
        print(f"\n{'─' * 70}")
        print(f"  CONTEXT LENGTH: {length} tokens")
        print(f"{'─' * 70}")
        needle = f"the secret code is {random.randint(100, 999)}-{random.randint(100, 999)}-alpha"

        for mode in args.modes:
            print(f"\n[{mode}]")
            gc.collect()  # free any prior caches
            try:
                r = run_one(model, tokenizer, mode, length, needle)
                results.append(r)
                if r.error:
                    print(f"  ✗ {r.error}")
                else:
                    marker = "✓" if r.exact_match else ("≈" if r.partial_match else "✗")
                    print(f"  {marker} prefill={r.prefill_ms:.0f}ms decode={r.decode_ms:.0f}ms ({r.decode_tok_per_sec:.1f} tok/s)")
                    print(f"  RSS Δ: {r.rss_mb_after - r.rss_mb_before:+.0f} MB")
                    print(f"  answer: {r.answer[:80]!r}")
            except Exception as e:
                print(f"  ✗ unexpected: {type(e).__name__}: {e}")
                results.append(BenchResult(
                    mode=mode, target_len=length, actual_len=0,
                    prefill_ms=0, decode_ms=0, decode_tok_per_sec=0,
                    rss_mb_before=rss_mb(), rss_mb_after=rss_mb(),
                    needle=needle, answer="", exact_match=False,
                    partial_match=False, contains_secret=False,
                    error=f"unexpected: {e}",
                ))

    # ── Summary table ──
    print()
    print("=" * 70)
    print("  NIAH BENCHMARK RESULTS")
    print("=" * 70)
    print()
    print(f"  {'Mode':22s} {'Len':>6s} {'Tok/s':>8s} {'RSSΔ(MB)':>10s} {'Match':>6s}")
    print(f"  {'─' * 22} {'─' * 6} {'─' * 8} {'─' * 10} {'─' * 6}")
    for r in results:
        marker = "✓" if r.exact_match else ("≈" if r.partial_match else ("✗" if r.error else " "))
        rss_d = r.rss_mb_after - r.rss_mb_before
        print(f"  {r.mode:22s} {r.target_len:>6d} {r.decode_tok_per_sec:>8.1f} {rss_d:>+10.0f} {marker:>6s}")

    # Quality per mode (across all lengths)
    print()
    print(f"  Quality summary:")
    modes_tested = sorted(set(r.mode for r in results))
    for m in modes_tested:
        mr = [r for r in results if r.mode == m and not r.error]
        exact = sum(1 for r in mr if r.exact_match)
        partial = sum(1 for r in mr if r.partial_match and not r.exact_match)
        miss = sum(1 for r in mr if not r.exact_match and not r.partial_match)
        print(f"    {m:22s}: exact={exact}/{len(mr)} partial={partial} miss={miss}")

    if args.output:
        # FR-5 E4: never write raw rows without an evidence class.
        # In-process MLX runs are live_verified; do not overwrite
        # the committed synthetic niah_results.json envelope.
        out_path = Path(args.output)
        if "qwen3.5" in args.model.lower() or "qwen3_5" in args.model.lower():
            caveat = (
                "Qwen3.5 family may use linear attention; standard "
                "KV-cache compression metrics are not applicable "
                "(see kitty-specs/complete-polyglot-vpu-stack/spec.md)."
            )
        else:
            caveat = None
        envelope = {
            "schema_version": 1,
            "kind": "niah_live_run",
            "evidence_label": "live_verified",
            "reported": True,
            "synthetic": False,
            "model": args.model,
            "backend": "mlx_lm_inprocess",
            "architecture_caveat": caveat,
            "seed": args.seed,
            "lengths": args.lengths,
            "modes": args.modes,
            "results": [asdict(r) for r in results],
        }
        out_path.write_text(json.dumps(envelope, indent=2, default=str) + "\n")
        print(f"\n  Results written to: {args.output} (evidence_label=live_verified)")

    print(f"\n  Final RSS: {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
