#!/usr/bin/env python3
"""
NIAH (Needle-In-A-Haystack) benchmark for phenotype-omlx TurboQuant+ MLX.

DEPRECATED for operator acceptance — prefer Portage/Harbor NIAH API smoke:

    bash scripts/evals/run_via_harbor.sh --niah

Keep this script for TurboQuant+ KV-mode matrices on Metal hosts only.
New operator surfaces belong under ``evals/harbor/`` + LangSmith plugin.

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

FR-5 E3 instrumentation: TurboKV metrics on full-attention layers only —
packed_state_present, kv_packed_bytes, kv_turbo_resident_bytes vs
kv_fp16_baseline_bytes, and byte_reduction_effective. Linear-attention
layers keep native caches and are excluded from compression claims.
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# HF home stays stable; online/offline is decided after --model is known
# (see configure_hf_env). Do not force HF_HUB_OFFLINE=0 at import time —
# absolute/local model paths must stay offline-capable.
os.environ.setdefault("HF_HOME", "/Users/kooshapari/.cache/huggingface")


def configure_hf_env(model: str) -> None:
    """Prefer offline Hub access when --model is a local filesystem path.

    Hub repo ids still clear offline flags so a missing local cache can fetch.
    """
    path = Path(model).expanduser()
    if path.is_absolute() or path.exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

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


def build_needle_prompt_ids(tokenizer, target_len: int, needle: str) -> list[int]:
    """Build exactly ``target_len`` tokenizer tokens with the needle at 75% depth.

    Text-length heuristics are not valid for NIAH because tokenizer compression varies
    by model. Constructing the prompt in token space makes the requested context length
    an invariant of the benchmark rather than an estimate.
    """
    intro = "Read the following passage and recall the critical fact mentioned.\n\n"
    needle_text = f"Important context: {needle}. This fact is critical."
    intro_ids = list(tokenizer.encode(intro))
    needle_ids = list(tokenizer.encode(needle_text))
    filler_unit = list(tokenizer.encode(" the"))
    fixed_len = len(intro_ids) + len(needle_ids)
    if not filler_unit:
        raise ValueError("tokenizer produced no filler token")
    if target_len <= fixed_len:
        raise ValueError(
            f"target_len={target_len} is too short for prompt framing ({fixed_len} tokens)"
        )

    filler_len = target_len - fixed_len
    repeated = (filler_unit * ((filler_len // len(filler_unit)) + 1))[:filler_len]
    before_len = (filler_len * 3) // 4
    return intro_ids + repeated[:before_len] + needle_ids + repeated[before_len:]


def _arr_nbytes(value) -> int:
    """Safe nbytes for mlx arrays / None."""
    if value is None:
        return 0
    return int(getattr(value, "nbytes", 0) or 0)


def nested_nbytes(obj) -> int:
    """Sum nbytes across mlx arrays nested in lists/tuples/cache.state."""
    if obj is None:
        return 0
    if hasattr(obj, "nbytes") and not isinstance(obj, (str, bytes)):
        try:
            return int(obj.nbytes)
        except Exception:
            return 0
    if isinstance(obj, (list, tuple)):
        return sum(nested_nbytes(x) for x in obj)
    return 0


def is_turbo_kv_cache(cache) -> bool:
    """True for full TurboKVCache (not Lite / not mlx_lm KVCache)."""
    return type(cache).__name__ == "TurboKVCache"


def full_attention_indices(model) -> list[int]:
    """Indices of layers that are not linear-attention (TurboKV-applicable)."""
    return [
        i
        for i, layer in enumerate(model.layers)
        if not getattr(layer, "is_linear", False)
    ]


def cache_entry_nbytes(cache) -> int:
    """Resident bytes for one cache object (TurboKV.nbytes or state walk)."""
    if is_turbo_kv_cache(cache) and hasattr(cache, "nbytes"):
        try:
            return int(cache.nbytes)
        except Exception:
            pass
    n = nested_nbytes(getattr(cache, "state", None))
    for attr in ("keys", "values", "_keys", "_values"):
        n += nested_nbytes(getattr(cache, attr, None))
    inner = getattr(cache, "_inner_kv", None)
    if inner is not None and inner is not cache:
        n += cache_entry_nbytes(inner)
    return n


def measure_fp16_full_attn_bytes(cache_list, model) -> int:
    """FP16 (or native) resident KV bytes on full-attention layers only."""
    total = 0
    for i in full_attention_indices(model):
        if i < len(cache_list):
            total += cache_entry_nbytes(cache_list[i])
    return total


def layer_turbo_bytes(cache) -> tuple[int, int]:
    """Return ``(transient_raw_bytes, packed_bytes)`` for one TurboKVCache."""
    raw = (
        _arr_nbytes(getattr(cache, "_raw_keys", None))
        + _arr_nbytes(getattr(cache, "_raw_values", None))
    )
    for arr in getattr(cache, "_pending_raw_keys", None) or []:
        raw += _arr_nbytes(arr)
    for arr in getattr(cache, "_pending_raw_values", None) or []:
        raw += _arr_nbytes(arr)
    packed = (
        _arr_nbytes(getattr(cache, "_packed_keys", None))
        + _arr_nbytes(getattr(cache, "_packed_values", None))
        + _arr_nbytes(getattr(cache, "_key_norms", None))
        + _arr_nbytes(getattr(cache, "_value_norms", None))
    )
    return raw, packed


def layer_is_compressed(cache) -> bool:
    """A layer counts as compressed when TurboKV has packed storage or the flag."""
    if bool(getattr(cache, "_is_compressed", False)):
        return True
    _, packed = layer_turbo_bytes(cache)
    return packed > 0


def measure_turbo_cache_metrics(
    cache_list,
    fp16_baseline_bytes: int = 0,
) -> dict:
    """Aggregate TurboKV metrics for full-attention (TurboKV-installed) layers only.

    Proves packed-state presence and byte-reduction vs an FP16 baseline measured
    on the same full-attention layer set (not RSS deltas).
    """
    turbo = [c for c in cache_list if is_turbo_kv_cache(c)]
    kv_raw = 0
    kv_packed = 0
    kv_inner = 0
    kv_resident = 0
    compressed = 0
    for c in turbo:
        raw, packed = layer_turbo_bytes(c)
        kv_raw += raw
        kv_packed += packed
        inner = getattr(c, "_inner_kv", None)
        if inner is not None:
            kv_inner += cache_entry_nbytes(inner)
        kv_resident += cache_entry_nbytes(c)
        if layer_is_compressed(c):
            compressed += 1
    packed_state_present = compressed > 0 and kv_packed > 0
    ratio = None
    byte_reduction_effective = False
    if fp16_baseline_bytes > 0 and kv_resident > 0:
        ratio = kv_resident / float(fp16_baseline_bytes)
        byte_reduction_effective = (
            packed_state_present and kv_resident < fp16_baseline_bytes
        )
    return {
        "turbo_layers": len(turbo),
        "attention_layers": len(turbo),
        "compressed_layers": compressed,
        "packed_state_present": packed_state_present,
        "kv_raw_bytes": kv_raw,
        "kv_packed_bytes": kv_packed,
        "kv_inner_fp16_bytes": kv_inner,
        "kv_turbo_resident_bytes": kv_resident,
        "kv_fp16_baseline_bytes": int(fp16_baseline_bytes),
        "byte_reduction_ratio": ratio,
        "byte_reduction_effective": byte_reduction_effective,
        # Alias: only true when byte reduction vs FP16 baseline is proven.
        "compression_effective": byte_reduction_effective,
    }


def maybe_materialize_turbo_compression(cache_list) -> int:
    """Force first-decode compression on TurboKV layers still raw after prefill."""
    n = 0
    for c in cache_list:
        if not is_turbo_kv_cache(c):
            continue
        if getattr(c, "_is_compressed", False):
            continue
        flush = getattr(c, "_flush_pending", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                pass
        compress = getattr(c, "_compress_raw_cache", None)
        offset = int(getattr(c, "offset", 0) or 0)
        if callable(compress) and offset > 0:
            try:
                compress()
                if layer_is_compressed(c):
                    n += 1
            except Exception:
                pass
    return n


def _make_turbo_cache(bits: int, key_bits: int | None):
    """Construct TurboKVCache with a low compress threshold for NIAH lengths."""
    from mlx.nn.layers.turbo_kv_cache import TurboKVCache

    return TurboKVCache(bits=bits, key_bits=key_bits, min_compress_tokens=64)


def _empty_metrics() -> dict:
    return {
        "turbo_layers": 0,
        "attention_layers": 0,
        "compressed_layers": 0,
        "packed_state_present": False,
        "kv_raw_bytes": 0,
        "kv_packed_bytes": 0,
        "kv_inner_fp16_bytes": 0,
        "kv_turbo_resident_bytes": 0,
        "kv_fp16_baseline_bytes": 0,
        "byte_reduction_ratio": None,
        "byte_reduction_effective": False,
        "compression_effective": False,
    }


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
    turbo_layers: int = 0
    attention_layers: int = 0
    compressed_layers: int = 0
    packed_state_present: bool = False
    kv_raw_bytes: int = 0
    kv_packed_bytes: int = 0
    kv_inner_fp16_bytes: int = 0
    kv_turbo_resident_bytes: int = 0
    kv_fp16_baseline_bytes: int = 0
    byte_reduction_ratio: float | None = None
    byte_reduction_effective: bool = False
    compression_effective: bool = False


def run_one(
    model,
    tokenizer,
    mode: str,
    length: int,
    needle: str,
    fp16_baseline_bytes: int = 0,
) -> BenchResult:
    """Run one (mode, length) benchmark. Returns a BenchResult."""
    mx, mlx_lm = _ensure_mlx()
    rss_before = rss_mb()

    prompt_ids = build_needle_prompt_ids(tokenizer, length, needle)
    actual_len = len(prompt_ids)
    print(f"  [{mode:18s}] {length:5d} target / {actual_len:5d} actual tokens")

    n_layers = len(model.layers)
    is_qwen35 = "qwen3_5" in type(model).__module__ or "qwen3_5" in str(
        getattr(model, "model_type", "")
    )
    from mlx_lm.models import cache

    cache_list = cache.make_prompt_cache(model)
    if mode == "baseline_fp16":
        pass
    elif is_qwen35:
        for i, layer in enumerate(model.layers):
            if getattr(layer, "is_linear", False):
                continue
            if mode in ("turbo_asymmetric", "turbo4"):
                cache_list[i] = _make_turbo_cache(bits=4, key_bits=None)
            elif mode == "turbo_symmetric":
                cache_list[i] = _make_turbo_cache(bits=4, key_bits=4)
            elif mode == "mlx_native_kv4":
                from mlx_lm.models.cache import QuantizedKVCache

                cache_list[i] = QuantizedKVCache(group_size=64, bits=4)
            else:
                raise ValueError(f"unknown mode: {mode}")
    elif mode in ("turbo_asymmetric", "turbo4"):
        cache_list = [_make_turbo_cache(bits=4, key_bits=None) for _ in range(n_layers)]
    elif mode == "turbo_symmetric":
        cache_list = [_make_turbo_cache(bits=4, key_bits=4) for _ in range(n_layers)]
    elif mode == "mlx_native_kv4":
        from mlx_lm.models.cache import QuantizedKVCache

        cache_list = [QuantizedKVCache(group_size=64, bits=4) for _ in range(n_layers)]
    else:
        raise ValueError(f"unknown mode: {mode}")

    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)

    t0 = time.perf_counter()
    try:
        mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt_ids,
            max_tokens=1,
            prompt_cache=cache_list,
            verbose=False,
            sampler=sampler,
        )
        prefill_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        return BenchResult(
            mode=mode,
            target_len=length,
            actual_len=actual_len,
            prefill_ms=0,
            decode_ms=0,
            decode_tok_per_sec=0,
            rss_mb_before=rss_before,
            rss_mb_after=rss_mb(),
            needle=needle,
            answer="",
            exact_match=False,
            partial_match=False,
            contains_secret=False,
            error=f"prefill: {type(e).__name__}: {e}",
        )

    metrics = _empty_metrics()
    if mode == "baseline_fp16":
        fp16_bytes = measure_fp16_full_attn_bytes(cache_list, model)
        metrics["kv_fp16_baseline_bytes"] = fp16_bytes
        metrics["attention_layers"] = len(full_attention_indices(model))
        print(
            f"    fp16 full-attn baseline bytes={fp16_bytes} "
            f"layers={metrics['attention_layers']}/{n_layers}"
        )
    elif mode in ("turbo_asymmetric", "turbo_symmetric", "turbo4"):
        forced = maybe_materialize_turbo_compression(cache_list)
        metrics = measure_turbo_cache_metrics(
            cache_list, fp16_baseline_bytes=fp16_baseline_bytes
        )
        print(
            f"    turbo metrics: turbo_layers={metrics['turbo_layers']}/{n_layers} "
            f"compressed={metrics['compressed_layers']} "
            f"packed_state={metrics['packed_state_present']} "
            f"forced_materialize={forced} "
            f"packed={metrics['kv_packed_bytes']} "
            f"resident={metrics['kv_turbo_resident_bytes']} "
            f"fp16_baseline={metrics['kv_fp16_baseline_bytes']} "
            f"ratio={metrics['byte_reduction_ratio']} "
            f"byte_reduction={metrics['byte_reduction_effective']}"
        )
        print("    cache types:", ", ".join(type(c).__name__ for c in cache_list))

    qa_prompt = (
        "\n\nQuestion: What is the critical fact mentioned in the passage above? "
        "Do not explain or think aloud. Respond with only the exact critical fact "
        "and nothing else.\n\nAnswer:"
    )
    qa_ids = tokenizer.encode(qa_prompt)

    t0 = time.perf_counter()
    try:
        answer_ids = mlx_lm.generate(
            model,
            tokenizer,
            prompt=qa_ids,
            max_tokens=128,
            prompt_cache=cache_list,
            verbose=False,
            sampler=sampler,
        )
        decode_ms = (time.perf_counter() - t0) * 1000.0
        n_tok = len(answer_ids) if isinstance(answer_ids, list) else 1
        decode_tps = n_tok / (decode_ms / 1000.0) if decode_ms > 0 else 0
    except Exception as e:
        return BenchResult(
            mode=mode,
            target_len=length,
            actual_len=actual_len,
            prefill_ms=prefill_ms,
            decode_ms=0,
            decode_tok_per_sec=0,
            rss_mb_before=rss_before,
            rss_mb_after=rss_mb(),
            needle=needle,
            answer="",
            exact_match=False,
            partial_match=False,
            contains_secret=False,
            error=f"decode: {type(e).__name__}: {e}",
            **metrics,
        )

    if mode == "baseline_fp16":
        metrics["kv_fp16_baseline_bytes"] = measure_fp16_full_attn_bytes(
            cache_list, model
        )
    elif mode in ("turbo_asymmetric", "turbo_symmetric", "turbo4"):
        metrics = measure_turbo_cache_metrics(
            cache_list, fp16_baseline_bytes=fp16_baseline_bytes
        )

    if isinstance(answer_ids, list):
        answer = tokenizer.decode(answer_ids)
    else:
        answer = str(answer_ids)

    secret = (
        needle.split("the secret code is ")[1]
        if "the secret code is " in needle
        else needle
    )
    exact = needle.strip() in answer
    partial = secret in answer
    contains_secret = secret in answer

    return BenchResult(
        mode=mode,
        target_len=length,
        actual_len=actual_len,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        decode_tok_per_sec=decode_tps,
        rss_mb_before=rss_before,
        rss_mb_after=rss_mb(),
        needle=needle,
        answer=answer,
        exact_match=exact,
        partial_match=partial,
        contains_secret=contains_secret,
        **metrics,
    )


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
    parser.add_argument("--model", type=str, default=None,
                        help="MLX model (default: config/smoke_models.json role=niah)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write results JSON to this file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for needle/filler")
    args = parser.parse_args()
    if not args.model:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "python"))
        from omlx_research.smoke_models import default_model_for
        args.model = default_model_for("niah")
    configure_hf_env(args.model)

    require_julia()
    random.seed(args.seed)

    print("=" * 70)
    print("  NIAH BENCHMARK — phenotype-omlx TurboQuant+ MLX")
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
    fp16_baseline_by_len: dict[int, int] = {}

    for length in args.lengths:
        print(f"\n{'─' * 70}")
        print(f"  CONTEXT LENGTH: {length} tokens")
        print(f"{'─' * 70}")
        needle = f"the secret code is {random.randint(100, 999)}-{random.randint(100, 999)}-alpha"

        # Prefer baseline first so turbo rows get a real FP16 baseline.
        modes = list(args.modes)
        if "baseline_fp16" in modes:
            modes = ["baseline_fp16"] + [m for m in modes if m != "baseline_fp16"]

        for mode in modes:
            print(f"\n[{mode}]")
            gc.collect()
            try:
                r = run_one(
                    model,
                    tokenizer,
                    mode,
                    length,
                    needle,
                    fp16_baseline_bytes=fp16_baseline_by_len.get(length, 0),
                )
                if mode == "baseline_fp16" and not r.error and r.kv_fp16_baseline_bytes:
                    fp16_baseline_by_len[length] = r.kv_fp16_baseline_bytes
                results.append(r)
                if r.error:
                    print(f"  ✗ {r.error}")
                else:
                    marker = "✓" if r.exact_match else ("≈" if r.partial_match else "✗")
                    print(
                        f"  {marker} prefill={r.prefill_ms:.0f}ms "
                        f"decode={r.decode_ms:.0f}ms ({r.decode_tok_per_sec:.1f} tok/s)"
                    )
                    print(f"  RSS Δ: {r.rss_mb_after - r.rss_mb_before:+.0f} MB")
                    if r.turbo_layers:
                        print(
                            f"  turbo: compressed={r.compressed_layers}/{r.turbo_layers} "
                            f"packed={r.kv_packed_bytes} resident={r.kv_turbo_resident_bytes} "
                            f"fp16={r.kv_fp16_baseline_bytes} "
                            f"ratio={r.byte_reduction_ratio} "
                            f"byte_reduction={r.byte_reduction_effective}"
                        )
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

    print()
    print("=" * 70)
    print("  NIAH BENCHMARK RESULTS")
    print("=" * 70)
    print()
    print(
        f"  {'Mode':22s} {'Len':>6s} {'Tok/s':>8s} {'RSSΔ(MB)':>10s} "
        f"{'Cmp':>7s} {'Reduc':>6s} {'Match':>6s}"
    )
    print(f"  {'─' * 22} {'─' * 6} {'─' * 8} {'─' * 10} {'─' * 7} {'─' * 6} {'─' * 6}")
    for r in results:
        marker = "✓" if r.exact_match else ("≈" if r.partial_match else ("✗" if r.error else " "))
        rss_d = r.rss_mb_after - r.rss_mb_before
        cmp = f"{r.compressed_layers}/{r.turbo_layers}" if r.turbo_layers else "-"
        reduc = "yes" if r.byte_reduction_effective else ("-" if not r.turbo_layers else "no")
        print(
            f"  {r.mode:22s} {r.target_len:>6d} {r.decode_tok_per_sec:>8.1f} "
            f"{rss_d:>+10.0f} {cmp:>7s} {reduc:>6s} {marker:>6s}"
        )

    print()
    print("  Quality summary:")
    for m in sorted(set(r.mode for r in results)):
        mr = [r for r in results if r.mode == m and not r.error]
        exact = sum(1 for r in mr if r.exact_match)
        partial = sum(1 for r in mr if r.partial_match and not r.exact_match)
        miss = sum(1 for r in mr if not r.exact_match and not r.partial_match)
        print(f"    {m:22s}: exact={exact}/{len(mr)} partial={partial} miss={miss}")

    turbo_rows = [r for r in results if r.turbo_layers and not r.error]
    if turbo_rows:
        print()
        print(
            f"  Compression summary: packed_state_any="
            f"{any(r.packed_state_present for r in turbo_rows)} "
            f"byte_reduction_any={any(r.byte_reduction_effective for r in turbo_rows)} "
            f"max_compressed_layers={max(r.compressed_layers for r in turbo_rows)}"
        )

    if args.output:
        out_path = Path(args.output)
        if "qwen3.5" in args.model.lower() or "qwen3_5" in args.model.lower():
            caveat = (
                "Qwen3.5 may mix linear + full attention. TurboKV metrics "
                "(packed_state, resident vs fp16 baseline) apply only to "
                "full-attention layers where TurboKVCache is installed."
            )
        else:
            caveat = None
        envelope = {
            "schema_version": 2,
            "kind": "niah_live_run",
            "evidence_label": "live_verified",
            "reported": True,
            "synthetic": False,
            "model": args.model,
            "backend": "mlx_lm_inprocess",
            "architecture_caveat": caveat,
            "kv_modes_applicable": bool(turbo_rows),
            "packed_state_any": (
                any(r.packed_state_present for r in turbo_rows) if turbo_rows else False
            ),
            "byte_reduction_any": (
                any(r.byte_reduction_effective for r in turbo_rows) if turbo_rows else False
            ),
            "compression_any_effective": (
                any(r.compression_effective for r in turbo_rows) if turbo_rows else False
            ),
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
