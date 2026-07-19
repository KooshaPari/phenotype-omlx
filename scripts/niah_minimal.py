#!/usr/bin/env python3
"""Minimal NIAH (Needle-In-A-Haystack) smoke test.

Verifies the pipeline works end-to-end against a small Qwen2.5-0.5B MLX model
without requiring the custom TurboKVCache module (which lives in
phenotype-omlx/mlx.nn.layers.turbo_kv_cache and is not installed).

Outputs retrieval accuracy, prefill/decode timing, and RSS memory delta.
Qwen3.5 is not yet on mlx-community in a quantized form as of this writing,
so we use Qwen2.5-0.5B-Instruct-4bit (mlx-community) as a substitute.
"""

import os, sys, time, gc, json, random, argparse
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HOME"] = "/Users/kooshapari/.cache/huggingface"

import psutil
import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import KVCache


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    print(f"NIAH minimal: model={args.model}, length={args.length}", flush=True)
    print(f"RSS before model load: {rss_mb():.0f} MB", flush=True)

    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(args.model)
    load_ms = (time.perf_counter() - t0) * 1000.0
    n_layers = len(model.layers)
    print(f"Model loaded in {load_ms:.0f}ms; {n_layers} layers; RSS={rss_mb():.0f} MB", flush=True)

    needle = f"the secret code is {random.randint(100, 999)}-{random.randint(100, 999)}-alpha"
    vocab = ["the", "a", "of", "in", "to", "and", "for", "with", "on", "lorem",
             "ipsum", "dolor", "sit", "amet", "consectetur", "phenotype", "machine",
             "learning", "model", "rust", "python"]
    filler = " ".join(random.choices(vocab, k=args.length // 6))
    intro = "Read the passage and recall the critical fact.\n\n"
    needle_para = f"\n\nImportant: {needle}. This is critical.\n\n"
    cut = len(filler) * 3 // 4
    prompt = intro + filler[:cut] + needle_para + filler[cut:]
    prompt_ids = tokenizer.encode(prompt)
    print(f"Prompt: {len(prompt_ids)} tokens", flush=True)

    cache = [KVCache() for _ in range(n_layers)]
    rss_before = rss_mb()

    t0 = time.perf_counter()
    mlx_lm.generate(model, tokenizer, prompt=prompt_ids, max_tokens=1,
                   prompt_cache=cache, verbose=False)
    prefill_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Prefill: {prefill_ms:.0f}ms", flush=True)

    qa = "\n\nQuestion: What is the critical fact? Respond concisely.\n\nAnswer:"
    qa_ids = tokenizer.encode(qa)
    t0 = time.perf_counter()
    answer_chunks = list(mlx_lm.generate(model, tokenizer, prompt=qa_ids, max_tokens=args.max_tokens,
                                  prompt_cache=cache, verbose=False))
    decode_ms = (time.perf_counter() - t0) * 1000.0
    if answer_chunks and isinstance(answer_chunks[0], str):
        answer = "".join(answer_chunks)
        n_tok = sum(len(tokenizer.encode(c, add_special_tokens=False)) for c in answer_chunks)
    else:
        answer = tokenizer.decode(answer_chunks)
        n_tok = len(answer_chunks)
    secret = needle.split("the secret code is ")[1]
    exact = needle.strip() in answer
    partial = secret in answer

    rss_after = rss_mb()
    result = {
        "model": args.model,
        "context_len": len(prompt_ids),
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "decode_tok_per_sec": n_tok / (decode_ms / 1000.0) if decode_ms > 0 else 0,
        "rss_mb_delta": rss_after - rss_before,
        "needle": needle,
        "answer": answer[:200],
        "exact_match": exact,
        "partial_match": partial,
        "limitations": [
            "Qwen3.5 not used: mlx-community has no Qwen3.5 quantized variant as of 2026-07-19",
            "TurboKVCache from mlx.nn.layers.turbo_kv_cache not installed in this env",
            "Single context length only; full NIAH sweeps require the custom TurboKVCache module",
        ],
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
