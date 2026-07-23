"""KV-cache memory ceiling measurement for Qwen models."""

import json

QWEN_0_8B = {
    "model": "Qwen3.5-0.8B",
    "hidden_size": 2048,
    "num_layers": 28,
    "num_kv_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "dtype_bytes": 2,  # fp16
}

QWEN_4B = {
    "model": "Qwen3.5-4B-Coder",
    "hidden_size": 2560,
    "num_layers": 36,
    "num_kv_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "dtype_bytes": 2,
}


def compute_kv_cache_mb(seq_len: int, config: dict) -> float:
    """Compute KV-cache memory for a given sequence length."""
    kv_per_layer = (
        2 * config["num_kv_heads"] * config["head_dim"] * config["dtype_bytes"]
    )
    total_bytes = kv_per_layer * config["num_layers"] * seq_len
    return total_bytes / (1024 * 1024)


if __name__ == "__main__":
    results = []
    for ctx_len in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        for config in [QWEN_0_8B, QWEN_4B]:
            mb = compute_kv_cache_mb(ctx_len, config)
            results.append(
                {
                    "model": config["model"],
                    "context_length": ctx_len,
                    "kv_cache_mb": round(mb, 2),
                    "num_layers": config["num_layers"],
                }
            )

    print(f"{'Model':<20} {'Context':>8} {'KV Cache':>10}")
    print("-" * 42)
    for r in results:
        print(f"{r['model']:<20} {r['context_length']:>8} {r['kv_cache_mb']:>8.2f}MB")

    with open("python/omlx_research/benchmarks/kv_cache_memory.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"\nPeak at 32K ctx: {compute_kv_cache_mb(32768, QWEN_4B):.2f} MB for 4B model"
    )
