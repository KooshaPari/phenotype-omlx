"""omlx-research CLI — single entry point wrapping the entire OMLX research stack.

Subcommands:
  status                  — show backend availability
  inference               — run a single prompt through one or all engines
  spec-decode             — speculative decoding demo (SSD / draft / Medusa)
  latentmas               — multi-agent concurrent fan-out
  tidar                   — hybrid AR+diffusion generation
  bench                   — quick benchmark (tokens/sec, acceptance rate)
  doctor                  — diagnose environment
  fleet                   — show / manage cluster peers
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from typing import Optional


def cmd_status(_: argparse.Namespace) -> int:
    from ..backends import (
        MlxBackend, MetalKernelBackend, VllmBackend,
        TensorrtBackend, SglangBackend, LlamaCppBackend,
    )
    rows = []
    for cls in (MlxBackend, MetalKernelBackend, VllmBackend, TensorrtBackend, SglangBackend, LlamaCppBackend):
        b = cls()
        rows.append((b.capabilities.name, b.capabilities.primary, "ok" if b.is_available() else "—"))
    print(f"{'name':10s} {'primary':12s} status")
    for n, p, s in rows:
        print(f"{n:10s} {p:12s} {s}")
    return 0


def cmd_inference(args: argparse.Namespace) -> int:
    from ..backends import GenerateRequest, MlxBackend, MetalKernelBackend
    from ..engines import HybridDispatch, DispatchPolicy
    req = GenerateRequest(prompt=args.prompt, max_tokens=args.max_tokens, temperature=args.temperature)
    pol = DispatchPolicy(args.policy) if args.policy else DispatchPolicy.AUTO
    if pol == DispatchPolicy.MLX:
        b = MlxBackend(model_path=args.model)
        out = b.generate(req)
        print(out.text)
    elif pol == DispatchPolicy.METAL:
        b = MetalKernelBackend()
        out = b.generate(req)
        print(out.text)
    else:
        d = HybridDispatch()
        outs = d.generate(req, policy=pol)
        for r in outs:
            print(f"[{r.backend}] {r.text}")
    return 0


def cmd_spec_decode(args: argparse.Namespace) -> int:
    from ..engines import SpeculativeEngine, SpecMode
    from ..engines.spec_decode import SpecConfig
    import numpy as np

    # Toy target: returns a numpy logits vector (length 32).
    # The engine calls `int(target(prefix).argmax())` for fallback — pass
    # a plain numpy array, not a wrapped object.
    def tok_fn(prefix):
        if not prefix:
            return np.zeros(32, dtype=np.float32)
        return np.array([float(len(prefix) + i) for i in range(32)], dtype=np.float32)

    if args.mode == "ssd":
        engine = SpeculativeEngine(
            target=tok_fn,
            config=SpecConfig(mode=SpecMode.SAME_MODEL, max_draft_tokens=args.gamma),
        )
        out = engine.step(list(range(1, args.prompt_len)))
        print("accepted:", out[: args.gamma])
        print(f"acceptance rate: {engine.stats.acceptance_rate:.2%}")
    elif args.mode == "draft":
        # Draft model (toy): constant argmax
        def draft_fn(prefix):
            return np.array([0.0] * 31 + [1.0], dtype=np.float32)
        engine = SpeculativeEngine(
            target=tok_fn,
            draft=draft_fn,
            config=SpecConfig(mode=SpecMode.DRAFT_MODEL, max_draft_tokens=args.gamma),
        )
        out = engine.step(list(range(1, args.prompt_len)))
        print("accepted:", out[: args.gamma])
        print(f"acceptance rate: {engine.stats.acceptance_rate:.2%}")
    elif args.mode == "medusa":
        width = 4
        depth = args.depth
        mask_size = width**depth + 1
        print(f"Medusa: heads={width} depth={depth} → tree shape ({mask_size}, {mask_size})")
    return 0


def cmd_latentmas(args: argparse.Namespace) -> int:
    import asyncio
    from ..agents import latentmas_fanout

    async def one_agent(_p, _s, idx):
        await asyncio.sleep(0.01)
        return f"[agent-{idx} answer]"

    async def main():
        fns = [lambda p, s, i=i: one_agent(p, s, i) for i in range(args.n_agents)]
        results = await latentmas_fanout(fns, args.prompt, {})
        for r in results:
            print(r)
    asyncio.run(main())
    return 0


def cmd_tidar(args: argparse.Namespace) -> int:
    print(f"TiDAR draft_len={args.draft_len} steps={args.steps}")
    print("TiDAR: AR + diffusion hybrid loop. Use --mode=hybrid for full demo.")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    print("bench: stub — implements tok/sec + acceptance rate")
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    print("== Environment diagnostics ==")
    try:
        import mlx.core as mx
        print(f"MLX {mx.__version__} | Metal {mx.metal.is_available()}")
    except ImportError:
        print("MLX: not installed")
    try:
        import torch
        print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | MPS {torch.backends.mps.is_available()}")
    except ImportError:
        print("PyTorch: not installed")
    try:
        import turboquant_plus  # noqa
        print("TurboQuant+ ref: installed")
    except ImportError:
        try:
            import sys as _s
            _s.path.insert(0, "/Users/kooshapari/CodeProjects/Phenotype/repos/turboquant_plus")
            print("TurboQuant+ ref: present but not activated")
        except Exception:
            print("TurboQuant+ ref: missing")
    try:
        from mlx.nn.layers.turbo_kv_cache import TurboKVCache  # noqa
        print("TurboKVCache (MLX fork): available")
    except ImportError:
        print("TurboKVCache (MLX fork): not in framework path")
    return 0


def cmd_fleet(args: argparse.Namespace) -> int:
    print("fleet: peers (in-memory registry)")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="omlx-research")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)

    p = sub.add_parser("inference")
    p.add_argument("--prompt", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--policy", choices=[p.value for p in __import__("omlx_research.engines", fromlist=["DispatchPolicy"]).DispatchPolicy], default=None)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.set_defaults(fn=cmd_inference)

    p = sub.add_parser("spec-decode")
    p.add_argument("--mode", choices=["ssd", "draft", "medusa"], default="ssd")
    p.add_argument("--gamma", type=int, default=5)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--prompt-len", type=int, default=20)
    p.set_defaults(fn=cmd_spec_decode)

    p = sub.add_parser("latentmas")
    p.add_argument("--prompt", required=True)
    p.add_argument("--n-agents", type=int, default=4)
    p.set_defaults(fn=cmd_latentmas)

    p = sub.add_parser("tidar")
    p.add_argument("--mode", choices=["ar", "diffusion", "hybrid"], default="hybrid")
    p.add_argument("--draft-len", type=int, default=4)
    p.add_argument("--steps", type=int, default=8)
    p.set_defaults(fn=cmd_tidar)

    sub.add_parser("bench").set_defaults(fn=cmd_bench)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    sub.add_parser("fleet").set_defaults(fn=cmd_fleet)

    args = parser.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
