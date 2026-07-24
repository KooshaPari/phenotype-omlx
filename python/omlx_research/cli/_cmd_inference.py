"""``omlx-research`` inference / compute subcommand implementations.

Moved out of ``cli/__init__.py`` in this module-size sweep (turn-N).
The previous turn-9 sweep carved out the eval-harness subcommand
(``_cmd_eval.py``); this sweep extracts the inference cluster — the
remaining non-trivial inline block in ``cli/__init__.py``.

Responsibility seam: every function in this module owns a slice of the
inference / compute / status surface that the top-level CLI dispatches
into. They share the same per-call lazy-import pattern (each pulls its
backend/engine dependencies inside the function body) so the CLI
startup cost is not affected by importing this module.

Public contract (re-exported by ``cli/__init__.py`` so existing
importers continue to work):

- :func:`cmd_status` — backend-availability snapshot (``status`` subcommand)
- :func:`cmd_inference` — single-prompt generation through one/all engines
- :func:`cmd_spec_decode` — speculative-decoding toy demo (SSD / draft / Medusa)
- :func:`cmd_latentmas` — multi-agent concurrent fan-out

Anything these functions reference that lives in a sibling module
(``_missing_dep.require_mlx_lm``, ``backends``, ``engines``, ``agents``)
stays where it is; this module only owns the four CLI handlers and
their per-call import glue.
"""

from __future__ import annotations

import argparse


def cmd_status(_: argparse.Namespace) -> int:
    from ..backends import (
        MlxBackend,
        MetalKernelBackend,
        VllmBackend,
        TensorrtBackend,
        SglangBackend,
        LlamaCppBackend,
    )

    rows = []
    for cls in (
        MlxBackend,
        MetalKernelBackend,
        VllmBackend,
        TensorrtBackend,
        SglangBackend,
        LlamaCppBackend,
    ):
        b = cls()
        rows.append(
            (
                b.capabilities.name,
                b.capabilities.primary,
                "ok" if b.is_available() else "—",
            )
        )
    print(f"{'name':10s} {'primary':12s} status")
    for n, p, s in rows:
        print(f"{n:10s} {p:12s} {s}")
    return 0


def cmd_inference(args: argparse.Namespace) -> int:
    from ..backends import GenerateRequest, MlxBackend, MetalKernelBackend
    from ..engines import HybridDispatch, DispatchPolicy

    req = GenerateRequest(
        prompt=args.prompt, max_tokens=args.max_tokens, temperature=args.temperature
    )
    pol = DispatchPolicy(args.policy) if args.policy else DispatchPolicy.AUTO
    if pol == DispatchPolicy.MLX:
        # Fail loudly with a structured install hint instead of a bare
        # ImportError when mlx_lm is missing — this is the production
        # decode path on Apple Silicon and the most common DX paper cut.
        from ._missing_dep import require_mlx_lm

        require_mlx_lm("omlx-research inference --policy mlx")
        b = MlxBackend(model_path=args.model)
        out = b.generate(req)
        print(out.text)
    elif pol == DispatchPolicy.METAL:
        b = MetalKernelBackend(model_path=args.model)
        out = b.generate(req)
        print(out.text)
    else:
        # AUTO / hybrid: mlx_lm may be required by whichever branch the
        # hybrid dispatcher picks; require it up front so the user sees
        # the structured install message before any partial work runs.
        from ._missing_dep import require_mlx_lm

        require_mlx_lm("omlx-research inference")
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
        print(
            f"Medusa: heads={width} depth={depth} → tree shape ({mask_size}, {mask_size})"
        )
    return 0


def cmd_latentmas(args: argparse.Namespace) -> int:
    import asyncio
    from ..agents import latentmas_fanout

    async def one_agent(_p, _s, idx):
        await asyncio.sleep(0.01)
        return f"[agent-{idx} answer]"

    async def main():
        def _make_agent(idx):
            async def agent(p, s):
                return await one_agent(p, s, idx)

            return agent

        fns = [_make_agent(i) for i in range(args.n_agents)]
        results = await latentmas_fanout(fns, args.prompt, {})
        for r in results:
            print(r)

    asyncio.run(main())
    return 0
