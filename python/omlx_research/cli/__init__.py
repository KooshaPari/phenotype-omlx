"""omlx-research CLI — single entry point wrapping the entire OMLX research stack.

Subcommands:
  status                  — show backend availability
  inference               — run a single prompt through one or all engines
  spec-decode             — speculative decoding demo (SSD / draft / Medusa)
  latentmas               — multi-agent concurrent fan-out
  tidar                   — hybrid AR+diffusion generation
  bench                   — quick benchmark (tokens/sec, acceptance rate)
  eval                    — run an eval-harness suite (mmlu/gpqa/terminal-bench/perplexity)
  doctor [--json]         — diagnose environment (Python, MLX, kernels, ABI, tests)
  fleet                   — show / manage cluster peers
  inspect <plan.json>     — load + validate a model plan; print summary
  explain <op-kind>       — print canonical op contract + kernel candidates
  tune <op-kind>          — produce a synthetic TuningRecord and cache it
  replay <trace-file>     — replay an execution trace in human-readable form
  compare <trace-a> <trace-b> — side-by-side trace comparison (JSON)
  evidence <plan-file>    — generate an evidence bundle (stdout + .json)
  promote <kernel-id>    — validate against --gates, sign and cache PromotionRecord
  quarantine <kernel-id> — append a Hold/Rollback audit-trail entry
  gates <list|add|remove|check> <kernel-id> — CRUD quality-gate configs
"""

from __future__ import annotations
import argparse
import json
import os
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
        # Fail loudly with a structured install hint instead of a bare
        # ImportError when mlx_lm is missing — this is the production
        # decode path on Apple Silicon and the most common DX paper cut.
        from ._missing_dep import require_mlx_lm
        require_mlx_lm("omlx-research inference --policy mlx")
        b = MlxBackend(model_path=args.model)
        out = b.generate(req)
        print(out.text)
    elif pol == DispatchPolicy.METAL:
        b = MetalKernelBackend()
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


# --- eval-harness subcommand -----------------------------------------------
#
# The canonical scoring surface lives in the Rust crate
# ``perf-core/eval-harness/`` (consumed via the kernel-registry). This
# Python wrapper is the CLI entry point that loads a dataset off disk and
# emits a JSON report — the actual scoring is delegated to the Rust
# scorer through the kernel-registry in production. The wrapper is
# deliberately a stub for now: it loads the dataset via Python's csv /
# json modules, counts "passes" against a deterministic heuristic, and
# prints the JSON report. When the kernel-registry binding for the
# eval-harness lands, this is the seam where the Rust scorer slots in.
#
# The suite identifiers mirror ``eval_harness::Suite`` (which uses
# ``#[serde(rename_all = "lowercase")]``) so the on-disk dataset
# identifiers stay byte-compatible with the Rust enum. The list is
# exposed publicly as :data:`EVAL_VALID_SUITES` so tests can pin the
# contract without re-deriving it from argparse's choices.

EVAL_VALID_SUITES: tuple[str, ...] = (
    "mmlu",
    "gpqa",
    "terminal-bench",
    "perplexity",
)


def _eval_load_dataset(suite: str, dataset_path: str) -> list[dict]:
    """Load an eval-harness dataset off disk in the documented shape.

    The Rust loaders (``perf-core/eval-harness/src/{dataset,mmlu,gpqa}.rs``)
    accept CSV for the multiple-choice suites (mmlu, gpqa) and JSONL for
    the open-ended ones (terminal-bench, perplexity). We mirror that
    split here so the wrapper's contract matches what the Rust crate
    expects on disk.

    Returns a list of raw row dicts; the wrapper is a stub so we do not
    validate per-suite schemas — the row count drives the report, and
    the schema check lives in the Rust loader.
    """
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"dataset not found: {dataset_path} (suite={suite})"
        )

    if suite in ("terminal-bench", "perplexity"):
        # JSONL: one JSON object per line.
        rows: list[dict] = []
        with open(dataset_path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"dataset {dataset_path} line {lineno}: "
                        f"invalid JSON: {e}"
                    ) from e
        return rows

    # CSV: dict-reader on the header row.
    import csv as _csv  # local import — keeps startup lean
    with open(dataset_path, "r", encoding="utf-8", newline="") as fh:
        reader = _csv.DictReader(fh)
        return [dict(row) for row in reader]


def _eval_stub_score(rows: list[dict], suite: str) -> int:
    """Deterministic stub: count rows with a non-empty answer field.

    The real scorer lives in Rust; this stub exists so the wrapper has
    an honest end-to-end contract for tests and CI smoke runs. It
    counts the same rows the Rust ``EvalHarness::run`` would mark as
    "answer present" for the multiple-choice suites.

    The ``answer`` field is canonical for the multiple-choice suites
    (mmlu/gpqa); the open-ended suites (terminal-bench/perplexity)
    store the expected completion under ``expected``. The stub
    accepts either so the wrapper's JSON-report contract is uniform
    across all four suites.
    """
    del suite  # unused in the stub; reserved for suite-specific heuristics
    score = 0
    for r in rows:
        answer = str(r.get("answer", "")).strip()
        if not answer:
            answer = str(r.get("expected", "")).strip()
        if answer:
            score += 1
    return score


def cmd_eval(args: argparse.Namespace) -> int:
    """Run an eval-harness suite against a local dataset file.

    Loads the dataset (CSV for multiple-choice, JSONL for open-ended),
    asks the kernel-registry's eval-harness binding to score it (stub
    today: count rows with a non-empty answer), and prints a JSON
    report on stdout. ``--report PATH`` also persists the report to
    disk so it can be archived alongside the run.
    """
    suite: str = args.suite
    dataset_path: str = args.dataset
    backend: str = args.backend
    report_path: Optional[str] = args.report

    try:
        rows = _eval_load_dataset(suite, dataset_path)
    except FileNotFoundError as e:
        # Surface a structured argparse-style error and exit 2 — the
        # standard exit code for CLI usage errors so scripts wrapping
        # the CLI can distinguish "bad input" from "internal failure".
        print(f"omlx-research eval: error: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"omlx-research eval: error: {e}", file=sys.stderr)
        return 2

    tasks = len(rows)
    passed = _eval_stub_score(rows, suite)
    score = (passed / tasks) if tasks else 0.0

    report = {
        "suite": suite,
        "tasks": tasks,
        "passed": passed,
        "score": round(score, 6),
        "backend": backend,
        "harness": "eval-harness (Rust crate via kernel-registry; stub scorer in Python wrapper)",
        "dataset": dataset_path,
        "report_path": report_path,
        "timestamp": time.time(),
    }
    payload = json.dumps(report)
    print(payload)

    if report_path is not None:
        out_dir = os.path.dirname(os.path.abspath(report_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")

    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    # Imported lazily so the heavy doctor check imports (subprocess, mlx,
    # platform) don't pay off when users invoke other subcommands.
    from .doctor import cmd_doctor as _doctor_cmd
    return _doctor_cmd(args)


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

    p = sub.add_parser("bench")
    p.set_defaults(fn=cmd_bench)

    # --------------------------------------------------------------- eval
    p = sub.add_parser(
        "eval",
        help=(
            "run an eval-harness suite (mmlu, gpqa, terminal-bench, "
            "perplexity) against a local dataset file"
        ),
    )
    p.add_argument(
        "--suite", required=True, choices=list(EVAL_VALID_SUITES),
        help=(
            "suite identifier; matches eval_harness::Suite "
            "(mmlu, gpqa, terminal-bench, perplexity)"
        ),
    )
    p.add_argument(
        "--dataset", required=True,
        help="path to a dataset file (CSV for mmlu/gpqa, JSONL for terminal-bench/perplexity)",
    )
    p.add_argument(
        "--backend", default="metal",
        help="backend identifier reported in the JSON envelope (default: metal)",
    )
    p.add_argument(
        "--report", default=None,
        help="optional path; when set, the JSON report is also written to disk",
    )
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser(
        "doctor",
        help="diagnose the runtime environment (Python, MLX, kernels, ABI, tests)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="emit a JSON envelope to stdout instead of the human summary",
    )
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("fleet")
    p.set_defaults(fn=cmd_fleet)

    # ------------------------------------------------------------------ inspect
    from .commands import (
        cmd_compare, cmd_evidence, cmd_explain, cmd_gates,
        cmd_inspect, cmd_promote, cmd_quarantine, cmd_replay, cmd_tune,
    )

    p = sub.add_parser(
        "inspect",
        help="load + validate a model plan JSON file and print a summary",
    )
    p.add_argument("plan", nargs="?", default=None,
                   help="path to a model-plan JSON file (omit when using --empty)")
    p.add_argument("--empty", action="store_true",
                   help="use a built-in 2-operator synthetic plan")
    p.add_argument("--show-states", action="store_true",
                   help="also list each state entry")
    p.add_argument("--show-deps", action="store_true",
                   help="also list each edge entry")
    p.set_defaults(fn=cmd_inspect)

    # ------------------------------------------------------------------ explain
    p = sub.add_parser(
        "explain",
        help="print the canonical contract for an op-kind (+ kernel candidates if --shape given)",
    )
    p.add_argument("op_kind", help="operator kind, e.g. DenseMatmul, RoPE, RMSNorm")
    p.add_argument("--shape", default=None,
                   help="comma-separated shape dims, e.g. 1024,1024,4096")
    p.set_defaults(fn=cmd_explain)

    # ---------------------------------------------------------------------- tune
    p = sub.add_parser(
        "tune",
        help="produce a deterministic synthetic TuningRecord and write it to ~/.cache/omlx/tune/",
    )
    p.add_argument("op_kind", help="operator kind, e.g. DenseMatmul")
    p.add_argument("--shape", default=None,
                   help="comma-separated shape dims, e.g. 1024,1024,4096")
    p.add_argument("--samples", type=int, default=16,
                   help="number of samples to take (default: 16)")
    p.add_argument("--warmup", type=int, default=3,
                   help="number of warmup iterations (default: 3)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the deterministic jitter (default: 0)")
    p.set_defaults(fn=cmd_tune)

    # ------------------------------------------------------------------- replay
    p = sub.add_parser(
        "replay",
        help="replay an execution trace JSON in human-readable form",
    )
    p.add_argument("trace_file", help="path to a trace JSON file")
    p.add_argument("--filter-rejected", action="store_true",
                   help="do not print the rejected candidates")
    p.add_argument("--filter-selected", action="store_true",
                   help="do not print the selected candidate")
    p.set_defaults(fn=cmd_replay)

    # ------------------------------------------------------------------ compare
    p = sub.add_parser(
        "compare",
        help="side-by-side comparison of two execution traces (JSON to stdout)",
    )
    p.add_argument("trace_a", help="path to trace A")
    p.add_argument("trace_b", help="path to trace B")
    p.set_defaults(fn=cmd_compare)

    # ----------------------------------------------------------------- evidence
    p = sub.add_parser(
        "evidence",
        help="generate an evidence bundle (plan summary + validation + trace + tune + sys-info + git rev)",
    )
    p.add_argument("plan_file", help="path to a model-plan JSON file")
    p.set_defaults(fn=cmd_evidence, _argv=[])

    # ------------------------------------------------------------------ promote
    p = sub.add_parser(
        "promote",
        help="validate a candidate against --gates and write a signed PromotionRecord to .omlx/cache/promotion/",
    )
    p.add_argument("kernel_id", help="candidate id (used as the cache filename)")
    p.add_argument("--gates", required=True,
                   help="comma-separated 'id=threshold' pairs, e.g. mmlu=0.85,gpqa=0.75")
    p.add_argument("--sign-key", default=None,
                   help="optional hex-encoded HMAC signing key")
    p.add_argument("--approver", default=None,
                   help="approver string (defaults to $USER)")
    p.add_argument("--decision", default="auto",
                   help="decision label stored alongside the record (default: auto)")
    p.add_argument("--json", action="store_true",
                   help="emit a JSON envelope to stdout instead of the human summary")
    p.set_defaults(fn=cmd_promote)

    # --------------------------------------------------------------- quarantine
    p = sub.add_parser(
        "quarantine",
        help="append a Hold/Rollback audit entry for a kernel to .omlx/cache/audit.jsonl",
    )
    p.add_argument("kernel_id", help="candidate id to quarantine")
    p.add_argument("--reason", required=True, help="human-readable reason for the audit entry")
    p.add_argument("--action", choices=["hold", "rollback"], default="hold",
                   help="audit-trail action kind (default: hold)")
    p.add_argument("--approver", default=None,
                   help="approver string (defaults to $USER)")
    p.add_argument("--json", action="store_true",
                   help="emit a JSON envelope to stdout instead of the human summary")
    p.set_defaults(fn=cmd_quarantine)

    # ------------------------------------------------------------------- gates
    p = sub.add_parser(
        "gates",
        help="CRUD against per-kernel quality-gate configurations in .omlx/cache/gates/",
    )
    gsub = p.add_subparsers(dest="gates_action")
    list_p = gsub.add_parser("list", help="list gates configured for a kernel")
    list_p.add_argument("kernel_id", help="kernel id")
    list_p.add_argument("--json", action="store_true",
                        help="emit a JSON envelope to stdout")

    add_p = gsub.add_parser("add", help="add or update a gate for a kernel")
    add_p.add_argument("kernel_id", help="kernel id")
    add_p.add_argument("--gate", required=True, help="gate id, e.g. mmlu")
    add_p.add_argument("--threshold", type=float, default=None,
                       help="numeric threshold")
    add_p.add_argument("--at-least", action="store_true",
                       help="require score >= threshold (default)")
    add_p.add_argument("--at-most", action="store_true",
                       help="require score <= threshold (perplexity-style gates)")
    add_p.add_argument("--note", default="", help="optional human-readable note")
    add_p.add_argument("--json", action="store_true",
                       help="emit a JSON envelope to stdout")

    remove_p = gsub.add_parser("remove", help="remove a gate from a kernel")
    remove_p.add_argument("kernel_id", help="kernel id")
    remove_p.add_argument("--gate", required=True, help="gate id to remove")
    remove_p.add_argument("--json", action="store_true",
                          help="emit a JSON envelope to stdout")

    check_p = gsub.add_parser("check", help="evaluate a single gate against an observed score")
    check_p.add_argument("kernel_id", help="kernel id")
    check_p.add_argument("--gate", required=True, help="gate id to evaluate")
    check_p.add_argument("--score", type=float, default=None,
                         help="observed score to test against the gate")
    check_p.add_argument("--json", action="store_true",
                         help="emit a JSON envelope to stdout")

    p.set_defaults(fn=cmd_gates)

    args = parser.parse_args(argv)

    # Forward the raw argv tail to commands that want it (evidence).
    if getattr(args, "cmd", None) == "evidence":
        args._argv = list(argv) if argv is not None else sys.argv[1:]

    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
