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
  vpu-dashboard          — FR-7 oMLX vPU dashboard (health + status + panel)
"""

from __future__ import annotations
import argparse
import sys
from typing import Optional

# Subcommand implementations that live in their own modules are imported
# up-front so the argparse ``set_defaults(fn=cmd_<x>)`` wiring below
# can refer to them by name. The eval subcommand is a special case: its
# implementation was carved out of this file in turn-9's module-size
# sweep (the eval block was the largest single coherent unit in the
# file, owning ``cmd_eval`` + ``_eval_load_dataset`` + ``_eval_stub_score``
# + the ``EVAL_VALID_SUITES`` constant). It is re-exported below so
# existing importers (``from omlx_research.cli import EVAL_VALID_SUITES,
# main``) continue to work unchanged.
from ._cmd_eval import EVAL_VALID_SUITES, cmd_eval
from ._cmd_inference import (
    cmd_status, cmd_inference, cmd_spec_decode, cmd_latentmas,
)


# --- inference / compute subcommands -----------------------------------------
#
# Moved to ``_cmd_inference.py`` in this module-size sweep; the inference
# cluster (status / inference / spec_decode / latentmas) was the second
# largest coherent unit in this file. ``cmd_status``, ``cmd_inference``,
# ``cmd_spec_decode`` and ``cmd_latentmas`` are re-exported via the
# ``from ._cmd_inference import ...`` line at the top of this module so
# existing importers continue to work unchanged.


def cmd_tidar(args: argparse.Namespace) -> int:
    print(f"TiDAR draft_len={args.draft_len} steps={args.steps}")
    print("TiDAR: AR + diffusion hybrid loop. Use --mode=hybrid for full demo.")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    print("bench: stub — implements tok/sec + acceptance rate")
    return 0


# --- eval-harness subcommand -----------------------------------------------
#
# Moved to ``_cmd_eval.py`` in turn-9's module-size sweep; the eval block
# was the largest single coherent unit in this file. ``cmd_eval`` and the
# public ``EVAL_VALID_SUITES`` constant are re-exported via the
# ``from ._cmd_eval import ...`` line at the top of this module so
# existing importers continue to work unchanged.


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
    p.add_argument(
        "--report",
        default=None,
        help="path to eval-harness EvaluationReport JSON (FR-6; replaces synthetic evidence)",
    )
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

    # ----------------------------------------------------------- vpu-dashboard
    def _cmd_vpu_dashboard(a: argparse.Namespace) -> int:
        from omlx_research.vpu_dashboard import serve

        return serve(host=a.host, port=a.port)

    p = sub.add_parser(
        "vpu-dashboard",
        help="FR-7: serve canonical oMLX vPU dashboard (panel + /health + status JSON)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.set_defaults(fn=_cmd_vpu_dashboard)

    args = parser.parse_args(argv)

    # Forward the raw argv tail to commands that want it (evidence).
    if getattr(args, "cmd", None) == "evidence":
        args._argv = list(argv) if argv is not None else sys.argv[1:]

    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
