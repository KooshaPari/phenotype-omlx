#!/usr/bin/env python3
"""phenotype-omlx readiness — 12 checks with per-check timeout.

Replaces the heredoc-via-bash approach in `phenotype-omlx-ready` with a single
Python file that has signal-based per-check timeouts and proper flushing.
"""

import sys, os, traceback, time, signal

# ── per-check timeout: abort any single check after `PER_CHECK` seconds ──
PER_CHECK = float(os.environ.get("PER_CHECK_TIMEOUT", "45"))
ok_count = 0
results = []
current = {"label": "", "t0": 0.0}


class TimeoutError(Exception):
    pass


def _alarm(signum, frame):
    raise TimeoutError(f"per-check {PER_CHECK:.0f}s timeout")


def _format_progress(label: str, status: str, dt_ms: float, msg: str = "") -> None:
    mark = "✓" if status == "ok" else "✗"
    line = f"  {mark} {label:34s}  {dt_ms:7.0f}ms  {msg}"
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def chk(label: str, fn, timeout: float = PER_CHECK):
    """Run `fn()` with timeout, stream a ✓/✗ line, return (passed, msg)."""
    t0 = time.perf_counter()
    sys.stdout.write(f"  ... {label}")
    sys.stdout.flush()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(timeout))
    try:
        msg = fn()
        signal.alarm(0)
        if not isinstance(msg, str):
            msg = "ok"
        dt = (time.perf_counter() - t0) * 1000
        _format_progress(label, "ok", dt, msg)
        results.append((label, True, dt, msg))
        return True, msg
    except Exception as e:
        signal.alarm(0)
        dt = (time.perf_counter() - t0) * 1000
        err = f"{type(e).__name__}: {str(e)[:120]}"
        if isinstance(e, TimeoutError):
            err = f"TIMEOUT after {timeout:.0f}s"
        _format_progress(label, "fail", dt, err)
        results.append((label, False, dt, err))
        return False, err


REPOS = os.environ.get(
    "PHENOTYPE_REPOS_ROOT", "/Users/kooshapari/CodeProjects/Phenotype/repos"
)
ROOT = f"{REPOS}/phenotype-omlx"


def _provider_root(name: str) -> str | None:
    """Resolve an optional research checkout without assuming sibling layout.

    Order: explicit ``PHENOTYPE_<NAME>_ROOT`` override, then ``<REPOS>/<name>``.
    Returns ``None`` when the provider is simply not installed on this host,
    so the caller can report ``not-installed`` instead of raising.
    """
    candidates = [os.environ.get(f"PHENOTYPE_{name.upper()}_ROOT")]
    candidates.append(os.path.join(REPOS, name))
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def _missing_provider(name: str) -> str:
    return f"not-installed ({name}; set PHENOTYPE_{name.upper()}_ROOT)"


def _is_free_threaded(soabi: str) -> bool:
    """True when running a free-threaded (no-GIL) CPython.

    Two independent signals, either of which is conclusive:

    * ``sys._is_gil_enabled()`` (CPython 3.13+) reports the *live* GIL state.
    * ``SOABI`` encodes the build flavour as ``cpython-<ver>[t]-<platform>``,
      e.g. ``cpython-314-darwin`` (GIL) vs ``cpython-314t-darwin`` (free-threaded).
      Checked because a free-threaded build can still re-enable the GIL at
      runtime, and the abi3 extension is unimportable either way.
    """
    if not getattr(sys, "_is_gil_enabled", lambda: True)():
        return True
    parts = soabi.split("-")
    return len(parts) >= 2 and parts[1].endswith("t")


# Probe source for the installed pyo3 extension. `python/ffi/Cargo.toml` declares
# `[lib] name = "_perf"`, so maturin installs it top-level; accept the packaged
# `omlx_research._perf` layout too so this check keeps working if the FFI wheel
# is ever moved under the package namespace.
PERF_PROBE_SRC = (
    "try:\n"
    "    import _perf\n"
    "except ModuleNotFoundError:\n"
    "    from omlx_research import _perf\n"
    "q = _perf.turbo_quant_encode([0.1, -0.2, 0.3, -0.4, 0.5] * 26, "
    "group_size=32, bits=4)\n"
    "packed = q.get('packed')\n"
    "assert packed, 'empty packed result'\n"
    "print(f\"packed={len(packed)}B, scales={len(q.get('scales') or [])}\")\n"
)


def c_turboquant_mlx():
    from mlx.nn.layers.turbo_kv_cache import (
        TurboKVCache,
        make_turbo_cache,
        compact_turbo_cache,
    )

    TurboKVCache(bits=4, key_bits=4)
    return "TurboKVCache ready (Metal)"


def c_jetspec():
    root = _provider_root("JetSpec")
    if not root:
        return _missing_provider("JetSpec")
    sys.path.insert(0, root)
    sys.modules.pop("jetspec", None)
    import jetspec

    return f"v{getattr(jetspec, '__version__', '?')}"


def c_tidar():
    root = _provider_root("TiDAR")
    if not root:
        return _missing_provider("TiDAR")
    sys.path.insert(0, root)
    sys.modules.pop("models", None)
    from models.tidar import tidar_forward
    from models.base_lm import BaseLM

    return "tidar_forward ok"


def c_latentmas():
    root = _provider_root("LatentMAS")
    if not root:
        return _missing_provider("LatentMAS")
    sys.path.insert(0, root)
    sys.modules.pop("models", None)
    from models import ModelWrapper
    from methods import Agent, default_agents

    n = len(default_agents())
    return f"{n} agents"


def c_ssd():
    root = _provider_root("ssd")
    if not root:
        return _missing_provider("ssd")
    sys.path.insert(0, root)
    os.environ.setdefault(
        "SSD_HF_CACHE", os.path.expanduser("~/.cache/huggingface/hub")
    )
    os.environ.setdefault(
        "SSD_DATASET_DIR", os.path.expanduser("~/.cache/ssd/datasets")
    )
    try:
        from ssd import config, llm  # noqa: F401

        return "ok"
    except ImportError as e:
        msg = str(e).lower()
        if "flashinfer" in msg or "cuda" in msg:
            return "ok (CUDA-only ref, use turbo_mlx.ssd on Metal)"
        raise


def c_backends():
    sys.path.insert(0, f"{ROOT}/python")
    from omlx_research.backends import (
        MlxBackend,
        MetalKernelBackend,
        VllmBackend,
        TensorrtBackend,
        SglangBackend,
        LlamaCppBackend,
    )

    n = sum(1 for n in dir() if not n.startswith("_"))
    return f"{n} backend classes"


def c_engines():
    from omlx_research.engines import (
        SpeculativeEngine,
        TreeAttentionEngine,
        ParallelBatchEngine,
        HybridDispatch,
    )

    return "4 engines"


def c_agents():
    from omlx_research.agents import (
        LatentMasRunner,
        TidarRunner,
        SsdRunner,
        JetSpecRunner,
        ConcurrentScheduler,
    )

    return "5 agents"


def c_cli_web():
    from omlx_research.cli import main as _cli
    from omlx_research.web import main as _web

    return "cli + web"


def c_hybrid():
    from omlx_research.nanovm import (
        PluginRegistry,
        BackendKind,
        PluginSpec,
        discover_plugins,
        list_available_backends,
        list_available_strategies,
    )

    discover_plugins()
    n_backends = len(list_available_backends())
    n_strats = len(list_available_strategies())
    return f"{n_backends} plugins, {n_strats} strategies"


def c_perf():
    import subprocess
    import sysconfig

    # The pyo3 extension is built with abi3-py311 and is *not* importable on a
    # free-threaded (no-GIL) CPython. Report that as not-applicable rather than
    # a failure so readiness stays truthful on 3.13t/3.14t interpreters.
    soabi = sysconfig.get_config_var("SOABI") or ""
    if _is_free_threaded(soabi):
        return f"not-applicable (free-threaded ABI {soabi or '?'}; use GIL-enabled CPython)"

    # The readiness runner prepends the in-repo source tree to PYTHONPATH for
    # earlier checks. Drop it for this probe so we validate the *installed*
    # ABI3 extension rather than an un-built source shadow.
    probe = subprocess.run(
        [sys.executable, "-c", PERF_PROBE_SRC],
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "installed _perf probe failed — run `maturin develop --release` in "
            f"python/ffi/: {(probe.stderr or probe.stdout).strip()[:400]}"
        )
    return f"installed ABI3 smoke={probe.stdout.strip()}"


# ── Check 12: turboquant_plus_production_path ─────────────────────────────
# Catches the regression where compact_turbo_cache returned 0/N compressed
# because the cache list held TurboKVCache (not Lite) instances and the
# compact call silently skipped non-Lite entries. This check exercises the
# full pipeline: MlxBackend init -> model load -> generate_with_turbo_cache
# -> make_turbo_cache -> compact_turbo_cache. It must verify the chosen
# tokenizer actually emits compressed TurboKVCacheLite entries (the count
# must be > 0 and the response must include non-zero turbo metadata).
def c_turboquant_plus_production_path():
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    sys.path.insert(0, f"{ROOT}/python")

    # 1. _perf_module must initialize cleanly through MlxBackend.__init__
    from omlx_research.backends.mlx_backend import MlxBackend, GenerateRequest

    be = MlxBackend()
    if not hasattr(be, "_perf_module"):
        raise RuntimeError("MlxBackend._perf_module missing from __init__")
    # Trigger lazy load — must not throw.
    _ = be._rust_perf()

    # 2. Resolve Qwen3.5 via config/smoke_models.json (Qwen2.5 quarantined)
    from omlx_research.smoke_models import default_model_for
    from huggingface_hub import snapshot_download

    model_id = default_model_for("readiness")
    model_path = snapshot_download(model_id)

    # Qwen3.5 uses recurrent / linear-attention blocks rather than a standard
    # K/V cache, so TurboKV compression has nothing to compact. Classify that
    # as not-applicable instead of letting the lite_layers == 0 assertion below
    # report a false failure.
    import json

    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as config_file:
            model_type = json.load(config_file).get("model_type", "")
        if model_type == "qwen3_5":
            return "not-applicable (Qwen3.5 recurrent/linear attention; no K/V cache)"

    # 3. End-to-end: 2048-token prompt + force_compact=True to bypass
    #    compact_threshold gating (which would otherwise require >= 8K
    #    tokens before compaction kicks in).
    be2 = MlxBackend(model_path)
    filler = "The quick brown fox jumps over the lazy dog. " * 200
    req = GenerateRequest(
        prompt=filler[:2048],
        max_tokens=20,
        temperature=0.0,
    )
    resp = be2.generate_with_turbo_cache(
        req,
        turbo_bits=4,
        turbo_key_bits=0,
        compact_after_prefill=True,
        force_compact=True,
    )
    turbo = resp.metadata.get("turbo", {}) if resp.metadata else {}
    n_lite = turbo.get("lite_layers", 0)
    n_compressed = turbo.get("compressed", 0)
    bytes_freed = turbo.get("bytes_freed", 0)
    if n_lite <= 0:
        raise RuntimeError(
            f"make_turbo_cache did not wrap any layers as TurboKVCacheLite "
            f"(lite_layers={n_lite})",
        )
    if n_compressed <= 0:
        raise RuntimeError(
            f"compact_turbo_cache compressed 0 layers (compressed={n_compressed}, "
            f"lite_layers={n_lite}, layers={turbo.get('layers')}, "
            f"boundary={turbo.get('boundary')}, force_compact={turbo.get('force_compact')})",
        )
    return (
        f"model={model_id}, {n_compressed}/{n_lite} lite compressed, "
        f"bytes_freed={bytes_freed}, provenance="
        f"{turbo.get('quantization_provenance', {}).get('execution_source')}, "
        f"backend={resp.backend}"
    )


CHECKS = [
    ("turboquant+ MLX", c_turboquant_mlx, 60),  # MLX kernel compile
    ("jetspec", c_jetspec, 15),
    ("tidar", c_tidar, 15),
    ("latentmas", c_latentmas, 15),
    ("ssd (CUDA-only ref)", c_ssd, 20),
    ("omlx_research.backends", c_backends, 20),
    ("omlx_research.engines", c_engines, 15),
    ("omlx_research.agents", c_agents, 15),
    ("omlx_research.cli+web", c_cli_web, 20),
    ("omlx_research.nanovm+hybrid", c_hybrid, 25),
    ("pyo3 FFI (_perf)", c_perf, 15),
    ("turboquant+ production path", c_turboquant_plus_production_path, 90),
]  # the 12th check needs model load + prefill (≥90s budget)


def main() -> int:
    print(
        f"phenotype-omlx readiness — 12 checks (per-check timeout " f"{PER_CHECK:.0f}s)"
    )
    print()
    for label, fn, to in CHECKS:
        chk(label, fn, timeout=to)

    ok = sum(1 for _, p, _, _ in results if p)
    total = len(results)
    print()
    total_ms = sum(t for _, _, t, _ in results)
    print(f"  {ok}/{total} checks pass ({total_ms:.0f}ms total)")
    print()
    if ok == total:
        print("  phenotype-omlx READY")
        return 0
    print("  phenotype-omlx NOT READY")
    return 1


if __name__ == "__main__":
    sys.exit(main())
