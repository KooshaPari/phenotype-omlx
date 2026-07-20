"""Individual doctor checks.

Each check function returns a :class:`omlx_research.cli.doctor.Check`
and is wrapped in a broad ``Exception`` guard at the call site
(``run_doctor``), so a single broken check can never abort the report.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

from ._doctor_shared import (
    EXPECTED_KERNEL_OP_COUNT,
    FAIL,
    MIN_PYTHON,
    PASS,
    WARN,
    Check,
    collect_kernel_op_tags,
    project_root,
    read_abi_version,
    read_cargo_version,
)


# Re-export the four newest checks so the existing `checks.<name>`
# access pattern keeps working. The implementations were originally
# collected in `_doctor_extra_checks.py` (turn-4 batch), but that
# module grew to 530L and was split per-topic in turn-9 into three
# siblings — see the module-size-sweep pattern documented in
# turn-8 resume notes §6.1.
#   * `_doctor_extra_niah.py`   — NIAH benchmark check
#   * `_doctor_extra_eval.py`   — eval-harness subcommand check
#   * `_doctor_extra_kernel.py` — package version +
#                                 regress-baseline dispatch envelope
from ._doctor_extra_niah import (  # noqa: E402,F401  (re-export)
    julia_required_on_eval_path,
    niah_benchmark_non_legacy_path,
    niah_benchmark_present,
)
from ._doctor_extra_eval import (  # noqa: E402,F401  (re-export)
    eval_harness_subcommand_runnable,
)
from ._doctor_extra_kernel import (  # noqa: E402,F401  (re-export)
    omlx_research_version,
    regress_baseline_dispatch_envelope,
)
from ._doctor_turn5_checks import (  # noqa: E402,F401  (re-export)
    dispatch_script_metal_exists,
    dispatch_script_sglang_exists,
    dispatch_script_vllm_exists,
    niah_regression_baseline_exists,
)
# Turn-10 INTERNAL checks live in two sibling modules to keep each
# file at or below the 500-line cap. After turn-12 measured
# ``_doctor_internal_checks.py`` at 576 lines, the two longest checks
# (``metal_runtime_lib_test_count_at_least_25`` and
# ``python_cli_subcommand_count_at_least_6``) were carved out into
# :mod:`omlx_research.cli._doctor_internal_checks_split`. Both halves
# are re-exported here so the existing ``checks.<name>`` access
# pattern keeps working.
from ._doctor_internal_checks import (  # noqa: E402,F401  (re-export)
    coverage_tag_count_at_least_25,
    eval_harness_suite_count_at_least_4,
)
from ._doctor_internal_checks_split import (  # noqa: E402,F401  (re-export)
    metal_runtime_lib_test_count_at_least_25,
    python_cli_subcommand_count_at_least_6,
)
from ._doctor_internal_checks_turn12 import (  # noqa: E402,F401  (re-export)
    cargo_workspace_crate_count_at_least_15,
    ddm_continuous_schedule_variants_at_least_4,
)


def python_version() -> Check:
    current = (sys.version_info.major, sys.version_info.minor)
    expected = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}+"
    if current >= MIN_PYTHON:
        return Check(
            id="python_version",
            description=f"Python interpreter >= {expected}",
            status=PASS,
            details=f"{platform.python_version()} on {platform.platform()}",
        )
    return Check(
        id="python_version",
        description=f"Python interpreter >= {expected}",
        status=FAIL,
        details=(
            f"found Python {platform.python_version()} on {platform.platform()} "
            f"(need {expected})"
        ),
    )


def mlx_core() -> Check:
    is_apple_silicon = (sys.platform == "darwin" and platform.machine() == "arm64")
    try:
        import mlx.core as mx  # type: ignore
    except Exception as e:
        return Check(
            id="mlx_core_available",
            description="mlx.core importable (required on Apple Silicon)",
            status=FAIL if is_apple_silicon else WARN,
            details=f"import failed: {type(e).__name__}: {e}",
        )
    metal_note = ""
    try:
        if hasattr(mx, "metal") and hasattr(mx.metal, "is_available"):
            metal_note = f" | metal_available={bool(mx.metal.is_available())}"
    except Exception:
        metal_note = " | metal_status=unknown"
    return Check(
        id="mlx_core_available",
        description="mlx.core importable (required on Apple Silicon)",
        status=PASS,
        details=f"mlx.core {getattr(mx, '__version__', 'unknown')}{metal_note}",
    )


def mlx_lm() -> Check:
    try:
        import mlx_lm  # type: ignore  # noqa: F401
    except Exception as e:
        return Check(
            id="mlx_lm_available",
            description="mlx_lm importable (production-path inference)",
            status=WARN,
            details=(
                f"not installed ({type(e).__name__}); production-path "
                f"tests will be skipped. Install with `pip install mlx-lm` "
                f"when you need end-to-end inference."
            ),
        )
    return Check(
        id="mlx_lm_available",
        description="mlx_lm importable (production-path inference)",
        status=PASS,
        details=f"mlx_lm {getattr(mlx_lm, '__version__', 'unknown')}",
    )


# Subcommands whose production path strictly requires `mlx_lm`. When
# the user invokes one of these commands in an environment without
# `mlx_lm`, the doctor should escalate the warning to a hard fail so
# they don't ship a build that can only run the doctor. (See
# `cli/__init__.py::cmd_inference` for the matching require_mlx_lm gate.)
_MLX_LM_REQUIRED_COMMANDS: frozenset[str] = frozenset({"inference"})


def mlx_lm_required_by_command(cmd: str) -> Check:
    """Cross-reference the active command against the mlx_lm requirement.

    The static ``mlx_lm`` check above returns ``WARN`` regardless of
    the active command. This companion check is only meaningful in
    the doctor context where the user names the command they intend
    to run (e.g. ``omlx-research doctor --command inference``); when
    that command is in :data:`_MLX_LM_REQUIRED_COMMANDS` and mlx_lm
    is missing, escalate to ``FAIL`` so the report reflects that
    the active command path is broken.
    """
    desc = f"mlx_lm importable for active command `{cmd}`"
    try:
        import mlx_lm  # type: ignore  # noqa: F401
    except Exception as e:
        if cmd in _MLX_LM_REQUIRED_COMMANDS:
            return Check(
                id="mlx_lm_required_by_command",
                description=desc,
                status=FAIL,
                details=(
                    f"`{cmd}` requires mlx_lm in production but it is "
                    f"not importable ({type(e).__name__}: {e}). Install "
                    f"with `pip install mlx-lm` (and `pip install mlx-core` "
                    f"on Apple Silicon)."
                ),
            )
        return Check(
            id="mlx_lm_required_by_command",
            description=desc,
            status=PASS,
            details=(
                f"command `{cmd}` does not require mlx_lm; nothing to fail."
            ),
        )
    return Check(
        id="mlx_lm_required_by_command",
        description=desc,
        status=PASS,
        details=f"mlx_lm {getattr(mlx_lm, '__version__', 'unknown')} available for `{cmd}`",
    )


def turboquant_rust_extension() -> Check:
    try:
        import _perf  # type: ignore  # noqa: F401
    except Exception as e:
        return Check(
            id="turboquant_rust_extension_available",
            description="`_perf` Rust extension importable (TurboQuant Rust path)",
            status=WARN,
            details=(
                f"not importable ({type(e).__name__}: {e}); the Python "
                f"fallback path still works, but Rust-SIMD speedups are "
                f"disabled until `maturin develop` is run from "
                f"perf-core/turbo-quant."
            ),
        )
    return Check(
        id="turboquant_rust_extension_available",
        description="`_perf` Rust extension importable (TurboQuant Rust path)",
        status=PASS,
        details=f"_perf {getattr(_perf, '__version__', 'unknown')}",
    )


def kernel_registry_version() -> Check:
    version = read_cargo_version("perf-core/kernel-registry")
    if version == "unknown":
        return Check(
            id="kernel_registry_version",
            description="kernel-registry crate version (from Cargo.toml)",
            status=WARN,
            details="could not read perf-core/kernel-registry/Cargo.toml",
        )
    return Check(
        id="kernel_registry_version",
        description="kernel-registry crate version (from Cargo.toml)",
        status=PASS,
        details=version,
    )


def regress_baseline_version() -> Check:
    version = read_cargo_version("perf-core/regress-baseline")
    if version == "unknown":
        return Check(
            id="regress_baseline_version",
            description="regress-baseline crate version (from Cargo.toml)",
            status=WARN,
            details="could not read perf-core/regress-baseline/Cargo.toml",
        )
    return Check(
        id="regress_baseline_version",
        description="regress-baseline crate version (from Cargo.toml)",
        status=PASS,
        details=version,
    )


def model_kernels_operator_coverage() -> Check:
    tags = collect_kernel_op_tags()
    desc = (
        f"model-kernels KernelOp tag coverage "
        f"(expect >= {EXPECTED_KERNEL_OP_COUNT})"
    )
    if not tags:
        return Check(
            id="model_kernels_operator_coverage",
            description=desc,
            status=WARN,
            details="could not parse perf-core/model-kernels/src/lib.rs tag() arms",
        )
    count = len(tags)
    head = ", ".join(tags[:6])
    tail = "..." if count > 6 else ""
    summary = f"{count} tags: {head}{tail}"
    if count >= EXPECTED_KERNEL_OP_COUNT:
        return Check(
            id="model_kernels_operator_coverage",
            description=desc,
            status=PASS,
            details=summary,
        )
    return Check(
        id="model_kernels_operator_coverage",
        description=desc,
        status=WARN,
        details=(
            f"{summary} — only {count} tag(s) found; "
            f"expected at least {EXPECTED_KERNEL_OP_COUNT}"
        ),
    )


def native_abi_v1() -> Check:
    crate_dir = os.path.join(project_root(), "perf-core", "native-abi")
    cargo_path = os.path.join(crate_dir, "Cargo.toml")
    desc = "native-abi v1 crate present (perf-core/native-abi)"
    if not os.path.isdir(crate_dir) or not os.path.isfile(cargo_path):
        return Check(
            id="native_abi_v1",
            description=desc,
            status=WARN,
            details=f"{crate_dir} missing Cargo.toml",
        )
    abi = read_abi_version()
    if abi is None:
        return Check(
            id="native_abi_v1",
            description=desc,
            status=PASS,
            details="crate present; could not parse ABI_VERSION_CURRENT",
        )
    if not abi.startswith("1."):
        return Check(
            id="native_abi_v1",
            description=desc,
            status=WARN,
            details=f"ABI version {abi} != 1.x",
        )
    return Check(
        id="native_abi_v1",
        description=desc,
        status=PASS,
        details=f"ABI v{abi}",
    )


def airlock_v2() -> Check:
    path = shutil.which("airlock-v2")
    if path:
        return Check(
            id="airlock_v2_installed",
            description="airlock-v2 binary on PATH",
            status=PASS,
            details=path,
        )
    return Check(
        id="airlock_v2_installed",
        description="airlock-v2 binary on PATH",
        status=WARN,
        details=(
            "NOT INSTALLED — airlock-v2 is a known unresolved P2 from "
            "the session; documenting it explicitly here so doctor users "
            "see the gap. Install once the upstream crate ships."
        ),
    )


# NOTE: New checks added 2026-07-19 (version, NIAH, eval-harness,
# regress-baseline dispatch envelope) were split into three per-topic
# modules in turn-9 to keep each file under the 350L target:
# `_doctor_extra_niah.py`, `_doctor_extra_eval.py`, and
# `_doctor_extra_kernel.py`. They are re-exported at the top of this
# file so the existing `checks.<name>` access pattern keeps working.


def tests_runnable() -> Check:
    """Run pytest --collect-only on the package; exit 0/5 is acceptable."""
    python_dir = os.path.join(project_root(), "python")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "omlx_research",
        "--ignore=omlx_research/tests/test_mlx_backend.py",
    ]
    desc = "pytest can collect the omlx_research test suite"
    try:
        proc = subprocess.run(
            cmd,
            cwd=python_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return Check(
            id="tests_runnable",
            description=desc,
            status=FAIL,
            details=f"could not invoke pytest: {type(e).__name__}: {e}",
        )
    rc = proc.returncode
    if rc in (0, 5):
        return Check(
            id="tests_runnable",
            description=desc,
            status=PASS,
            details=f"pytest --collect-only exited {rc} (acceptable)",
        )
    err_tail = (proc.stderr or proc.stdout).strip().splitlines()[-1] if (
        proc.stderr or proc.stdout
    ) else ""
    return Check(
        id="tests_runnable",
        description=desc,
        status=FAIL,
        details=f"pytest --collect-only exited {rc}: {err_tail[:200]}",
    )
