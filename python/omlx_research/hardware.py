"""Hardware detection — what does this machine have? What should run on it?

Used by the hybrid orchestrator to choose the best backends for a request.
For example, on Apple Silicon it picks MLX/Metal; on an NVIDIA box it picks
vLLM or TensorRT; on AMD ROCm it picks vLLM ROCm; on CPU-only it picks
llama.cpp.

The detection is non-invasive: only `import` checks, no actual model loading.
"""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import json
from dataclasses import dataclass, field
from enum import Enum

from .nanovm import BackendKind

logger = logging.getLogger(__name__)


class Accelerator(str, Enum):
    CPU = "cpu"
    METAL = "metal"           # Apple Metal GPU
    CUDA = "cuda"             # NVIDIA
    ROCM = "rocm"             # AMD
    TPU = "tpu"               # Google
    NEURON = "neuron"         # AWS Inferentia / Trainium


@dataclass
class HardwareProfile:
    """Snapshot of what this machine has."""
    os: str                                              # "Darwin" | "Linux" | "Windows"
    arch: str                                            # "arm64" | "x86_64"
    cpu_count_logical: int
    cpu_count_physical: int
    memory_total_gb: float
    accelerators: list[Accelerator] = field(default_factory=list)
    apple_silicon_model: str = ""                        # e.g. "M1 Pro" if detected

    # Optional topology/telemetry fields. ``None`` means the platform did not
    # provide evidence; it is deliberately different from ``False``.
    cpu_performance_cores: int | None = None
    cpu_efficiency_cores: int | None = None
    metal_device_name: str | None = None
    metal_gpu_cores: int | None = None
    unified_memory_gb: float | None = None
    thermal_state: str | None = None
    memory_pressure: str | None = None
    neural_engine_available: bool | None = None
    npu_available: bool | None = None
    capability_evidence: dict[str, str] = field(default_factory=dict)

    # Best backend for each candidate kind on this hardware, ordered by preference.
    backend_preference: dict[BackendKind, list[BackendKind]] = field(default_factory=dict)

    def recommend(self, kind: BackendKind) -> BackendKind:
        """Return the most preferred backend kind for this hardware, given the desired one."""
        prefs = self.backend_preference.get(kind, [kind])
        return prefs[0] if prefs else kind

    def has(self, accel: Accelerator) -> bool:
        return accel in self.accelerators

    def summary(self) -> str:
        accs = ", ".join(a.value.upper() for a in self.accelerators) or "CPU only"
        return (
            f"{self.os}/{self.arch} | {self.cpu_count_logical} cores | "
            f"{self.memory_total_gb:.1f} GB RAM | accel=[{accs}]"
            + (f" | {self.apple_silicon_model}" if self.apple_silicon_model else "")
        )


def _run_command(args: list[str]) -> str | None:
    """Run a short, read-only platform probe and return stdout if available."""
    try:
        return subprocess.check_output(
            args, text=True, timeout=2, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _sysctl_int(name: str) -> int | None:
    value = _run_command(["sysctl", "-n", name]) if platform.system() == "Darwin" else None
    try:
        return int(value) if value else None
    except ValueError:
        return None


def _detect_cpu_topology() -> tuple[int | None, int | None, int | None, str | None]:
    """Return physical, performance, efficiency counts and evidence source."""
    system = platform.system()
    if system == "Darwin":
        physical = _sysctl_int("hw.physicalcpu")
        # Apple exposes these on heterogeneous systems; absent keys are normal
        # on Intel Macs and older macOS releases.
        performance = _sysctl_int("hw.perflevel0.logicalcpu")
        efficiency = _sysctl_int("hw.perflevel1.logicalcpu")
        source = "sysctl" if any(v is not None for v in (physical, performance, efficiency)) else None
        return physical, performance, efficiency, source
    if system == "Linux":
        physical = None
        performance = efficiency = None
        text = _run_command(["lscpu", "-J"])
        if text:
            try:
                entries = json.loads(text).get("lscpu", [])
                values = {e.get("field", "").rstrip(":"): e.get("data") for e in entries}
                sockets = int(values.get("Socket(s)", 1))
                cores = int(values["Core(s) per socket"]) if values.get("Core(s) per socket") else None
                physical = sockets * cores if cores is not None else None
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
        # Linux hybrid topology is exposed per CPU on newer kernels. Count
        # core_type values only when every value is readable and unambiguous.
        try:
            from pathlib import Path
            types = [
                Path(path).read_text().strip()
                for path in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology/core_type"))
            ]
            if types and all(types):
                performance = types.count("1") or None
                efficiency = types.count("2") or None
        except (OSError, ValueError):
            pass
        return physical, performance, efficiency, "lscpu/sysfs" if physical or performance else None
    return None, None, None, None


def _detect_apple_metal() -> tuple[str | None, int | None, str | None]:
    """Return Metal device name/core count only when system evidence exists."""
    if platform.system() != "Darwin":
        return None, None, None
    text = _run_command(["system_profiler", "SPDisplaysDataType", "-json"])
    if not text:
        return None, None, None
    try:
        data = json.loads(text)
        displays = data.get("SPDisplaysDataType", [])
        if not displays:
            return None, None, None
        item = displays[0]
        name = item.get("sppci_model") or item.get("_name")
        cores = item.get("spdisplays_core_count")
        try:
            cores = int(cores) if cores is not None else None
        except (TypeError, ValueError):
            cores = None
        return name, cores, "system_profiler"
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, None, None


def _detect_thermal_state() -> tuple[str | None, str | None]:
    """Read a coarse thermal state; no temperature is invented when absent."""
    system = platform.system()
    if system == "Darwin":
        text = _run_command(["pmset", "-g", "therm"])
        if text:
            lowered = text.lower()
            if "critical" in lowered:
                return "critical", "pmset"
            if "warning" in lowered or "heavy" in lowered:
                return "elevated", "pmset"
            return "nominal", "pmset"
    if system == "Linux":
        try:
            from pathlib import Path
            states = [Path(p).read_text().strip().lower() for p in Path("/sys/class/thermal").glob("thermal_zone*/trip_point_*_type")]
            if any("critical" in s for s in states):
                return "critical", "sysfs"
        except OSError:
            pass
    return None, None


def _detect_memory_pressure() -> tuple[str | None, str | None]:
    if platform.system() == "Darwin":
        text = _run_command(["memory_pressure"])
        if text:
            lowered = text.lower()
            if "critical" in lowered:
                return "critical", "memory_pressure"
            if "warn" in lowered:
                return "warning", "memory_pressure"
            return "nominal", "memory_pressure"
    return None, None


def _detect_apple_silicon() -> str:
    if platform.system() != "Darwin":
        return ""
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=2
        ).strip()
        return out
    except Exception:                                # noqa: BLE001
        return "Apple Silicon"


def _detect_metal() -> bool:
    try:
        import mlx.core as mx                        # noqa: F401
        return mx.metal.is_available() if hasattr(mx, "metal") else False
    except Exception:                                # noqa: BLE001
        return False


def _detect_cuda() -> tuple[bool, str]:
    try:
        import torch                                # noqa: F401
        import torch as t
        if t.cuda.is_available():
            count = t.cuda.device_count()
            name = t.cuda.get_device_name(0) if count else "?"
            return True, f"{count}x {name}"
        return False, ""
    except Exception:                                # noqa: BLE001
        return False, ""


def _detect_rocm() -> tuple[bool, str]:
    # torch reports ROCm as "cuda" backend with rocm version
    if shutil.which("rocm-smi") is None:
        return False, ""
    try:
        out = subprocess.check_output(["rocm-smi"], text=True, timeout=2, stderr=subprocess.DEVNULL)
        if "GPU" in out or "Card" in out:
            return True, out.split("\n", 1)[0]
    except Exception:                                # noqa: BLE001
        pass
    return False, ""


def _memory_total_gb() -> float:
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=2).strip()
            return int(out) / 1024 / 1024 / 1024
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return kb / 1024 / 1024
    except Exception:                                # noqa: BLE001
        pass
    return 0.0


def detect_hardware() -> HardwareProfile:
    """Build a `HardwareProfile` describing this machine."""
    import os as _os
    accels: list[Accelerator] = [Accelerator.CPU]
    apple_model = _detect_apple_silicon()
    is_darwin = platform.system() == "Darwin"
    is_linux = platform.system() == "Linux"

    if _detect_metal() and is_darwin:
        accels.append(Accelerator.METAL)
    cuda, _cuda_info = _detect_cuda()
    if cuda:
        accels.append(Accelerator.CUDA)
    if is_linux:
        rocm, _ = _detect_rocm()
        if rocm:
            accels.append(Accelerator.ROCM)

    physical, performance, efficiency, topology_source = _detect_cpu_topology()
    metal_name, metal_cores, metal_source = _detect_apple_metal()
    thermal, thermal_source = _detect_thermal_state()
    pressure, pressure_source = _detect_memory_pressure()
    evidence = {
        key: value for key, value in {
            "cpu_topology": topology_source,
            "metal": metal_source,
            "thermal": thermal_source,
            "memory_pressure": pressure_source,
        }.items() if value is not None
    }
    logical = _os.cpu_count() or 1
    # A physical count is unknown when the platform probe failed. Preserve the
    # historical integer field for callers, while exposing the evidence-backed
    # value through the optional topology field only.
    physical_compat = physical if physical is not None else logical

    profile = HardwareProfile(
        os=platform.system(),
        arch=platform.machine(),
        cpu_count_logical=logical,
        cpu_count_physical=physical_compat,
        memory_total_gb=_memory_total_gb(),
        accelerators=accels,
        apple_silicon_model=apple_model if is_darwin and "Apple" in apple_model else "",
        cpu_performance_cores=performance,
        cpu_efficiency_cores=efficiency,
        metal_device_name=metal_name,
        metal_gpu_cores=metal_cores,
        unified_memory_gb=_memory_total_gb() if is_darwin else None,
        thermal_state=thermal,
        memory_pressure=pressure,
        # No generic API can prove ANE/XDNA availability. Leave both unknown;
        # provider-specific runtime probes must set these after verification.
        neural_engine_available=None,
        npu_available=None,
        capability_evidence=evidence,
    )

    # Build the backend preference table based on what we detected.
    profile.backend_preference = _build_preferences(profile)

    return profile


def _build_preferences(profile: HardwareProfile) -> dict[BackendKind, list[BackendKind]]:
    """For each backend kind, what's the best fallback order on this hardware?"""
    has_metal = profile.has(Accelerator.METAL)
    has_cuda = profile.has(Accelerator.CUDA)
    has_rocm = profile.has(Accelerator.ROCM)

    # Default order: requested kind, then cpu fallback, then any GPU backend.
    cpu = BackendKind.LLAMACPP
    prefs: dict[BackendKind, list[BackendKind]] = {}

    # MLX/Metal requested: prefer MLX on Apple Silicon, else fall back to llama.cpp CPU
    prefs[BackendKind.MLX_METAL] = (
        [BackendKind.MLX_METAL, cpu] if has_metal
        else [cpu, BackendKind.LLAMACPP]
    )

    # SGLang is the cross-platform default — works wherever CUDA/ROCm/Metal is
    if has_cuda:
        prefs[BackendKind.SGLANG] = [BackendKind.SGLANG, BackendKind.VLLM, cpu]
    elif has_rocm:
        prefs[BackendKind.SGLANG] = [BackendKind.SGLANG, BackendKind.VLLM, cpu]
    elif has_metal:
        prefs[BackendKind.SGLANG] = [BackendKind.MLX_METAL, cpu]
    else:
        prefs[BackendKind.SGLANG] = [cpu]

    # vLLM: NVIDIA primary, ROCm experimental, else CPU llama.cpp
    if has_cuda:
        prefs[BackendKind.VLLM] = [BackendKind.VLLM, BackendKind.SGLANG, cpu]
    elif has_rocm:
        prefs[BackendKind.VLLM] = [BackendKind.VLLM, cpu]
    else:
        prefs[BackendKind.VLLM] = [cpu]

    # TensorRT: NVIDIA only
    prefs[BackendKind.TENSORRT] = (
        [BackendKind.TENSORRT, BackendKind.VLLM, cpu] if has_cuda else [cpu]
    )

    # llama.cpp: universal fallback
    prefs[BackendKind.LLAMACPP] = [BackendKind.LLAMACPP]

    # Strategies don't live in BackendKind — they're StrategyKind.
    # When a strategy asks "which backend for me?" the orchestrator delegates
    # to the plugin's `compatible_backends` and picks the first one available.

    return prefs


__all__ = ["Accelerator", "HardwareProfile", "detect_hardware"]
