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

    profile = HardwareProfile(
        os=platform.system(),
        arch=platform.machine(),
        cpu_count_logical=_os.cpu_count() or 1,
        cpu_count_physical=_os.cpu_count() or 1,           # python doesn't expose phys easily
        memory_total_gb=_memory_total_gb(),
        accelerators=accels,
        apple_silicon_model=apple_model if is_darwin and "Apple" in apple_model else "",
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

    # Agents always default to their native backend
    prefs[BackendKind.LATENTMAS] = [BackendKind.LATENTMAS, BackendKind.MLX_METAL, cpu]
    prefs[BackendKind.TIDAR] = [BackendKind.TIDAR, BackendKind.MLX_METAL, cpu]
    prefs[BackendKind.JETSPEC] = [BackendKind.JETSPEC, BackendKind.MLX_METAL, cpu]
    prefs[BackendKind.SSD] = [BackendKind.SSD, BackendKind.JETSPEC, cpu]

    return prefs


__all__ = ["Accelerator", "HardwareProfile", "detect_hardware"]