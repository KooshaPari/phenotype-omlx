"""Non-executing admission tests for the local Qwen3.5 NIAH Harbor job."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "evals"))

from qwen35_niah_preflight import preflight  # noqa: E402
from qwen35_niah_preflight import PreflightError  # noqa: E402


class Qwen35NiahPreflightTests(unittest.TestCase):
    def test_accepts_bounded_qwen35_job_with_admissible_resources(self) -> None:
        plan = preflight(
            ROOT / "evals/harbor/jobs/niah-qwen35-local.yaml",
            available_memory_bytes=8 * 1024**3,
            port_available=lambda _: True,
        )

        self.assertEqual(plan.model, "mlx-community/Qwen3.5-0.8B-OptiQ-4bit")
        self.assertEqual(plan.port, 8081)
        self.assertFalse(plan.workload_executed)
        self.assertFalse(plan.model_loaded)

    def test_rejects_unavailable_port_without_starting_workload(self) -> None:
        with self.assertRaisesRegex(PreflightError, "port 8081 is unavailable"):
            preflight(
                ROOT / "evals/harbor/jobs/niah-qwen35-local.yaml",
                available_memory_bytes=8 * 1024**3,
                port_available=lambda _: False,
            )

    def test_rejects_insufficient_memory_without_starting_workload(self) -> None:
        with self.assertRaisesRegex(PreflightError, "available memory"):
            preflight(
                ROOT / "evals/harbor/jobs/niah-qwen35-local.yaml",
                available_memory_bytes=3 * 1024**3,
                port_available=lambda _: True,
            )

    def test_rejects_unsafe_job_limits(self) -> None:
        unsafe = ROOT / "evals/harbor/jobs/niah-qwen35-local.yaml"
        with self.assertRaisesRegex(PreflightError, "n_concurrent_trials"):
            preflight(
                unsafe,
                available_memory_bytes=8 * 1024**3,
                port_available=lambda _: True,
                job_text=unsafe.read_text(encoding="utf-8").replace(
                    "n_concurrent_trials: 1", "n_concurrent_trials: 2"
                ),
            )

    def test_rejects_non_qwen35_model(self) -> None:
        job = ROOT / "evals/harbor/jobs/niah-qwen35-local.yaml"
        with self.assertRaisesRegex(PreflightError, "Qwen3.5-only"):
            preflight(
                job,
                available_memory_bytes=8 * 1024**3,
                port_available=lambda _: True,
                job_text=job.read_text(encoding="utf-8").replace("Qwen3.5", "Qwen2.5"),
            )


if __name__ == "__main__":
    unittest.main()
