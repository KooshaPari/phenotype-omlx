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
    def test_harbor_wrapper_runs_preflight_before_harbor(self) -> None:
        wrapper = (ROOT / "scripts/evals/run_via_harbor_local.sh").read_text(
            encoding="utf-8"
        )
        preflight = "qwen35_niah_preflight.py"
        self.assertIn(preflight, wrapper)
        self.assertLess(wrapper.index(preflight), wrapper.index("harbor run"))

    def test_accepts_bounded_qwen35_job_with_admissible_resources(self) -> None:
        plan = preflight(
            ROOT / "evals/harbor/jobs/niah-qwen35-local.yaml",
            available_memory_bytes=8 * 1024**3,
            port_available=lambda _: True,
        )

        self.assertEqual(plan.model, "mlx-community/Qwen3.5-0.8B-OptiQ-4bit")
        self.assertEqual(plan.port, 8766)
        self.assertFalse(plan.workload_executed)
        self.assertFalse(plan.model_loaded)

    def test_rejects_unavailable_port_without_starting_workload(self) -> None:
        with self.assertRaisesRegex(PreflightError, "port 8766 is unavailable"):
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

    def test_rejects_launcher_port_that_disagrees_with_job_endpoint(self) -> None:
        job = ROOT / "evals/harbor/jobs/niah-qwen35-local.yaml"
        with self.assertRaisesRegex(PreflightError, "port mismatch"):
            preflight(
                job,
                available_memory_bytes=8 * 1024**3,
                port_available=lambda _: True,
                port=8081,
            )

    def test_rejects_qwen35_lookalike_model(self) -> None:
        job = ROOT / "evals/harbor/jobs/niah-qwen35-local.yaml"
        with self.assertRaisesRegex(PreflightError, "approved Qwen3.5 model"):
            preflight(
                job,
                available_memory_bytes=8 * 1024**3,
                port_available=lambda _: True,
                job_text=job.read_text(encoding="utf-8").replace(
                    "mlx-community/Qwen3.5-0.8B-OptiQ-4bit",
                    "mlx-community/Qwen3.5-0.8B-OptiQ-4bit-unapproved",
                ),
            )

    def test_rejects_disagreement_between_job_model_fields(self) -> None:
        job = ROOT / "evals/harbor/jobs/niah-qwen35-local.yaml"
        with self.assertRaisesRegex(PreflightError, "approved Qwen3.5 model"):
            preflight(
                job,
                available_memory_bytes=8 * 1024**3,
                port_available=lambda _: True,
                job_text=job.read_text(encoding="utf-8").replace(
                    'OMLX_READY_MODEL: "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"',
                    'OMLX_READY_MODEL: "mlx-community/Qwen3.5-0.8B-OptiQ-4bit-unapproved"',
                ),
            )


if __name__ == "__main__":
    unittest.main()
