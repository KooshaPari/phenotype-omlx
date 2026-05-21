# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmark ecosystem adapters."""

import pytest

from omlx.eval import BENCHMARKS
from omlx.eval.dsl import BenchmarkDSL, BenchmarkSuite, standard_suite
from omlx.eval.inspect_bridge import InspectBridge
from omlx.eval.registry import BenchmarkPackage, BenchmarkRegistry
from omlx.eval.swe_bench import SWEBenchBenchmark
from omlx.eval.terminal_bench import TerminalBenchBenchmark, _has_dangerous_command


class TestSWEBenchBenchmark:
    def test_extract_answer_from_diff_block(self):
        bench = SWEBenchBenchmark()
        response = (
            "Here is the fix:\n"
            "```diff\n"
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
            "```"
        )
        assert bench.extract_answer(response, {}).startswith("diff --git")

    def test_check_answer_accepts_patch_shape(self):
        bench = SWEBenchBenchmark()
        patch = (
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
        )
        assert bench.check_answer(patch, {}) is True

    def test_registered(self):
        assert BENCHMARKS["swe_bench"] is SWEBenchBenchmark


class TestTerminalBenchBenchmark:
    def test_extract_answer_from_shell_block(self):
        bench = TerminalBenchBenchmark()
        response = "```bash\nls\ngrep -R needle .\n```"
        assert bench.extract_answer(response, {}) == "ls\ngrep -R needle ."

    def test_check_answer_matches_expected_commands(self):
        bench = TerminalBenchBenchmark()
        item = {"commands": ["grep -R needle src", "cat result.txt"]}
        assert bench.check_answer("grep -R needle src\ncat result.txt", item) is True

    def test_rejects_dangerous_command(self):
        assert _has_dangerous_command("rm -rf /") is True
        bench = TerminalBenchBenchmark()
        assert bench.check_answer("rm -rf /", {"commands": ["ls"]}) is False

    def test_registered(self):
        assert BENCHMARKS["terminal_bench"] is TerminalBenchBenchmark


class TestBenchmarkDSL:
    def test_build_and_round_trip(self, tmp_path):
        suite = (
            BenchmarkDSL("custom")
            .describe("demo")
            .add("terminal", "terminal_bench", sample_size=3, weight=2.0, data_path="x.jsonl")
            .build()
        )
        path = tmp_path / "suite.yaml"
        suite.save(path)
        loaded = BenchmarkSuite.load(path)
        assert loaded.name == "custom"
        assert loaded.steps[0].params == {"data_path": "x.jsonl"}

    def test_standard_suite(self):
        suite = standard_suite("code")
        assert any(step.benchmark == "swe_bench" for step in suite.steps)


class TestInspectBridge:
    def test_sample_and_result_conversion(self):
        bench = TerminalBenchBenchmark()
        item = {"task_id": "t1", "description": "List files", "commands": ["ls"]}
        bridge = InspectBridge(bench)
        sample = bridge.sample_from_item(item)
        result = bridge.result_from_output(sample, "ls")
        assert sample.id == "t1"
        assert result.correct is True


class TestBenchmarkRegistry:
    def test_register_get_search_remove(self, tmp_path):
        registry = BenchmarkRegistry(tmp_path / "registry.json", auto_persist=False)
        package = BenchmarkPackage(name="demo", entry_point="pkg:Bench", description="Demo benchmark")
        registry.register(package)
        assert registry.get("demo") == package
        assert registry.search("demo") == [package]
        registry.remove("demo")
        with pytest.raises(KeyError):
            registry.get("demo")