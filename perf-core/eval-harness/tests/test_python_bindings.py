"""Tests for the eval-harness PyO3 bindings.

Build the extension first:
    cd perf-core/eval-harness
    maturin develop --features python

Then run:
    pytest tests/test_python_bindings.py -v
"""

import json
import pytest

from _eval_harness import (
    Suite,
    TaskSpec,
    TaskResult,
    EvaluationReport,
    DatasetProvenance,
    SuiteReportEntry,
    MultiSuiteReport,
    Criteria,
    normalize_answer,
    score_exact,
    score_choice,
    evaluate_task,
    validate_report,
    ingest_report,
    run_suite_summary,
)


# ── normalize_answer ───────────────────────────────────────────────────────


class TestNormalizeAnswer:
    def test_basic(self):
        assert normalize_answer("  Hello  ") == "hello"

    def test_strips_trailing_punctuation(self):
        assert normalize_answer("answer.") == "answer"

    def test_lowercases(self):
        assert normalize_answer("YES") == "yes"

    def test_empty(self):
        assert normalize_answer("") == ""


# ── score_exact ────────────────────────────────────────────────────────────


class TestScoreExact:
    def test_match(self):
        assert score_exact("hello", "hello") is True

    def test_case_insensitive(self):
        assert score_exact("Hello", "hello") is True

    def test_whitespace_insensitive(self):
        assert score_exact("  hello  ", "hello") is True

    def test_mismatch(self):
        assert score_exact("hello", "world") is False


# ── score_choice ───────────────────────────────────────────────────────────


class TestScoreChoice:
    def test_parenthesized_match(self):
        assert score_choice("The answer is (b)", "B", 4) is True

    def test_standalone_letter(self):
        assert score_choice("B.", "B", 4) is True

    def test_mismatch(self):
        assert score_choice("A", "B", 4) is False

    def test_zero_choices(self):
        assert score_choice("A", "B", 0) is False


# ── Suite ──────────────────────────────────────────────────────────────────


class TestSuite:
    def test_from_str(self):
        assert Suite.from_str("mmlu") == Suite.Mmlu
        assert Suite.from_str("gpqa") == Suite.Gpqa
        assert Suite.from_str("terminal-bench") == Suite.TerminalBench
        assert Suite.from_str("perplexity") == Suite.Perplexity

    def test_from_str_invalid(self):
        with pytest.raises(ValueError, match="unknown suite"):
            Suite.from_str("invalid")

    def test_as_str(self):
        assert Suite.Mmlu.as_str() == "mmlu"
        assert Suite.Gpqa.as_str() == "gpqa"

    def test_repr(self):
        assert repr(Suite.Mmlu) == "Suite.mmlu"

    def test_equality(self):
        assert Suite.Mmlu == Suite.Mmlu
        assert Suite.Mmlu != Suite.Gpqa


# ── PyTaskSpec ─────────────────────────────────────────────────────────────


class TestTaskSpec:
    def test_multiple_choice(self):
        t = TaskSpec.multiple_choice(
            "q1", Suite.Mmlu, "What is 2+2?", ["3", "4", "5"], "B"
        )
        assert t.id == "q1"
        assert t.suite == Suite.Mmlu
        assert t.prompt == "What is 2+2?"
        assert t.choices == ["3", "4", "5"]
        assert t.expected == "B"
        assert t.is_multiple_choice() is True

    def test_open_ended(self):
        t = TaskSpec.open_ended("q2", Suite.Gpqa, "Explain gravity", expected="force")
        assert t.id == "q2"
        assert t.suite == Suite.Gpqa
        assert t.choices == []
        assert t.is_multiple_choice() is False

    def test_with_criteria(self):
        c = Criteria(
            expected_commands=["ls"],
            required_output=["file.txt"],
            forbidden_output=["error"],
        )
        t = TaskSpec.open_ended("q3", Suite.TerminalBench, "list files", criteria=c)
        assert t.criteria is not None
        assert t.criteria.expected_commands == ["ls"]


# ── evaluate_task ──────────────────────────────────────────────────────────


class TestEvaluateTask:
    def test_correct_exact(self):
        t = TaskSpec.open_ended("q1", Suite.Mmlu, "prompt", expected="answer")
        r = evaluate_task(t, "answer")
        assert r.correct is True
        assert r.score == 1.0

    def test_incorrect_exact(self):
        t = TaskSpec.open_ended("q1", Suite.Mmlu, "prompt", expected="answer")
        r = evaluate_task(t, "wrong")
        assert r.correct is False
        assert r.score == 0.0

    def test_mcq_correct(self):
        t = TaskSpec.multiple_choice("q1", Suite.Mmlu, "Question", ["A", "B", "C"], "B")
        r = evaluate_task(t, "(b)")
        assert r.correct is True

    def test_mcq_incorrect(self):
        t = TaskSpec.multiple_choice("q1", Suite.Mmlu, "Question", ["A", "B", "C"], "B")
        r = evaluate_task(t, "(a)")
        assert r.correct is False

    def test_criteria_scoring(self):
        c = Criteria(
            required_output=["success"],
            forbidden_output=["error"],
        )
        t = TaskSpec.open_ended("q1", Suite.TerminalBench, "do task", criteria=c)
        r = evaluate_task(t, "task completed successfully")
        assert r.correct is True

        r2 = evaluate_task(t, "task failed with error")
        assert r2.correct is False


# ── PyEvaluationReport ─────────────────────────────────────────────────────


class TestEvaluationReport:
    def test_from_results(self):
        r1 = TaskResult(
            task_id="q1",
            suite=Suite.Mmlu,
            prompt_tokens=5,
            completion_tokens=1,
            completion="A",
            normalized_completion="a",
            correct=True,
            score=1.0,
            latency_ms=10.0,
            matched_answer=None,
        )
        r2 = TaskResult(
            task_id="q2",
            suite=Suite.Mmlu,
            prompt_tokens=5,
            completion_tokens=1,
            completion="B",
            normalized_completion="b",
            correct=False,
            score=0.0,
            latency_ms=20.0,
            matched_answer=None,
        )
        report = EvaluationReport.from_results(Suite.Mmlu, [r1, r2])
        assert report.suite == Suite.Mmlu
        assert report.task_count == 2
        assert report.correct_count == 1
        assert report.accuracy == 0.5

    def test_json_roundtrip(self):
        r1 = TaskResult(
            task_id="q1",
            suite=Suite.Mmlu,
            prompt_tokens=1,
            completion_tokens=1,
            completion="x",
            normalized_completion="x",
            correct=True,
            score=1.0,
            latency_ms=1.0,
            matched_answer=None,
        )
        report = EvaluationReport.from_results(Suite.Mmlu, [r1])
        json_str = report.to_json()
        restored = EvaluationReport.from_json(json_str)
        assert restored.task_count == report.task_count
        assert restored.accuracy == report.accuracy

    def test_empty_report(self):
        report = EvaluationReport.from_results(Suite.Gpqa, [])
        assert report.task_count == 0
        assert report.accuracy == 0.0


# ── validate_report ────────────────────────────────────────────────────────


class TestValidateReport:
    def test_valid_report(self):
        r1 = TaskResult(
            task_id="q1",
            suite=Suite.Mmlu,
            prompt_tokens=1,
            completion_tokens=1,
            completion="x",
            normalized_completion="x",
            correct=True,
            score=1.0,
            latency_ms=1.0,
            matched_answer=None,
        )
        report = EvaluationReport.from_results(Suite.Mmlu, [r1])
        errors = validate_report(report)
        assert errors == []

    def test_invalid_task_count(self):
        report = EvaluationReport(Suite.Mmlu, 5, 1, 0.2, 0.2, [])
        errors = validate_report(report)
        assert any("task_count" in e for e in errors)

    def test_invalid_correct_count(self):
        r1 = TaskResult(
            task_id="q1",
            suite=Suite.Mmlu,
            prompt_tokens=1,
            completion_tokens=1,
            completion="x",
            normalized_completion="x",
            correct=True,
            score=1.0,
            latency_ms=1.0,
            matched_answer=None,
        )
        report = EvaluationReport.from_results(Suite.Mmlu, [r1])
        report.correct_count = 0
        errors = validate_report(report)
        assert any("correct_count" in e for e in errors)


# ── ingest_report ──────────────────────────────────────────────────────────


class TestIngestReport:
    def test_valid_json(self):
        r1 = TaskResult(
            task_id="q1",
            suite=Suite.Mmlu,
            prompt_tokens=1,
            completion_tokens=1,
            completion="x",
            normalized_completion="x",
            correct=True,
            score=1.0,
            latency_ms=1.0,
            matched_answer=None,
        )
        report = EvaluationReport.from_results(Suite.Mmlu, [r1])
        json_str = report.to_json()
        restored, errors = ingest_report(json_str)
        assert errors == []
        assert restored.task_count == 1

    def test_invalid_json(self):
        with pytest.raises(ValueError):
            ingest_report("not json")


# ── run_suite_summary ──────────────────────────────────────────────────────


class TestRunSuiteSummary:
    def test_valid_results(self):
        r1 = TaskResult(
            task_id="q1",
            suite=Suite.Mmlu,
            prompt_tokens=1,
            completion_tokens=1,
            completion="x",
            normalized_completion="x",
            correct=True,
            score=1.0,
            latency_ms=1.0,
            matched_answer=None,
        )
        report = run_suite_summary(Suite.Mmlu, [r1])
        assert report.task_count == 1
        assert report.accuracy == 1.0

    def test_empty_results(self):
        report = run_suite_summary(Suite.Gpqa, [])
        assert report.task_count == 0


# ── DatasetProvenance ──────────────────────────────────────────────────────


class TestDatasetProvenance:
    def test_construction(self):
        prov = DatasetProvenance(
            "https://example.com/data.csv",
            "v1.0",
            "test",
            b"content",
            10,
        )
        assert prov.source == "https://example.com/data.csv"
        assert prov.source_revision == "v1.0"
        assert prov.split == "test"
        assert prov.task_count == 10
        assert len(prov.sha256) == 64


# ── MultiSuiteReport ───────────────────────────────────────────────────────


class TestMultiSuiteReport:
    def test_from_reports(self):
        r1 = TaskResult(
            task_id="q1",
            suite=Suite.Mmlu,
            prompt_tokens=1,
            completion_tokens=1,
            completion="x",
            normalized_completion="x",
            correct=True,
            score=1.0,
            latency_ms=1.0,
            matched_answer=None,
        )
        prov = DatasetProvenance("src", "v1", "test", b"x", 1)
        entry = SuiteReportEntry(prov, EvaluationReport.from_results(Suite.Mmlu, [r1]))
        multi = MultiSuiteReport.from_reports([entry])
        assert multi.task_count == 1
        assert multi.correct_count == 1
        assert multi.overall_accuracy == 1.0

    def test_json_roundtrip(self):
        r1 = TaskResult(
            task_id="q1",
            suite=Suite.Mmlu,
            prompt_tokens=1,
            completion_tokens=1,
            completion="x",
            normalized_completion="x",
            correct=True,
            score=1.0,
            latency_ms=1.0,
            matched_answer=None,
        )
        prov = DatasetProvenance("src", "v1", "test", b"x", 1)
        entry = SuiteReportEntry(prov, EvaluationReport.from_results(Suite.Mmlu, [r1]))
        multi = MultiSuiteReport.from_reports([entry])
        json_str = multi.to_json()
        restored = MultiSuiteReport.from_json(json_str)
        assert restored.task_count == multi.task_count
        assert restored.overall_accuracy == multi.overall_accuracy
