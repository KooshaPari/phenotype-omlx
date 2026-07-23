"""Tests for the doctor check registry system."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from omlx_research.cli._doctor_shared import Check, PASS, WARN, FAIL
from omlx_research.cli._doctor_registry import (
    register_check,
    get_all_checks,
    run_all_checks,
    _CHECK_REGISTRY,
    _ensure_all_modules_imported,
)


def _load_all_checks():
    _ensure_all_modules_imported()


_load_all_checks()

_SKIP_CHECK_NAMES = frozenset(
    {
        "tests_runnable",
        "doctor_check_count_at_least_18",
        "eval_harness_subcommand_runnable",
        "mlx_lm",
    }
)


def _run_checks_safe():
    removed = []
    for entry in list(_CHECK_REGISTRY):
        if entry[1] in _SKIP_CHECK_NAMES:
            _CHECK_REGISTRY.remove(entry)
            removed.append(entry)
    try:
        return run_all_checks(), removed
    finally:
        _CHECK_REGISTRY.extend(removed)


class TestCheckSharedTypes:
    def test_check_dataclass_fields(self):
        c = Check(id="x", description="y", status=PASS)
        assert c.id == "x"
        assert c.description == "y"
        assert c.status == PASS
        assert c.details == ""

    def test_check_with_details(self):
        c = Check(id="x", description="y", status=FAIL, details="reason here")
        assert c.details == "reason here"

    def test_status_constants_are_strings(self):
        assert PASS == "pass"
        assert WARN == "warn"
        assert FAIL == "fail"


class TestRegisterCheckDecorator:
    def test_decorator_registers_function(self):
        original_len = len(_CHECK_REGISTRY)

        @register_check(priority=50, name="test_dummy_check")
        def _dummy():
            return Check(id="test_dummy_check", description="dummy", status=PASS)

        assert len(_CHECK_REGISTRY) == original_len + 1
        entry = _CHECK_REGISTRY[-1]
        assert entry[0] == 50
        assert entry[1] == "test_dummy_check"
        assert callable(entry[2])

    def test_decorator_without_parentheses(self):
        original_len = len(_CHECK_REGISTRY)

        @register_check
        def _bare():
            return Check(id="bare", description="bare check", status=PASS)

        entry = _CHECK_REGISTRY[-1]
        assert entry[1] == "_bare"

    def test_decorator_default_priority_is_100(self):
        original_len = len(_CHECK_REGISTRY)

        @register_check(name="default_pri")
        def _default_pri():
            return Check(id="default_pri", description="", status=PASS)

        entry = _CHECK_REGISTRY[-1]
        assert entry[0] == 100

    def test_registered_function_return_value_unchanged(self):
        @register_check
        def _noop():
            return Check(id="noop", description="noop", status=PASS)

        result = _noop()
        assert isinstance(result, Check)
        assert result.id == "noop"


class TestGetAllChecks:
    def test_returns_nonempty_list(self):
        checks = get_all_checks()
        assert len(checks) > 0, "Registry should have at least one check"

    def test_all_entries_are_callable(self):
        checks = get_all_checks()
        for check_fn in checks:
            assert callable(check_fn), f"Check {check_fn} is not callable"

    def test_registry_includes_python_version_check(self):
        checks = get_all_checks()
        names = [fn.__name__ for fn in checks]
        assert "python_version" in names

    def test_registry_includes_mlx_core_check(self):
        checks = get_all_checks()
        names = [fn.__name__ for fn in checks]
        assert "mlx_core" in names

    def test_registry_includes_airlock_v2_check(self):
        checks = get_all_checks()
        names = [fn.__name__ for fn in checks]
        assert "airlock_v2" in names

    def test_registry_has_at_least_10_checks(self):
        checks = get_all_checks()
        assert len(checks) >= 10

    def test_registry_includes_kernel_registry_version_check(self):
        checks = get_all_checks()
        names = [fn.__name__ for fn in checks]
        assert "kernel_registry_version" in names


class TestRunAllChecks:
    def test_returns_list(self):
        results, _ = _run_checks_safe()
        assert isinstance(results, list)

    def test_all_results_are_check_instances(self):
        results, _ = _run_checks_safe()
        for r in results:
            assert isinstance(r, Check), f"Expected Check, got {type(r)}"

    def test_each_result_has_valid_status(self):
        results, _ = _run_checks_safe()
        valid_statuses = {PASS, WARN, FAIL}
        for r in results:
            assert r.status in valid_statuses

    def test_each_result_has_nonempty_id(self):
        results, _ = _run_checks_safe()
        for r in results:
            assert r.id

    def test_results_count_matches_nonsubprocess_checks(self):
        results, removed = _run_checks_safe()
        full_count = len(get_all_checks())
        assert len(results) == full_count - len(removed)

    def test_python_version_check_present_in_results(self):
        results, _ = _run_checks_safe()
        ids = [r.id for r in results]
        assert "python_version" in ids

    def test_run_all_checks_survives_raising_check(self):
        @register_check(priority=9999, name="test_crasher_check")
        def _crasher():
            raise RuntimeError("intentional test crash")

        removed = []
        for entry in list(_CHECK_REGISTRY):
            if entry[1] in _SKIP_CHECK_NAMES:
                _CHECK_REGISTRY.remove(entry)
                removed.append(entry)

        try:
            results = run_all_checks()
            crasher_results = [r for r in results if r.id == "_crasher"]
            assert len(crasher_results) == 1, (
                f"Expected crasher check in results, got ids: {[r.id for r in results[-5:]]}"
            )
            assert crasher_results[0].status == FAIL
            assert "RuntimeError" in crasher_results[0].details
        finally:
            _CHECK_REGISTRY[:] = [
                e for e in _CHECK_REGISTRY if e[1] != "test_crasher_check"
            ]
            _CHECK_REGISTRY.extend(removed)
