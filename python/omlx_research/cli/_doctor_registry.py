"""Doctor check registry — ``@register_check`` decorator + ``run_all_checks()``.

All doctor checks across the cli package self-register via the
:func:`register_check` decorator.  The orchestrator in
:mod:`omlx_research.cli.doctor` calls :func:`run_all_checks` instead
of maintaining a hand-curated ``CHECKS`` list.

Import ordering
---------------
``run_all_checks`` imports every sibling check module so all
``@register_check`` decorators fire before any check is executed.
Import order is irrelevant because the ``priority`` kwarg controls
execution order (lower runs first; meta-check uses a high priority
so it always sees the complete registry).
"""

from __future__ import annotations

import importlib
from typing import Callable, Optional

from ._doctor_shared import Check, FAIL

_CHECK_REGISTRY: list[tuple[int, str, Callable[[], Check]]] = []


def register_check(
    fn: Optional[Callable[[], Check]] = None,
    *,
    priority: int = 100,
    name: Optional[str] = None,
) -> Callable[[], Check]:
    """Decorator that registers a doctor check function.

    Parameters
    ----------
    fn:
        The check callable (when used without arguments).
    priority:
        Execution order — lower runs first.  The meta-check uses 900
        so it always observes the complete registry.
    name:
        Optional override for the check id shown in the report.
        Defaults to ``fn.__name__``.
    """

    def _decorator(func: Callable[[], Check]) -> Callable[[], Check]:
        check_name = name or func.__name__
        _CHECK_REGISTRY.append((priority, check_name, func))
        return func

    if fn is not None:
        return _decorator(fn)
    return _decorator


def get_all_checks() -> list[Callable[[], Check]]:
    """Return registered check functions sorted by priority."""
    return [fn for _pri, _name, fn in sorted(_CHECK_REGISTRY)]


def run_all_checks() -> list[Check]:
    """Import all check modules, then execute every registered check.

    Each check is wrapped in a broad ``Exception`` guard so a single
    broken check cannot abort the whole report — failures degrade to
    ``fail`` with the exception class name in the details.
    """
    _ensure_all_modules_imported()
    results: list[Check] = []
    for check_fn in get_all_checks():
        try:
            results.append(check_fn())
        except Exception as e:
            results.append(
                Check(
                    id=getattr(check_fn, "__name__", "unknown"),
                    description="(description unavailable — check raised)",
                    status=FAIL,
                    details=f"{type(e).__name__}: {e}",
                )
            )
    return results


def _ensure_all_modules_imported() -> None:
    """Force-import every sibling check module so decorators fire.

    Idempotent — repeated calls are no-ops after the first invocation.
    """
    if getattr(run_all_checks, "_modules_loaded", False):
        return
    _modules = [
        "._doctor_checks",
        "._doctor_extra_niah",
        "._doctor_extra_eval",
        "._doctor_extra_kernel",
        "._doctor_turn5_checks",
        "._doctor_internal_checks",
        "._doctor_internal_checks_split",
        "._doctor_internal_checks_turn12",
        "._doctor_meta_checks",
    ]
    for mod_name in _modules:
        try:
            importlib.import_module(mod_name, package="omlx_research.cli")
        except Exception:
            pass
    run_all_checks._modules_loaded = True  # type: ignore[attr-defined]
