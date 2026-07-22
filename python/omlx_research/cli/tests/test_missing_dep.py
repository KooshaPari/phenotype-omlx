"""Tests for the structured ``mlx_lm``-missing error helper.

These tests pin the contract for
``omlx_research.cli._missing_dep.require_mlx_lm``:

- raises a ``RuntimeError`` (not a bare ``ImportError``) when ``mlx_lm``
  is not importable in the current interpreter;
- the message includes the install hint for the *call site* passed in
  (``{where}`` is interpolated);
- subsequent calls reuse the cached result so we don't re-pay the import
  cost on every invocation.
"""

from __future__ import annotations

import sys
import types

import pytest


# `_missing_dep` is a tiny module — import once at module load. If it
# doesn't exist yet (TDD), the test file fails fast with ImportError
# and the developer knows to implement the helper.
from omlx_research.cli._missing_dep import require_mlx_lm


@pytest.fixture
def fresh_missing_dep():
    """Reset the module-level cache between tests so we don't leak state.

    The cache lives on the module as `_cache`. We capture it on entry
    and restore it on exit so tests are independent of run order.
    """
    import omlx_research.cli._missing_dep as mod
    saved = getattr(mod, "_cache", None)
    # Drop any cached resolution.
    if hasattr(mod, "_cache"):
        del mod._cache
    yield mod
    if saved is not None:
        mod._cache = saved
    elif hasattr(mod, "_cache"):
        del mod._cache


def _hide_mlx_lm(monkeypatch):
    """Force `import mlx_lm` to raise ImportError for the duration of a test."""
    monkeypatch.delitem(sys.modules, "mlx_lm", raising=False)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)


def _expose_mlx_lm(monkeypatch):
    """Inject a minimal `mlx_lm` module so import succeeds without the real package."""
    fake = types.ModuleType("mlx_lm")
    fake.__version__ = "0.0.0-test"
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)


def test_require_mlx_lm_raises_when_missing(monkeypatch, fresh_missing_dep):
    """When mlx_lm can't be imported, raise RuntimeError (not bare ImportError)."""
    _hide_mlx_lm(monkeypatch)
    with pytest.raises(RuntimeError) as cm:
        require_mlx_lm("omlx-research run")
    msg = str(cm.value)
    assert "mlx_lm is required" in msg
    assert "pip install mlx-lm" in msg


def test_require_mlx_lm_returns_module_when_present(monkeypatch, fresh_missing_dep):
    """When mlx_lm is importable, return the module object (not None)."""
    _expose_mlx_lm(monkeypatch)
    mod = require_mlx_lm("omlx-research run")
    assert mod is sys.modules["mlx_lm"]
    # __version__ should be visible from the fake.
    assert mod.__version__ == "0.0.0-test"


def test_require_mlx_lm_caches_resolution(monkeypatch, fresh_missing_dep):
    """After a successful resolution, the cached module is reused on later calls.

    We inject a fake module on the first call, then *break* the module
    table (set mlx_lm = None) before the second call. The second call
    must still return the cached module without re-importing.
    """
    _expose_mlx_lm(monkeypatch)
    first = require_mlx_lm("omlx-research serve")
    assert first is sys.modules["mlx_lm"]

    # Break subsequent `import mlx_lm` so a fresh resolution would fail.
    monkeypatch.setitem(sys.modules, "mlx_lm", None)

    second = require_mlx_lm("omlx-research serve")
    assert second is first, "require_mlx_lm must cache and not re-import"


def test_require_mlx_lm_error_message_includes_install_hint(monkeypatch, fresh_missing_dep):
    """The structured message must tell the user how to install mlx_lm."""
    _hide_mlx_lm(monkeypatch)
    with pytest.raises(RuntimeError) as cm:
        require_mlx_lm("omlx-research eval")
    msg = str(cm.value)
    # Both the pip-install line and the mlx-core follow-up should appear.
    assert "pip install mlx-lm" in msg
    assert "mlx-core" in msg
    # And the doctor / --no-mlx-lm hint for the "decode-path only" path.
    assert "doctor" in msg or "--no-mlx-lm" in msg


def test_require_mlx_lm_error_message_names_call_site(monkeypatch, fresh_missing_dep):
    """The {where} argument must appear in the error message verbatim."""
    _hide_mlx_lm(monkeypatch)
    site = "omlx-research run --model foo"
    with pytest.raises(RuntimeError) as cm:
        require_mlx_lm(site)
    assert site in str(cm.value), (
        "call site passed to require_mlx_lm must appear in the error "
        "so users know which command triggered the dependency requirement"
    )


def test_require_mlx_lm_repeated_missing_calls_raise_consistently(monkeypatch, fresh_missing_dep):
    """Each call with mlx_lm missing must raise — no half-cached state."""
    _hide_mlx_lm(monkeypatch)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            require_mlx_lm("omlx-research run")