"""Structured error helper for missing optional dependencies.

When a CLI subcommand reaches into a runtime that isn't installed in
the current environment (most commonly ``mlx_lm`` on non-Apple-Silicon
hosts or in CI), we want the user to see a message that tells them
exactly what to install — not a bare ``ImportError: No module named
'mlx_lm'``.

The single public helper here, :func:`require_mlx_lm`, encapsulates
that contract:

* It performs ``import mlx_lm`` once and caches the resolution in the
  module-level :data:`_cache` so subsequent calls don't re-pay the
  import cost.
* On a successful resolution it returns the module so callers can use
  it directly (``from omlx_research.cli._missing_dep import
  require_mlx_lm; mlx_lm = require_mlx_lm(__name__)``).
* On failure it raises :class:`RuntimeError` with a structured
  multi-line message that:

  - names the call site (``{where}``);
  - lists the install hint for both ``mlx-lm`` and ``mlx-core`` (Apple
    Silicon needs both);
  - points users who want to *skip* ``mlx_lm`` at the
    ``omlx-research doctor`` subcommand or the ``--no-mlx-lm`` flag.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional

__all__ = ["require_mlx_lm"]


# Module-level cache. ``None`` means "not resolved yet"; a
# :class:`types.ModuleType` means "mlx_lm is importable". Failures are
# deliberately *not* cached, so a fresh ``pip install`` takes effect
# without restarting the process.
_cache: Optional[object] = None


_INSTALL_HINT_TEMPLATE = """\
mlx_lm is required for {where}.

Install with:
    pip install mlx-lm

On Apple Silicon, also ensure mlx-core is installed:
    pip install mlx-core

To run without mlx_lm (decode-path only), use the
`omlx-research doctor` subcommand or `--no-mlx-lm` flag."""


def require_mlx_lm(where: str) -> ModuleType:
    """Import ``mlx_lm`` or raise a structured :class:`RuntimeError`.

    Args:
        where: Human-readable call-site identifier (e.g.
            ``"omlx-research run"`` or ``__name__``). Interpolated into
            the error message so the user can see which command path
            triggered the dependency requirement.

    Returns:
        The ``mlx_lm`` module on success.

    Raises:
        RuntimeError: when ``mlx_lm`` cannot be imported. The message
            follows the multi-line template above.
    """
    global _cache
    # If we've already resolved successfully, return the cached module.
    # NOTE: read via globals() so we tolerate the test fixture
    # resetting the attribute (which would otherwise raise NameError
    # in this function's frame).
    cached = globals().get("_cache")
    if cached is not None:
        return cached  # type: ignore[return-value]

    try:
        mod = importlib.import_module("mlx_lm")
    except Exception as e:  # noqa: BLE001 — any import-time error is fatal here
        raise RuntimeError(
            _INSTALL_HINT_TEMPLATE.format(where=where)
            + f"\n\n(Underlying import error: {type(e).__name__}: {e})"
        ) from e

    _cache = mod
    return mod