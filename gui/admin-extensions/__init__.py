"""OMLX Admin Extensions — bridges the MLX research stack into the web admin GUI.

register_routes(app)
    Registers the research-panel blueprint with the OMLX admin Flask/FastAPI app
    so the developer dashboard can surface backend status, agent dispatch,
    and TurboQuant+ controls.

Usage
-----
    from gui.admin_extensions import register_routes
    register_routes(app)          # app is your Flask or FastAPI instance

Blueprint prefix: /api/research
"""

from __future__ import annotations

from typing import Any


def register_routes(app: Any) -> None:
    """Mount the research-panel blueprint onto *app*.

    Parameters
    ----------
    app : Flask | FastAPI
        The parent OMLX admin application instance.  Accepts any object that
        exposes ``.register_blueprint()`` (Flask) or ``.include_router()``
        (FastAPI).  Detection is automatic.

    Raises
    ------
    TypeError
        If *app* does not expose a recognised registration method.
    """
    # Late import so the package can be imported even if Flask is absent.
    from .api.research_panel import research_bp, research_router

    if hasattr(app, "register_blueprint"):
        # Flask-style admin app.
        app.register_blueprint(research_bp)
        return

    if hasattr(app, "include_router"):
        # FastAPI-style admin app.
        app.include_router(research_router)
        return

    if hasattr(app, "mount"):
        # Starlette-style bare ASGI app.
        from starlette.routing import Mount

        app.mount(
            "/api/research",
            Mount(research_bp),
        )
        return

    raise TypeError(
        f"Cannot register research routes on {type(app).__name__!r}: "
        f"expected Flask (register_blueprint), FastAPI (include_router), "
        f"or Starlette (mount)."
    )


__all__ = ["register_routes"]
