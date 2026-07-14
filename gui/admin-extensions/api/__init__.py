"""Research panel API module."""
from .research_panel import (
    research_bp,
    research_router,
    _BACKENDS,
    _AGENT_DESCRIPTIONS,
    _dispatch_agent,
    _backend_status,
    _all_backend_statuses,
    _turboquant_config,
    _turboquant_apply_config,
    _try_import_turboquant,
)

__all__ = [
    "research_bp",
    "research_router",
    "_BACKENDS",
    "_AGENT_DESCRIPTIONS",
    "_dispatch_agent",
    "_backend_status",
    "_all_backend_statuses",
    "_turboquant_config",
    "_turboquant_apply_config",
    "_try_import_turboquant",
]
