"""Re-export FR-7 dashboard entrypoints."""

from .server import build_status, health_payload, main, serve

__all__ = ["build_status", "health_payload", "main", "serve"]
