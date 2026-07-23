"""Extend installed MLX namespaces with the persistent TurboQuant layer.

The MLX wheel owns ``mlx.nn.layers`` as a regular package, so adding the
TurboQuant repository root to ``PYTHONPATH`` alone cannot expose
``mlx.nn.layers.turbo_kv_cache``.  Python imports ``sitecustomize`` at startup;
we append the audited persistent layer directory without shadowing MLX itself.
"""

from __future__ import annotations

import os
from pathlib import Path


def _extend_mlx_layers() -> None:
    layer_file = Path(
        os.environ.get(
            "OMLX_TURBOQUANT_LAYER",
            "~/.omlx/turboquant-plus/mlx/nn/layers/turbo_kv_cache.py",
        )
    ).expanduser()
    if not layer_file.is_file():
        return
    try:
        import mlx.nn.layers as layers
    except ImportError:
        return
    layer_root = str(layer_file.parent)
    search_path = getattr(layers, "__path__", None)
    if search_path is not None and layer_root not in search_path:
        search_path.append(layer_root)


_extend_mlx_layers()
