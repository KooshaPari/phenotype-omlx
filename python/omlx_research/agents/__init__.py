"""Concurrent multi-agent execution: LatentMAS, TiDAR, SSD, JetSpec.

Each agent is an adapter that wraps a third-party model and exposes a uniform
`async def step(prompt, state)` interface, so the same scheduler can fan out,
chain, or fall back between them.
"""

from .latentmas_runner import LatentMasRunner, latentmas_fanout
from .tidar_runner import TidarRunner, tidar_ar_diffusion_loop
from .ssd_runner import SsdRunner
from .jetspec_runner import JetSpecRunner, jetspec_draft_tree
from .scheduler import ConcurrentScheduler, Strategy

__all__ = [
    "LatentMasRunner",
    "latentmas_fanout",
    "TidarRunner",
    "tidar_ar_diffusion_loop",
    "SsdRunner",
    "JetSpecRunner",
    "jetspec_draft_tree",
    "ConcurrentScheduler",
    "Strategy",
]
