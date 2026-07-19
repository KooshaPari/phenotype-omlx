"""omlx-research CLI commands.

Each subcommand lives in its own module to keep the main ``cli/__init__.py``
file small. Modules expose a ``cmd_<name>(args: argparse.Namespace) -> int``
function that returns the process exit code.

Re-exports are kept intentionally minimal — only the ``cmd_*`` callables are
exposed because that's all ``main()`` needs.
"""

from .inspect import cmd_inspect
from .explain import cmd_explain
from .tune import cmd_tune
from .replay import cmd_replay
from .compare import cmd_compare
from .evidence import cmd_evidence

__all__ = [
    "cmd_inspect",
    "cmd_explain",
    "cmd_tune",
    "cmd_replay",
    "cmd_compare",
    "cmd_evidence",
]
