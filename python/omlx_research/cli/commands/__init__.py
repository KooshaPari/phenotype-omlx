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
from .gates import cmd_gates
from .promote import cmd_promote
from .quarantine import cmd_quarantine

__all__ = [
    "cmd_compare",
    "cmd_evidence",
    "cmd_explain",
    "cmd_gates",
    "cmd_inspect",
    "cmd_promote",
    "cmd_quarantine",
    "cmd_replay",
    "cmd_tune",
]
