"""omlx-research CLI entry-point shim that lets `python -m omlx_research` work."""
import sys
from . import main as _main

if __name__ == "__main__":
    sys.exit(_main())
