#!/usr/bin/env python3
"""Enforce that a source collection root is not itself a Git repository."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import sys


@dataclass(frozen=True)
class BoundaryResult:
    """Pure result for the non-Git super-root invariant."""

    root: str
    ok: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def check_non_git_root(root: Path) -> BoundaryResult:
    """Reject both normal and worktree-link Git metadata at the collection root."""

    resolved = root.expanduser().resolve()
    metadata = resolved / ".git"
    if metadata.exists() or metadata.is_symlink():
        return BoundaryResult(str(resolved), False, "ROOT_GIT_METADATA")
    return BoundaryResult(str(resolved), True, "ROOT_NOT_GIT")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    result = check_non_git_root(args.root)
    if args.json_output:
        print(json.dumps(result.as_dict(), sort_keys=True))
    else:
        print(f"[polyrepo] {result.reason}: {result.root}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
