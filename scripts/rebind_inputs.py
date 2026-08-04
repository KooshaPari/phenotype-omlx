"""Immutable JSON input snapshots for candidate rebind preparation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


SHA256_LENGTH = 64


class CandidateRebindError(ValueError):
    """Raised when evidence cannot safely be prepared for review."""


@dataclass(frozen=True)
class InputSnapshot:
    """Validated JSON bytes and their parsed object."""

    path: Path
    document: dict[str, Any]
    sha256: str


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise CandidateRebindError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _reject_nonfinite(value: str) -> Any:
    raise CandidateRebindError(f"non-finite JSON constant is not allowed: {value}")


def _read_regular_bytes(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as exc:
        raise CandidateRebindError(f"{label} does not exist as a regular file") from exc
    except OSError as exc:
        raise CandidateRebindError(f"{label} must be a regular file") from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise CandidateRebindError(f"{label} must be a regular file")
            return handle.read()
    except OSError as exc:
        raise CandidateRebindError(f"cannot read {label}") from exc


def load_json_snapshot(path: Path, label: str) -> InputSnapshot:
    """Read a regular JSON file once so later replacement cannot alter the record."""

    raw = _read_regular_bytes(path, label)
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, CandidateRebindError) as exc:
        raise CandidateRebindError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise CandidateRebindError(f"{label} root must be an object")
    return InputSnapshot(path=path, document=document, sha256=hashlib.sha256(raw).hexdigest())


def canonical_digest(document: Mapping[str, Any]) -> str:
    """Return the canonical digest used by the rebind evidence contract."""

    payload = {key: value for key, value in document.items() if key != "integrity"}
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CandidateRebindError("record cannot be canonicalized") from exc
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value: Any) -> bool:
    """Return whether a value is a lowercase SHA-256 hex digest."""

    return isinstance(value, str) and len(value) == SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def repository_file_descriptor(
    repo_root: Path, item: Mapping[str, Any], label: str
) -> dict[str, str]:
    """Validate a repository-local regular file and preserve its declared digest."""

    raw_path = item.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise CandidateRebindError(f"{label} path must be non-empty")
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise CandidateRebindError(f"{label} path must be repository-relative")
    candidate = repo_root / relative_path
    current = repo_root
    for component in relative_path.parts:
        current /= component
        if current.is_symlink():
            raise CandidateRebindError(f"{label} path must not traverse a symlink")
    try:
        candidate.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise CandidateRebindError(f"{label} path escapes repository root") from exc
    declared_digest = item.get("sha256")
    if not is_sha256(declared_digest):
        raise CandidateRebindError(f"{label} sha256 must be a lowercase SHA-256")
    actual_digest = hashlib.sha256(_read_regular_bytes(candidate, label)).hexdigest()
    if actual_digest != declared_digest:
        raise CandidateRebindError(f"{label} {raw_path} SHA-256 does not match")
    return {"path": raw_path, "sha256": declared_digest}
