"""Strict parser and validator for immutable Harbor result and artifact URIs."""

from __future__ import annotations

from typing import Any
from urllib.parse import SplitResult, urlsplit


class TrustedHarborEnvelopeError(ValueError):
    """Raised when an envelope fails structural, policy, or signature checks."""


_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~-")


def _require_uri(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrustedHarborEnvelopeError(f"{name} must be a non-empty string")
    return value


def _require_identifier(value: str, name: str) -> None:
    if value in {".", ".."} or not value or any(character not in _UNRESERVED for character in value):
        raise TrustedHarborEnvelopeError(f"{name} identifier must be an ASCII unreserved token")


def _parse_harbor_uri(uri: str, name: str, authority: str) -> SplitResult:
    try:
        parsed = urlsplit(uri)
        port = parsed.port
    except ValueError as exc:
        raise TrustedHarborEnvelopeError(f"{name} must be a valid immutable Harbor URI") from exc
    if parsed.scheme != "harbor" or parsed.netloc != authority:
        raise TrustedHarborEnvelopeError(f"{name} must use an immutable harbor:// URI")
    if parsed.username is not None or parsed.password is not None or port is not None:
        raise TrustedHarborEnvelopeError(f"{name} must not include URI credentials or a port")
    if parsed.query or parsed.fragment:
        raise TrustedHarborEnvelopeError(f"{name} must not include a query or fragment")
    return parsed


def _path_components(parsed: SplitResult, name: str, count: int) -> list[str]:
    components = parsed.path.split("/")
    if len(components) != count + 1 or components[0] or any(not component for component in components[1:]):
        raise TrustedHarborEnvelopeError(f"{name} must contain exactly {count} path components")
    return components[1:]


def require_harbor_uri(value: Any, name: str) -> str:
    """Validate the scheme for an immutable Harbor URI without binding its path."""
    uri = _require_uri(value, name)
    try:
        parsed = urlsplit(uri)
    except ValueError as exc:
        raise TrustedHarborEnvelopeError(f"{name} must be a valid immutable Harbor URI") from exc
    if parsed.scheme != "harbor":
        raise TrustedHarborEnvelopeError(f"{name} must use an immutable harbor:// URI")
    return uri


def require_exact_harbor_authorization_uri(value: Any, window_id: str) -> str:
    """Require a canonical Harbor authorization URI bound to the supplied window."""
    _require_identifier(window_id, "authorization window")
    uri = _require_uri(value, "authorization.immutable_auth_uri")
    components = _path_components(
        _parse_harbor_uri(uri, "authorization URI", "authorization"), "authorization URI", 1
    )
    canonical_uri = f"harbor://authorization/{window_id}"
    if components != [window_id] or uri != canonical_uri:
        raise TrustedHarborEnvelopeError(
            "authorization URI does not exactly bind its authorization window"
        )
    return uri


def require_exact_harbor_result_uri(value: Any, job_id: str, trial_id: str) -> str:
    """Require a canonical Harbor result URI bound to the supplied identifiers."""
    _require_identifier(job_id, "job")
    _require_identifier(trial_id, "trial")
    uri = _require_uri(value, "harbor.immutable_result_uri")
    components = _path_components(_parse_harbor_uri(uri, "result URI", "results"), "result URI", 2)
    if components != [job_id, trial_id]:
        raise TrustedHarborEnvelopeError("Harbor result URI does not exactly bind its job and trial")
    return uri


def require_harbor_artifact_uri(value: Any, job_id: str, trial_id: str) -> str:
    """Require a canonical Harbor artifact URI bound to the supplied identifiers."""
    _require_identifier(job_id, "job")
    _require_identifier(trial_id, "trial")
    uri = _require_uri(value, "artifact.uri")
    job, trial, artifact_name = _path_components(
        _parse_harbor_uri(uri, "artifact.uri", "artifacts"), "artifact URI", 3
    )
    _require_identifier(artifact_name, "artifact URI")
    if [job, trial] != [job_id, trial_id]:
        raise TrustedHarborEnvelopeError("artifact URI does not exactly bind its Harbor job and trial")
    return uri
