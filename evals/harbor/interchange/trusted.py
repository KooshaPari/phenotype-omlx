"""Fail-closed consumer for a signed Trusted Harbor Envelope v1.

This module verifies an issuer-signed evidence envelope.  It intentionally does
not sign local Harbor output: a local signature cannot turn execution metadata
into an independently attested run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


class TrustedHarborEnvelopeError(ValueError):
    """Raised when an envelope fails structural, policy, or signature checks."""


@dataclass(frozen=True)
class TrustedHarborPolicy:
    """Explicit trust anchor and non-negotiable Qwen3.5 execution contract."""

    trusted_keys: Mapping[str, bytes]
    expected_issuer: str
    expected_model: str
    expected_model_config_sha256: str
    expected_task_id: str
    expected_environment: str
    expected_context_tokens: int


@dataclass(frozen=True)
class VerifiedTrustedHarborEnvelope:
    """Minimal trusted projection for promotion-only consumers."""

    attestation_verified: bool
    harbor_job_id: str
    harbor_trial_id: str
    source_head: str
    signed_payload_sha256: str


_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "issuer",
        "harbor",
        "run",
        "authorization",
        "artifacts",
        "observability",
        "signature",
    }
)
_ISSUER_FIELDS = frozenset({"name", "environment", "portage_commit", "exporter_version", "key_id"})
_HARBOR_FIELDS = frozenset(
    {
        "job_id",
        "trial_id",
        "trial_name",
        "task_id",
        "job_config_sha256",
        "result_sha256",
        "immutable_result_uri",
        "n_trials",
        "requested_context_tokens",
    }
)
_RUN_FIELDS = frozenset(
    {
        "model",
        "model_config_sha256",
        "candidate_repo",
        "branch",
        "source_head",
        "started_at",
        "finished_at",
    }
)
_AUTHORIZATION_FIELDS = frozenset({"window_id", "sidecar_sha256", "immutable_auth_uri"})
_ARTIFACT_FIELDS = frozenset({"uri", "sha256", "byte_count"})
_OBSERVABILITY_FIELDS = frozenset({"langfuse_session_id", "trace_id"})
_SIGNATURE_FIELDS = frozenset({"alg", "key_id", "signed_payload_sha256", "signature"})


def _canonical_json(document: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrustedHarborEnvelopeError("envelope is not canonical JSON") from exc


def _require_mapping(value: Any, name: str, allowed: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrustedHarborEnvelopeError(f"{name} must be an object")
    extras = set(value) - allowed
    missing = allowed - set(value)
    if extras:
        raise TrustedHarborEnvelopeError(f"{name} has unknown field(s): {sorted(extras)}")
    if missing:
        raise TrustedHarborEnvelopeError(f"{name} is missing field(s): {sorted(missing)}")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrustedHarborEnvelopeError(f"{name} must be a non-empty string")
    return value


def _require_digest(value: Any, name: str, length: int = 64) -> str:
    digest = _require_string(value, name)
    if len(digest) != length or any(character not in "0123456789abcdef" for character in digest):
        raise TrustedHarborEnvelopeError(f"{name} must be a lowercase {length}-hex digest")
    return digest


def _require_harbor_uri(value: Any, name: str) -> str:
    uri = _require_string(value, name)
    if not uri.startswith("harbor://"):
        raise TrustedHarborEnvelopeError(f"{name} must use an immutable harbor:// URI")
    return uri


def _require_timestamp(value: Any, name: str) -> datetime:
    raw = _require_string(value, name)
    if not raw.endswith("Z"):
        raise TrustedHarborEnvelopeError(f"{name} must be a UTC timestamp")
    try:
        return datetime.fromisoformat(f"{raw[:-1]}+00:00")
    except ValueError as exc:
        raise TrustedHarborEnvelopeError(f"{name} must be an ISO-8601 timestamp") from exc


def _validate_shape(document: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    root = _require_mapping(document, "envelope", _ROOT_FIELDS)
    if root["schema_version"] != "trusted-harbor-envelope/v1":
        raise TrustedHarborEnvelopeError("unsupported schema_version")
    issuer = _require_mapping(root["issuer"], "issuer", _ISSUER_FIELDS)
    harbor = _require_mapping(root["harbor"], "harbor", _HARBOR_FIELDS)
    run = _require_mapping(root["run"], "run", _RUN_FIELDS)
    authorization = _require_mapping(root["authorization"], "authorization", _AUTHORIZATION_FIELDS)
    observability = _require_mapping(root["observability"], "observability", _OBSERVABILITY_FIELDS)
    signature = _require_mapping(root["signature"], "signature", _SIGNATURE_FIELDS)
    artifacts = root["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise TrustedHarborEnvelopeError("artifacts must be a non-empty list")
    for artifact in artifacts:
        _require_mapping(artifact, "artifact", _ARTIFACT_FIELDS)
    return issuer, harbor, run, authorization, observability, signature


def _validate_policy(
    issuer: Mapping[str, Any],
    harbor: Mapping[str, Any],
    run: Mapping[str, Any],
    authorization: Mapping[str, Any],
    observability: Mapping[str, Any],
    policy: TrustedHarborPolicy,
) -> None:
    if issuer["name"] != policy.expected_issuer:
        raise TrustedHarborEnvelopeError("issuer is not trusted by policy")
    if issuer["environment"] != policy.expected_environment:
        raise TrustedHarborEnvelopeError("Harbor environment does not match policy")
    _require_digest(issuer["portage_commit"], "issuer.portage_commit", length=40)
    _require_string(issuer["exporter_version"], "issuer.exporter_version")
    _require_string(issuer["key_id"], "issuer.key_id")
    if run["model"] != policy.expected_model:
        raise TrustedHarborEnvelopeError("model does not match exact Qwen3.5 policy")
    if run["model_config_sha256"] != policy.expected_model_config_sha256:
        raise TrustedHarborEnvelopeError("model configuration digest does not match policy")
    _require_digest(run["source_head"], "run.source_head", length=40)
    _require_string(run["candidate_repo"], "run.candidate_repo")
    _require_string(run["branch"], "run.branch")
    if _require_timestamp(run["finished_at"], "run.finished_at") < _require_timestamp(
        run["started_at"], "run.started_at"
    ):
        raise TrustedHarborEnvelopeError("run timestamps are out of order")
    if harbor["task_id"] != policy.expected_task_id:
        raise TrustedHarborEnvelopeError("Harbor task does not match policy")
    matches_context = harbor["requested_context_tokens"] == policy.expected_context_tokens
    if harbor["n_trials"] != 1 or not matches_context:
        raise TrustedHarborEnvelopeError("Harbor trial or context contract does not match policy")
    job_id = _require_string(harbor["job_id"], "harbor.job_id")
    trial_id = _require_string(harbor["trial_id"], "harbor.trial_id")
    _require_string(harbor["trial_name"], "harbor.trial_name")
    _require_digest(harbor["job_config_sha256"], "harbor.job_config_sha256")
    _require_digest(harbor["result_sha256"], "harbor.result_sha256")
    _require_harbor_uri(harbor["immutable_result_uri"], "harbor.immutable_result_uri")
    window_id = _require_string(authorization["window_id"], "authorization.window_id")
    _require_digest(authorization["sidecar_sha256"], "authorization.sidecar_sha256")
    auth_uri = _require_harbor_uri(
        authorization["immutable_auth_uri"], "authorization.immutable_auth_uri"
    )
    if not auth_uri.endswith(f"/{window_id}"):
        raise TrustedHarborEnvelopeError("authorization URI does not bind its window")
    identifiers_match = (
        observability["langfuse_session_id"] == job_id
        and observability["trace_id"] == trial_id
    )
    if not identifiers_match:
        raise TrustedHarborEnvelopeError(
            "Langfuse session or trace does not bind Harbor identifiers"
        )


def verify_envelope(
    document: Mapping[str, Any], policy: TrustedHarborPolicy
) -> VerifiedTrustedHarborEnvelope:
    """Verify a strict signed envelope against an explicit public-key trust policy."""
    issuer, harbor, run, authorization, observability, signature = _validate_shape(document)
    _validate_policy(issuer, harbor, run, authorization, observability, policy)
    if signature["alg"] != "Ed25519":
        raise TrustedHarborEnvelopeError("unsupported signature algorithm")
    key_id = _require_string(signature["key_id"], "signature.key_id")
    if issuer["key_id"] != key_id:
        raise TrustedHarborEnvelopeError("issuer and signature key IDs do not match")
    public_key_bytes = policy.trusted_keys.get(key_id)
    if public_key_bytes is None:
        raise TrustedHarborEnvelopeError("signature key is not a trusted key")
    payload = {key: value for key, value in document.items() if key != "signature"}
    canonical_payload = _canonical_json(payload)
    payload_digest = hashlib.sha256(canonical_payload).hexdigest()
    if signature["signed_payload_sha256"] != payload_digest:
        raise TrustedHarborEnvelopeError("signed payload SHA-256 does not match envelope")
    try:
        signature_bytes = bytes.fromhex(
            _require_string(signature["signature"], "signature.signature")
        )
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(
            signature_bytes, canonical_payload
        )
    except (InvalidSignature, ValueError) as exc:
        raise TrustedHarborEnvelopeError("signature verification failed") from exc
    return VerifiedTrustedHarborEnvelope(
        attestation_verified=True,
        harbor_job_id=harbor["job_id"],
        harbor_trial_id=harbor["trial_id"],
        source_head=run["source_head"],
        signed_payload_sha256=payload_digest,
    )
