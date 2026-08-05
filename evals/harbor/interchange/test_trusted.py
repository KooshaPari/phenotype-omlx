"""Tests for the strict Trusted Harbor Envelope v1 consumer."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from . import trusted
from .trusted import TrustedHarborEnvelopeError, TrustedHarborPolicy, verify_envelope


MODEL = "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
MODEL_CONFIG_SHA256 = "a" * 64
JOB_ID = "job-0001"
TRIAL_ID = "trial-0001"
KEY_ID = "fixture-ed25519-v1"


def _canonical_payload(document: dict) -> bytes:
    payload = {key: value for key, value in document.items() if key != "signature"}
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _private_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(bytes(range(32)))


def _policy() -> TrustedHarborPolicy:
    public_key = _private_key().public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return TrustedHarborPolicy(
        trusted_keys={KEY_ID: public_key},
        expected_issuer="harbor-langfuse",
        expected_model=MODEL,
        expected_model_config_sha256=MODEL_CONFIG_SHA256,
        expected_task_id="omlx/niah-api-smoke",
        expected_environment="apple-container",
        expected_context_tokens=8192,
        expected_candidate_repo="phenotype-omlx",
        expected_branch="feature/trusted-envelope",
        expected_source_head="e" * 40,
    )


def _sign(document: dict, key_id: str = KEY_ID) -> dict:
    signed = deepcopy(document)
    signed.pop("signature", None)
    payload = _canonical_payload(signed)
    signed["signature"] = {
        "alg": "Ed25519",
        "key_id": key_id,
        "signed_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "signature": _private_key().sign(payload).hex(),
    }
    return signed


def _envelope() -> dict:
    return _sign(
        {
            "schema_version": "trusted-harbor-envelope/v1",
            "issuer": {
                "name": "harbor-langfuse",
                "environment": "apple-container",
                "portage_commit": "b" * 40,
                "exporter_version": "1.0.0",
                "key_id": KEY_ID,
            },
            "harbor": {
                "job_id": JOB_ID,
                "trial_id": TRIAL_ID,
                "trial_name": "omlx-niah-api-smoke__fixture",
                "task_id": "omlx/niah-api-smoke",
                "job_config_sha256": "c" * 64,
                "result_sha256": "d" * 64,
                "immutable_result_uri": "harbor://results/job-0001/trial-0001",
                "n_trials": 1,
                "requested_context_tokens": 8192,
            },
            "run": {
                "model": MODEL,
                "model_config_sha256": MODEL_CONFIG_SHA256,
                "candidate_repo": "phenotype-omlx",
                "branch": "feature/trusted-envelope",
                "source_head": "e" * 40,
                "started_at": "2026-08-05T00:00:00Z",
                "finished_at": "2026-08-05T00:01:00Z",
            },
            "authorization": {
                "window_id": "qwen35-fixture-window",
                "sidecar_sha256": "f" * 64,
                "immutable_auth_uri": "harbor://authorization/qwen35-fixture-window",
            },
            "artifacts": [
                {
                    "uri": "harbor://artifacts/job-0001/trial-0001/result.json",
                    "sha256": "1" * 64,
                    "byte_count": 128,
                }
            ],
            "observability": {
                "langfuse_session_id": JOB_ID,
                "trace_id": TRIAL_ID,
            },
        }
    )


def test_accepts_valid_signed_qwen35_harbor_envelope() -> None:
    envelope = verify_envelope(_envelope(), _policy())

    assert envelope.attestation_verified is True
    assert envelope.harbor_job_id == JOB_ID
    assert envelope.harbor_trial_id == TRIAL_ID


def test_rejects_unsigned_external_evidence() -> None:
    document = _envelope()
    document.pop("signature")

    with pytest.raises(TrustedHarborEnvelopeError, match="signature"):
        verify_envelope(document, _policy())


def test_rejects_unknown_field_even_when_signed() -> None:
    document = _envelope()
    document["untrusted_extension"] = "must-not-pass"
    document = _sign(document)

    with pytest.raises(TrustedHarborEnvelopeError, match="unknown field"):
        verify_envelope(document, _policy())


def test_rejects_tampered_signed_payload() -> None:
    document = _envelope()
    document["harbor"]["result_sha256"] = "0" * 64

    with pytest.raises(TrustedHarborEnvelopeError, match="signed payload SHA-256"):
        verify_envelope(document, _policy())


def test_rejects_unknown_signing_key() -> None:
    document = _envelope()
    document["issuer"]["key_id"] = "unknown-key"
    document = _sign(document, key_id="unknown-key")

    with pytest.raises(TrustedHarborEnvelopeError, match="trusted key"):
        verify_envelope(document, _policy())


def test_rejects_qwen25_even_when_signature_is_valid() -> None:
    document = _envelope()
    document["run"]["model"] = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    document = _sign(document)

    with pytest.raises(TrustedHarborEnvelopeError, match="model"):
        verify_envelope(document, _policy())


def test_rejects_signed_evidence_from_a_stale_source_head() -> None:
    document = _envelope()
    document["run"]["source_head"] = "0" * 40
    document = _sign(document)

    with pytest.raises(TrustedHarborEnvelopeError, match="source head"):
        verify_envelope(document, _policy())


def test_rejects_harbor_langfuse_identifier_mismatch() -> None:
    document = _envelope()
    document["observability"]["langfuse_session_id"] = "another-job"
    document = _sign(document)

    with pytest.raises(TrustedHarborEnvelopeError, match="session"):
        verify_envelope(document, _policy())


def test_rejects_non_exact_context_contract() -> None:
    document = _envelope()
    document["harbor"]["requested_context_tokens"] = 4096
    document = _sign(document)

    with pytest.raises(TrustedHarborEnvelopeError, match="context"):
        verify_envelope(document, _policy())


def test_rejects_mutable_or_unbound_artifact_uri() -> None:
    document = _envelope()
    document["artifacts"][0]["uri"] = "file:///tmp/result.json"
    document = _sign(document)

    with pytest.raises(TrustedHarborEnvelopeError, match="artifact.uri"):
        verify_envelope(document, _policy())


def test_rejects_result_uri_not_bound_to_harbor_job_and_trial() -> None:
    document = _envelope()
    document["harbor"]["immutable_result_uri"] = "harbor://results/other-job/other-trial"
    document = _sign(document)

    with pytest.raises(TrustedHarborEnvelopeError, match="result URI"):
        verify_envelope(document, _policy())


def test_loads_and_verifies_regular_signed_envelope_file(tmp_path) -> None:
    path = tmp_path / "trusted-envelope.json"
    path.write_text(json.dumps(_envelope()), encoding="utf-8")

    envelope = trusted.load_verified_envelope(path, _policy())

    assert envelope.harbor_job_id == JOB_ID
    assert envelope.harbor_trial_id == TRIAL_ID


def test_loader_rejects_duplicate_json_keys_before_verification(tmp_path) -> None:
    path = tmp_path / "duplicate-envelope.json"
    raw = json.dumps(_envelope())[:-1] + ',"schema_version":"trusted-harbor-envelope/v1"}'
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(TrustedHarborEnvelopeError, match="duplicate JSON key"):
        trusted.load_verified_envelope(path, _policy())


def test_loader_rejects_nonfinite_json_constants(tmp_path) -> None:
    path = tmp_path / "nonfinite-envelope.json"
    path.write_text('{"schema_version":NaN}', encoding="utf-8")

    with pytest.raises(TrustedHarborEnvelopeError, match="non-finite JSON constant"):
        trusted.load_verified_envelope(path, _policy())


def test_loader_rejects_non_utf8_envelope_file(tmp_path) -> None:
    path = tmp_path / "invalid-encoding-envelope.json"
    path.write_bytes(b"\xff")

    with pytest.raises(TrustedHarborEnvelopeError, match="valid UTF-8 JSON"):
        trusted.load_verified_envelope(path, _policy())


def test_loader_fails_closed_without_no_follow_support(tmp_path, monkeypatch) -> None:
    path = tmp_path / "trusted-envelope.json"
    path.write_text(json.dumps(_envelope()), encoding="utf-8")
    monkeypatch.delattr(trusted.os, "O_NOFOLLOW")

    with pytest.raises(TrustedHarborEnvelopeError, match="no-follow"):
        trusted.load_verified_envelope(path, _policy())


@pytest.mark.skipif(os.name == "nt", reason="symlink semantics are POSIX-specific")
def test_loader_rejects_symlinked_envelope_file(tmp_path) -> None:
    target = tmp_path / "target-envelope.json"
    target.write_text(json.dumps(_envelope()), encoding="utf-8")
    link = tmp_path / "linked-envelope.json"
    link.symlink_to(target)

    with pytest.raises(TrustedHarborEnvelopeError, match="regular file"):
        trusted.load_verified_envelope(link, _policy())
