"""Tests for strict Harbor result and artifact URI binding."""

from __future__ import annotations

import pytest

from .trusted import (
    TrustedHarborEnvelopeError,
    _require_exact_harbor_authorization_uri,
    _require_exact_harbor_result_uri,
    _require_harbor_artifact_uri,
)


@pytest.mark.parametrize(
    ("job_id", "trial_id"),
    [("job?override=true", "trial-0001"), ("job-0001", "trial-0001#fragment")],
)
def test_rejects_query_or_fragment_in_harbor_identifiers(job_id: str, trial_id: str) -> None:
    with pytest.raises(TrustedHarborEnvelopeError, match="identifier"):
        _require_exact_harbor_result_uri(f"harbor://results/{job_id}/{trial_id}", job_id, trial_id)


def test_rejects_percent_encoded_artifact_path_ambiguity() -> None:
    with pytest.raises(TrustedHarborEnvelopeError, match="artifact URI"):
        _require_harbor_artifact_uri(
            "harbor://artifacts/job-0001/trial-0001/result%2Fother.json",
            "job-0001",
            "trial-0001",
        )


@pytest.mark.parametrize(
    "uri",
    [
        "harbor://authorization/window-0001?override=true",
        "harbor://authorization/window-0001#fragment",
        "harbor://results/window-0001",
    ],
)
def test_rejects_noncanonical_authorization_uri_authority_or_suffix(uri: str) -> None:
    with pytest.raises(TrustedHarborEnvelopeError, match="authorization URI"):
        _require_exact_harbor_authorization_uri(uri, "window-0001")


@pytest.mark.parametrize(
    "uri",
    [
        "harbor://authorization/window%2D0001",
        "harbor://authorization/window-0001/extra",
    ],
)
def test_rejects_percent_encoded_or_split_authorization_uri_path(uri: str) -> None:
    with pytest.raises(TrustedHarborEnvelopeError, match="authorization URI"):
        _require_exact_harbor_authorization_uri(uri, "window-0001")


@pytest.mark.parametrize(
    "uri",
    [
        "harbor://authorization/window-0001?",
        "harbor://authorization/window-0001#",
        " harbor://authorization/window-0001",
        "harbor://authorization/window-0001 ",
        "harbor://authorization/window-0001\n",
        "HARBOR://authorization/window-0001",
        "harbor://AUTHORIZATION/window-0001",
    ],
)
def test_rejects_noncanonical_authorization_uri_serialization(uri: str) -> None:
    with pytest.raises(TrustedHarborEnvelopeError, match="authorization URI"):
        _require_exact_harbor_authorization_uri(uri, "window-0001")
