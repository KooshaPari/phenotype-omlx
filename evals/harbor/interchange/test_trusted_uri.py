"""Tests for strict Harbor result and artifact URI binding."""

from __future__ import annotations

import pytest

from .trusted import (
    TrustedHarborEnvelopeError,
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
