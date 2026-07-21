"""Verifier helper for omlx Qwen3.5 policy Harbor task."""


def test_policy_marker_exists():
    with open("/app/policy_ok.txt", encoding="utf-8") as f:
        assert f.read().strip() == "qwen35-ssot-ok"
