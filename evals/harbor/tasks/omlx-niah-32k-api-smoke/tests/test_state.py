import json


def test_niah_32k_exact_match():
    with open("/app/niah_result.json", encoding="utf-8") as f:
        d = json.load(f)
    assert d["exact_match"] is True
    assert d["requested_context_tokens"] == 32768
    assert d["prompt_tokens"] == 32768
    assert d["context_tokens_exact"] is True
    assert "42-alpha" in open("/app/niah_answer.txt", encoding="utf-8").read()
