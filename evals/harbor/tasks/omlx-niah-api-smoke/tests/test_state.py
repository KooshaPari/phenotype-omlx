import json


def test_niah_exact_match():
    with open("/app/niah_result.json", encoding="utf-8") as f:
        d = json.load(f)
    assert d["exact_match"] is True
    assert "42-alpha" in open("/app/niah_answer.txt", encoding="utf-8").read()
