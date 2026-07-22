def test_turbo_ssot_marker():
    assert open("/app/turbo_ssot_ok.txt", encoding="utf-8").read() == "turbo-qwen35-ssot-ok"
