import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "evals"))
import niah_openai_smoke as smoke  # noqa: E402


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(
            {"choices": [{"message": {"content": "42-alpha"}}]}
        ).encode()


def test_niah_request_forces_sequential_mlx_path(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8766/v1")
    monkeypatch.setenv("OPENAI_MODEL", "Qwen3.5-test")
    monkeypatch.setattr(smoke.urllib.request, "urlopen", fake_urlopen)

    result = smoke.run_niah()

    assert result["exact_match"] is True
    assert captured["body"]["seed"] == 0
    assert captured["body"]["temperature"] == 0
