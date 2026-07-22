"""FR-5 / E1–E4 tests: Julia, non-legacy paths, evidence-class discipline.

Acceptance:
- E1: ``julia`` required on the eval/doctor path — FAIL (not WARN) if missing.
- E2: ``scripts/niah_benchmark.py`` must not use legacy ``phenotype-omlx/python``.
- E4: committed ``niah_results.json`` must declare synthetic evidence class;
  live smoke must refuse to overwrite that envelope.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

from omlx_research.cli import _doctor_checks as checks_mod
from omlx_research.cli import _doctor_extra_niah as niah_extra
from omlx_research.cli._doctor_shared import FAIL, PASS, project_root


def test_julia_required_fails_when_missing(monkeypatch):
    """E1: missing ``julia`` on PATH ⇒ doctor check FAIL (loud)."""

    monkeypatch.setattr(niah_extra.shutil, "which", lambda _name: None)
    c = checks_mod.julia_required_on_eval_path()
    assert c.status == FAIL
    assert c.id == "julia_required_on_eval_path"
    assert "julia" in c.details.lower()


def test_julia_required_passes_when_present(monkeypatch):
    """E1: ``julia`` on PATH ⇒ PASS and surface the resolved binary."""

    monkeypatch.setattr(
        niah_extra.shutil, "which", lambda _name: "/opt/homebrew/bin/julia"
    )

    class _FakeProc:
        returncode = 0
        stdout = "julia version 1.12.6\n"
        stderr = ""

    monkeypatch.setattr(
        niah_extra.subprocess,
        "run",
        lambda *a, **kw: _FakeProc(),
    )
    c = checks_mod.julia_required_on_eval_path()
    assert c.status == PASS
    assert "/opt/homebrew/bin/julia" in c.details
    assert "1.12.6" in c.details


def test_niah_benchmark_source_has_no_legacy_python_path():
    """E2: committed NIAH script must not hardcode legacy absolute repos path."""

    root = Path(project_root())
    script = root / "scripts" / "niah_benchmark.py"
    text = script.read_text(encoding="utf-8")
    assert 'REPO / "phenotype-omlx/python"' not in text
    assert 'Path("/Users/kooshapari/CodeProjects/Phenotype/repos")' not in text
    assert 'ROOT / "python"' in text


def test_niah_benchmark_legacy_path_doctor_fails(monkeypatch, tmp_path):
    """Doctor FAIL when the on-disk NIAH script still embeds the legacy path."""

    bad = tmp_path / "niah_benchmark.py"
    bad.write_text(
        'import sys\n'
        'from pathlib import Path\n'
        'REPO = Path("/Users/kooshapari/CodeProjects/Phenotype/repos")\n'
        'sys.path.insert(0, str(REPO / "phenotype-omlx/python"))\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(niah_extra, "_find_niah_benchmark", lambda: str(bad))
    c = niah_extra.niah_benchmark_non_legacy_path()
    assert c.status == FAIL
    assert "legacy" in c.details.lower() or "phenotype-omlx/python" in c.details


def test_niah_benchmark_legacy_path_doctor_passes_for_repo_relative(monkeypatch, tmp_path):
    """Doctor PASS when the script uses repo-relative ``<root>/python``."""

    good = tmp_path / "niah_benchmark.py"
    good.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "ROOT = Path(__file__).resolve().parents[1]\n"
        'sys.path.insert(0, str(ROOT / "python"))\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(niah_extra, "_find_niah_benchmark", lambda: str(good))
    c = niah_extra.niah_benchmark_non_legacy_path()
    assert c.status == PASS


def test_niah_results_committed_envelope_is_synthetic_not_live():
    """FR-5 E4: committed niah_results.json must declare synthetic evidence class."""

    root = Path(project_root())
    path = root / "niah_results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("evidence_label") == "synthetic_target_rows"
    assert data.get("synthetic") is True
    assert data.get("reported") is True
    assert data.get("evidence_label") != "live_verified"
    assert len(data.get("targets") or []) >= 25


def test_niah_live_artifact_must_not_overwrite_envelope_name():
    """FR-5 E4: live smoke refuses to target niah_results.json."""

    script = Path(project_root()) / "scripts" / "niah_server_smoke.py"
    spec = importlib.util.spec_from_file_location("niah_server_smoke", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main(["--output", "niah_results.json", "--server-url", "http://127.0.0.1:9"])
    assert rc == 2


def test_niah_qwen35_live_rejects_qwen25():
    """FR-5 E3: helper must refuse Qwen2.5 / non-Qwen3.5 model ids."""

    import importlib.util

    script = Path(project_root()) / "scripts" / "niah_qwen35_live.py"
    spec = importlib.util.spec_from_file_location("niah_qwen35_live", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        mod._require_qwen35("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        raise AssertionError("expected SystemExit")
    except SystemExit as e:
        assert "Qwen3.5" in str(e)
    mod._require_qwen35("Qwen/Qwen3.5-0.8B")  # no raise


def test_niah_qwen35_live_artifact_is_qwen35():
    """FR-5 E3 acceptance artifact must be Qwen3.5 live_verified."""

    path = Path(project_root()) / "research" / "fr5_niah_qwen35_live.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "Qwen3.5" in data["model"]
    assert "Qwen2.5" not in data["model"]
    assert data["evidence_label"] == "live_verified"
    assert data["exact_match"] is True
    assert data.get("architecture_caveat")
    assert data.get("kv_modes_applicable") is False


def test_niah_prompt_ids_hit_requested_token_length():
    """NIAH prompt construction must produce the requested token count exactly."""

    import importlib.util

    script = Path(project_root()) / "scripts" / "niah_benchmark.py"
    spec = importlib.util.spec_from_file_location("niah_benchmark", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class FakeTokenizer:
        def encode(self, text):
            return list(range(len(text.split())))

    prompt_ids = mod.build_needle_prompt_ids(
        FakeTokenizer(), 128, "the secret code is 123-456-alpha"
    )
    assert len(prompt_ids) == 128


def test_niah_does_not_call_removed_turbo_cache_compress_api():
    """TurboKVCache compression must be reported through its current API only."""

    script = Path(project_root()) / "scripts" / "niah_benchmark.py"
    text = script.read_text(encoding="utf-8")
    assert "c.compress()" not in text
    assert "measure_turbo_cache_metrics" in text
    assert "compressed_layers" in text
    assert "byte_reduction_effective" in text
    assert "kv_fp16_baseline_bytes" in text
    assert "make_sampler(temp=0.0)" in text


def test_measure_turbo_cache_metrics_full_attn_only():
    """FR-5 E3: packed state alone is not enough — need resident < FP16 baseline."""

    import importlib.util

    script = Path(project_root()) / "scripts" / "niah_benchmark.py"
    spec = importlib.util.spec_from_file_location("niah_benchmark", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _Arr:
        def __init__(self, nbytes: int):
            self.nbytes = nbytes

    class TurboKVCache:  # noqa: N801 — mirror runtime type name
        def __init__(self, *, compressed: bool, packed: int = 0, raw: int = 0, resident: int = 0):
            self._is_compressed = compressed
            self._packed_keys = _Arr(packed) if packed else None
            self._packed_values = None
            self._key_norms = None
            self._value_norms = None
            self._raw_keys = _Arr(raw) if raw else None
            self._raw_values = None
            self._pending_raw_keys = []
            self._pending_raw_values = []
            self._inner_kv = None
            self._resident = resident

        @property
        def nbytes(self) -> int:
            return self._resident

    class KVCache:
        pass

    caches = [
        KVCache(),
        TurboKVCache(compressed=False, raw=1000, resident=1000),
        TurboKVCache(compressed=True, packed=200, raw=0, resident=400),
        KVCache(),
    ]
    # Without baseline: packed state visible but byte reduction unproven
    m0 = mod.measure_turbo_cache_metrics(caches, fp16_baseline_bytes=0)
    assert m0["turbo_layers"] == 2
    assert m0["compressed_layers"] == 1
    assert m0["kv_packed_bytes"] == 200
    assert m0["packed_state_present"] is True
    assert m0["byte_reduction_effective"] is False
    assert m0["compression_effective"] is False

    # With baseline larger than resident: byte reduction proven
    m1 = mod.measure_turbo_cache_metrics(caches, fp16_baseline_bytes=5000)
    assert m1["kv_turbo_resident_bytes"] == 1400
    assert m1["kv_fp16_baseline_bytes"] == 5000
    assert m1["byte_reduction_effective"] is True
    assert m1["compression_effective"] is True
    assert m1["byte_reduction_ratio"] == 1400 / 5000

    empty = mod.measure_turbo_cache_metrics([KVCache(), KVCache()], fp16_baseline_bytes=100)
    assert empty["turbo_layers"] == 0
    assert empty["byte_reduction_effective"] is False


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_instrumented_envelope_is_schema_v2_byte_reduction():
    """On-disk compression proof must stay schema v2 with byte_reduction_any."""

    root = Path(project_root())
    path = root / "research" / "fr5_niah_qwen35_0_8b_instrumented.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert data["packed_state_any"] is True
    assert data["byte_reduction_any"] is True
    assert data["evidence_label"] == "live_verified"
    assert _sha256(path) == (
        "862bc610c16a3e4b4b637e85670bf493e24a82a31adfc88c7efd7a5a7d6407d3"
    )


def test_committed_answer128_exact_retrieval_not_compression_proof():
    """Argis answer128 packs prove exact retrieval only — no compression metrics."""

    root = Path(project_root())
    cases = [
        (
            "fr5_niah_qwen35_4b_coder_authorized_answer128.json",
            "170b1bef2b200b80a90ad9ad2f96826bec304f0b9d28a1510e4d1341cf35b2b9",
            1,
        ),
        (
            "fr5_niah_qwen35_4b_coder_authorized_answer128_2048.json",
            "4ee9f73573c30cb0d8763ceb39e35736967f2af856b3722aa96c25faec2154bc",
            4,
        ),
    ]
    compression_keys = {
        "byte_reduction_any",
        "packed_state_any",
        "kv_turbo_resident_bytes",
        "byte_reduction_effective",
    }
    for name, digest, exact_count in cases:
        path = root / "research" / name
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema_version"] == 1
        assert data["evidence_label"] == "live_verified"
        assert compression_keys.isdisjoint(data.keys())
        assert sum(1 for r in data["results"] if r.get("exact_match")) == exact_count
        assert _sha256(path) == digest


def test_evidence_index_roles_match_committed_artifacts():
    """Index maps compression vs exact-retrieval roles without rewriting packs."""

    root = Path(project_root())
    index = json.loads(
        (root / "research" / "fr5_qwen35_evidence_index.json").read_text(encoding="utf-8")
    )
    by_path = {a["path"]: a for a in index["artifacts"]}
    instr = by_path["research/fr5_niah_qwen35_0_8b_instrumented.json"]
    assert instr["compression_proof"] is True
    a512 = by_path["research/fr5_niah_qwen35_4b_coder_authorized_answer128.json"]
    assert a512["compression_proof"] is False
    assert a512["role"] == "exact_retrieval"


def test_configure_hf_env_keeps_offline_for_local_model_path(monkeypatch):
    """Local absolute --model must not force HF_HUB_OFFLINE=0."""

    script = Path(project_root()) / "scripts" / "niah_benchmark.py"
    spec = importlib.util.spec_from_file_location("niah_benchmark_hf", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    local = "/Users/kooshapari/.omlx/models/example-local"
    mod.configure_hf_env(local)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    mod.configure_hf_env("mlx-community/Some-Hub-Model")
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "0"
