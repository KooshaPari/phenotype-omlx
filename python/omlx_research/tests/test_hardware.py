"""Evidence and topology tests for the non-invasive hardware probe."""

from __future__ import annotations

from omlx_research import hardware


def test_cpu_topology_darwin_uses_performance_levels(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    values = {
        "hw.physicalcpu": 10,
        "hw.perflevel0.logicalcpu": 8,
        "hw.perflevel1.logicalcpu": 2,
    }
    monkeypatch.setattr(hardware, "_sysctl_int", values.get)

    assert hardware._detect_cpu_topology() == (10, 8, 2, "sysctl")


def test_optional_accelerator_fields_are_unknown_without_evidence(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware, "_detect_apple_silicon", lambda: "Apple M1 Pro")
    monkeypatch.setattr(hardware, "_detect_metal", lambda: True)
    monkeypatch.setattr(hardware, "_detect_cuda", lambda: (False, ""))
    monkeypatch.setattr(hardware, "_detect_cpu_topology", lambda: (10, 8, 2, "sysctl"))
    monkeypatch.setattr(hardware, "_detect_apple_metal", lambda: (None, None, None))
    monkeypatch.setattr(hardware, "_detect_thermal_state", lambda: (None, None))
    monkeypatch.setattr(hardware, "_detect_memory_pressure", lambda: (None, None))
    monkeypatch.setattr(hardware, "_memory_total_gb", lambda: 16.0)

    profile = hardware.detect_hardware()

    assert profile.cpu_count_physical == 10
    assert profile.cpu_performance_cores == 8
    assert profile.cpu_efficiency_cores == 2
    assert profile.neural_engine_available is None
    assert profile.npu_available is None
    assert profile.metal_device_name is None
    assert profile.capability_evidence == {"cpu_topology": "sysctl"}


def test_summary_remains_compatible_with_new_optional_fields():
    profile = hardware.HardwareProfile(
        os="Linux",
        arch="x86_64",
        cpu_count_logical=16,
        cpu_count_physical=8,
        memory_total_gb=32.0,
    )

    assert "Linux/x86_64" in profile.summary()
    assert "32.0 GB RAM" in profile.summary()
