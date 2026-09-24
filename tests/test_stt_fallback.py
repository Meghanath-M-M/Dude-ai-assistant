"""CUDA is the default STT device; a machine without it must not crash.

STTEngine falls back to CPU at *load* time (this file) and at *decode* time
(the cuBLAS retry in transcribe) — so flipping the default to cuda is safe
everywhere, while machines that do have the GPU stop launching whisper-small
on CPU by accident (field round 8 re-soak: "STT warm-up: 0.01s on cpu").
"""

import pytest

from nova_agent.config.settings import DEFAULT_STT_DEVICE, Settings
from nova_agent.core.stt import STTEngine


def test_stt_device_defaults_to_cuda():
    assert DEFAULT_STT_DEVICE == "cuda"
    assert Settings().stt_device == "cuda"


def test_stt_load_falls_back_to_cpu_when_cuda_fails(monkeypatch):
    import faster_whisper

    calls = []

    class FakeWhisper:
        def __init__(self, model_name, device="cpu", compute_type="int8"):
            calls.append((model_name, device, compute_type))
            if device == "cuda":
                raise RuntimeError("CUDA failed to initialize: no cuBLAS")
            self.model_name = model_name

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisper)

    with pytest.warns(RuntimeWarning, match="using CPU"):
        engine = STTEngine("small", device="cuda", compute_type="int8")

    assert calls == [("small", "cuda", "int8"), ("small", "cpu", "int8")]
    assert engine.device == "cpu"
    assert engine.compute_type == "int8"


def test_a_cpu_load_failure_still_raises(monkeypatch):
    import faster_whisper

    class BrokenWhisper:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("model files are gone")

    monkeypatch.setattr(faster_whisper, "WhisperModel", BrokenWhisper)

    with pytest.raises(RuntimeError, match="model files are gone"):
        STTEngine("small", device="cpu")
