import numpy as np

from nova_agent.core.vad import VADRecorder
from nova_agent.core.wake_word import WakeWordEngine
from nova_agent.main import NovaAgent


def test_wake_word_threshold_defaults_follow_the_active_backend():
    assert WakeWordEngine.resolve_threshold(None, using_model=True) == 0.5
    assert WakeWordEngine.resolve_threshold(None, using_model=False) == 0.13
    assert WakeWordEngine.resolve_threshold(0.35, using_model=True) == 0.35


def test_wake_word_engine_uses_threshold(monkeypatch):
    engine = WakeWordEngine(model_path="dummy.onnx", threshold=0.5)
    monkeypatch.setattr(engine, "_predict_score", lambda _chunk: 0.8)

    assert engine.process_chunk(np.zeros(160, dtype=np.float32)) is True


def test_wake_word_engine_requires_a_strong_or_sustained_signal(monkeypatch):
    engine = WakeWordEngine(model_path="dummy.onnx", threshold=0.35)
    scores = iter([0.2, 0.3, 0.34, 0.6])
    monkeypatch.setattr(engine, "_predict_score", lambda _chunk: next(scores))

    assert engine.process_chunk(np.zeros(160, dtype=np.float32)) is False
    assert engine.process_chunk(np.zeros(160, dtype=np.float32)) is False
    assert engine.process_chunk(np.zeros(160, dtype=np.float32)) is False
    assert engine.process_chunk(np.zeros(160, dtype=np.float32)) is True


def test_wake_word_engine_detects_realistic_voice_energy():
    engine = WakeWordEngine(model_path="dummy.onnx", threshold=0.13)

    assert engine.process_chunk(np.full(160, 0.08, dtype=np.float32)) is True


def test_wake_word_engine_rejects_short_noise_bursts():
    engine = WakeWordEngine(model_path="dummy.onnx", threshold=0.35)
    scores = iter([0.4, 0.4, 0.4])

    class FakeMonkyPatch:
        def __init__(self):
            self.calls = 0

        def __call__(self, _chunk):
            self.calls += 1
            return next(scores)

    patcher = FakeMonkyPatch()
    engine._predict_score = patcher

    assert engine.process_chunk(np.zeros(160, dtype=np.float32)) is False
    assert engine.process_chunk(np.zeros(160, dtype=np.float32)) is False
    assert engine.process_chunk(np.zeros(160, dtype=np.float32)) is False


def test_wake_word_engine_calibrates_a_threshold_from_ambient_energy():
    audio = np.full(1600, 0.04, dtype=np.float32)
    recommended = WakeWordEngine.estimate_threshold(audio)

    assert recommended > 0.1
    assert recommended < 0.5


def test_wake_word_engine_falls_back_when_model_file_missing():
    engine = WakeWordEngine(model_path="assets/wake_word/absent_model.onnx", threshold=0.5)

    assert engine._oww_model is None
    assert engine.process_chunk(np.full(160, 0.6, dtype=np.float32)) is True


def test_nova_agent_debug_mode_reports_state(capsys):
    class FakeWakeEngine:
        def process_chunk(self, _chunk):
            return True

    class FakeVAD:
        def process_chunk(self, _chunk):
            return "command.wav"

    class FakeHUD:
        def set_state(self, state, label):
            pass

    agent = NovaAgent(wake_engine=FakeWakeEngine(), vad_reader=FakeVAD(), hud=FakeHUD(), debug=True)
    agent.process_chunk(np.full(160, 0.8, dtype=np.float32))

    captured = capsys.readouterr().out
    assert (
        "wake" in captured.lower()
        or "listening" in captured.lower()
        or "energy" in captured.lower()
    )


def test_vad_recording_finalizes_after_silence(tmp_path, monkeypatch):
    recorder = VADRecorder(sample_rate=16_000, max_silence=0)
    record_path = tmp_path / "command.wav"
    monkeypatch.setattr(recorder, "_write_audio", lambda audio: record_path)

    assert recorder.process_chunk(np.full(160, 0.5, dtype=np.float32)) is None
    result = recorder.process_chunk(np.zeros(160, dtype=np.float32))

    assert result == record_path
    assert recorder.is_recording is False


def test_vad_accepts_realistic_microphone_voice_energy():
    recorder = VADRecorder()

    assert recorder.process_chunk(np.full(160, 0.05, dtype=np.float32)) is None
    assert recorder.is_recording is True
