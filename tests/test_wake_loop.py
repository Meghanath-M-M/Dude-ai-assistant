import numpy as np

from nova_agent.core.vad import SileroVAD, VADRecorder
from nova_agent.core.wake_word import WakeWordEngine
from nova_agent.main import NovaAgent


def test_wake_word_threshold_defaults_follow_the_active_backend():
    assert WakeWordEngine.resolve_threshold(None, using_model=True) == 0.65
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


def test_wake_word_scoring_failure_is_logged_not_swallowed(caplog):
    import logging

    engine = WakeWordEngine(model_path="dummy.onnx", threshold=0.5)

    class ExplodingModel:
        def __init__(self):
            self.prediction_buffer = {}

        def predict(self, _pcm):
            raise ValueError("bad tensor shape")

    engine._oww_model = ExplodingModel()
    with caplog.at_level(logging.WARNING):
        score = engine.observe(np.zeros(160, dtype=np.float32))

    assert "bad tensor shape" in caplog.text
    assert 0.0 <= score <= 1.0


def test_wake_word_uses_the_last_buffer_frame_not_the_max():
    engine = WakeWordEngine(model_path="dummy.onnx", threshold=0.5)

    class OneShotModel:
        def __init__(self):
            self.prediction_buffer = {"hey_nova": [0.9, 0.9, 0.2]}

        def predict(self, _pcm):
            self.prediction_buffer = {"hey_nova": [0.1, 0.1, 0.75]}

    engine._oww_model = OneShotModel()
    score = engine.observe(np.zeros(160, dtype=np.float32))

    # Stale peaks (0.9) must not count; only the just-scored frame (0.75).
    assert score == 0.75
    assert engine.last_score == 0.75


def test_wake_status_reports_backend_model_threshold_and_score():
    engine = WakeWordEngine(model_path="dummy.onnx", threshold=0.42)

    status = engine.status()

    assert status["backend"] == "energy"
    assert status["model_path"] == "dummy.onnx"
    assert status["threshold"] == 0.42
    assert status["last_score"] is None
    engine.observe(np.full(160, 0.1, dtype=np.float32))
    assert engine.status()["last_score"] is not None


def test_agent_refuses_capture_while_speaking():
    from nova_agent.main import NovaAgent

    class WakeThatFires:
        def process_chunk(self, _chunk):
            return True

        def reset(self):
            pass

    agent = NovaAgent(wake_engine=WakeThatFires(), processor=object(), hud_enabled=False)
    agent.is_speaking = True

    assert agent.process_chunk(np.full(160, 0.9, dtype=np.float32)) is None
    assert agent.listening is False


def test_nova_agent_debug_mode_reports_state(capsys):
    class FakeWakeEngine:
        def process_chunk(self, _chunk):
            return True

        def reset(self):
            pass

    class FakeVAD:
        def process_chunk(self, _chunk):
            return "command.wav"

        def reset(self):
            pass

    class FakeHUD:
        def set_state(self, state, label):
            pass

        def show_transcript(self, text):
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
    # No VAD model: this exercises the documented loudness fallback.
    recorder = VADRecorder(sample_rate=16_000, max_silence=0, vad_model_path=None)
    record_path = tmp_path / "command.wav"
    monkeypatch.setattr(recorder, "_write_audio", lambda audio: record_path)

    assert recorder.process_chunk(np.full(160, 0.5, dtype=np.float32)) is None
    result = recorder.process_chunk(np.zeros(160, dtype=np.float32))

    assert result == record_path
    assert recorder.is_recording is False


def test_vad_accepts_realistic_microphone_voice_energy():
    recorder = VADRecorder(vad_model_path=None)

    assert recorder.process_chunk(np.full(160, 0.05, dtype=np.float32)) is None
    assert recorder.is_recording is True


def test_vad_accepts_low_amplitude_speech_when_the_model_is_stubbed(
    tmp_path, monkeypatch
):
    """Quiet-but-sustained speech passes via Silero even below the energy floor."""

    class StubSilero:
        available = True

        def probabilities(self, chunk):
            # The model hears speech in the quiet chunk the amplitude floor
            # (voice_threshold 0.025) would miss, and silence in a true pause.
            return [0.9] if float(np.max(np.abs(chunk))) > 0 else [0.0]

        def reset(self):
            pass

    recorder = VADRecorder(max_silence=0, vad_model_path=None)
    recorder.vad = StubSilero()
    record_path = tmp_path / "quiet.wav"
    monkeypatch.setattr(recorder, "_write_audio", lambda audio: record_path)

    # Amplitude far below the energy fallback's voice_threshold (0.025).
    assert recorder.process_chunk(np.full(160, 0.005, dtype=np.float32)) is None
    result = recorder.process_chunk(np.zeros(160, dtype=np.float32))

    assert result == record_path
    assert recorder.is_recording is False


def test_vad_writes_a_unique_recording_name_per_capture(tmp_path):
    from scipy.io import wavfile

    recorder = VADRecorder(
        max_silence=0, vad_model_path=None, command_path=tmp_path / "command.wav"
    )

    recorder.process_chunk(np.full(160, 0.5, dtype=np.float32))
    first = recorder.process_chunk(np.zeros(160, dtype=np.float32))
    recorder.process_chunk(np.full(160, 0.5, dtype=np.float32))
    second = recorder.process_chunk(np.zeros(160, dtype=np.float32))

    assert first != second
    assert first.parent == tmp_path
    assert first.stem.startswith("command_")
    for path in (first, second):
        sample_rate, _audio = wavfile.read(path)
        assert sample_rate == 16_000


def test_vad_recorder_reports_the_active_backend():
    assert VADRecorder(vad_model_path=None).backend == "energy"
    assert VADRecorder().backend in {"silero", "energy"}


def test_vad_discards_a_capture_that_never_held_speech(tmp_path, monkeypatch):
    recorder = VADRecorder(max_silence=0, min_speech=2, vad_model_path=None)
    monkeypatch.setattr(recorder, "_write_audio", lambda audio: tmp_path / "never.wav")

    # One loud frame is a cough, not a command.
    recorder.process_chunk(np.full(160, 0.5, dtype=np.float32))
    assert recorder.process_chunk(np.zeros(160, dtype=np.float32)) is None


def test_silero_vad_scores_each_window_and_resets():
    vad = SileroVAD()
    if not vad.available:
        import pytest

        pytest.skip(f"Silero VAD unavailable: {vad.load_error}")

    probabilities = vad.probabilities(np.zeros(1280, dtype=np.float32))

    assert len(probabilities) == 2  # 1280 samples / 512-sample windows
    assert all(0.0 <= value <= 1.0 for value in probabilities)

    vad.reset()
    assert len(vad.probabilities(np.zeros(512, dtype=np.float32))) == 1


def test_silero_vad_does_not_start_a_recording_on_silence():
    recorder = VADRecorder()
    if recorder.backend != "silero":
        import pytest

        pytest.skip("Silero VAD is not loaded.")

    assert recorder.process_chunk(np.zeros(1280, dtype=np.float32)) is None
    assert recorder.is_recording is False
