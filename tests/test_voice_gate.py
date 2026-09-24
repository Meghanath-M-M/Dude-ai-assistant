"""Voice gate: only the enrolled speaker may wake the assistant.

Text cannot separate a real "hey dude" from a Whisper prompt echo on noise —
both transcribe to the same phrase. A speaker embedding can: noise has no
speaker, and other people or media are not the enrolled voice.

The gate's contract is degradation: no voiceprint, no backend, or unreadable
audio all return ``None`` (open) and the wake proceeds on the phrase alone.
Only a confident mismatch (``False``) suppresses a wake — and every verdict
carries its score in the console line, so thresholds are tuned from data
(field round 10: the enrolled voice was being vetoed at 0.5).
"""

import numpy as np
import pytest
from scipy.io import wavfile

from nova_agent.config.settings import Settings
from nova_agent.core.voice_gate import VoiceGate, carve_windows, trim_silence
from nova_agent.main import NovaAgent

# Orthonormal unit vectors: cosines between any two of these are 0.0.
USER = np.array([1.0, 0.0, 0.0], dtype=np.float32)
OTHER = np.array([0.0, 1.0, 0.0], dtype=np.float32)
THIRD = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _voiceprint(path, vectors=(USER,)):
    np.save(path, np.stack([np.asarray(v, dtype=np.float32) for v in vectors]))
    return path


def _gate(tmp_path, embedder=lambda _s: USER, *, vectors=(USER,), enabled=True):
    settings = Settings(voice_gate=enabled)
    return VoiceGate(
        settings,
        embedder=embedder,
        voiceprint_path=tmp_path / "voiceprint.npy",
    )


def _loud_wav(path, seconds=1.0):
    audio = (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(16000 * seconds)) / 16000))
    wavfile.write(path, 16000, (audio * 32767).astype(np.int16))
    return path


# --- VoiceGate unit tests ---------------------------------------------------


def test_gate_is_open_without_a_voiceprint(tmp_path):
    gate = _gate(tmp_path)

    assert gate.engaged is False
    assert gate.verify_samples(np.ones(16000, dtype=np.float32)) is None
    assert "--enroll-voice" in gate.status()


def test_gate_is_open_when_disabled(tmp_path):
    _voiceprint(_gate(tmp_path).voiceprint_path)
    gate = _gate(tmp_path, enabled=False)

    assert gate.engaged is False
    assert gate.verify_samples(np.ones(16000, dtype=np.float32)) is None
    assert "disabled" in gate.status()


def test_engaged_gate_accepts_the_enrolled_voice(tmp_path):
    _voiceprint(_gate(tmp_path).voiceprint_path)
    gate = _gate(tmp_path)

    assert gate.engaged is True
    assert gate.verify_samples(np.ones(16000, dtype=np.float32)) is True
    assert gate.last_score == pytest.approx(1.0, abs=1e-6)
    assert "threshold" in gate.status()


def test_engaged_gate_rejects_another_voice(tmp_path):
    _voiceprint(_gate(tmp_path).voiceprint_path)
    gate = _gate(tmp_path, embedder=lambda _s: OTHER)

    assert gate.verify_samples(np.ones(16000, dtype=np.float32)) is False
    assert gate.last_score == pytest.approx(0.0, abs=1e-6)


def test_verification_takes_the_nearest_reference(tmp_path):
    """A probe matching *any* enrolled window passes; one orthogonal to all
    fails — nearest-reference scoring, not a single centroid."""
    _voiceprint(_gate(tmp_path).voiceprint_path, vectors=(USER, OTHER))
    aligned = _gate(tmp_path, embedder=lambda _s: (USER + THIRD) / np.sqrt(2))
    orthogonal = _gate(tmp_path, embedder=lambda _s: THIRD)

    assert aligned.verify_samples(np.ones(16000, dtype=np.float32)) is True
    assert orthogonal.verify_samples(np.ones(16000, dtype=np.float32)) is False
    assert orthogonal.last_score == pytest.approx(0.0, abs=1e-6)


def test_an_unreadable_voiceprint_degrades_open(tmp_path):
    path = tmp_path / "voiceprint.npy"
    path.write_bytes(b"this is not a numpy file")

    with pytest.warns(RuntimeWarning, match="voiceprint unreadable"):
        gate = _gate(tmp_path)

    assert gate.engaged is False
    assert gate.verify_samples(np.ones(16000, dtype=np.float32)) is None


def test_an_embedding_failure_degrades_open(tmp_path):
    _voiceprint(_gate(tmp_path).voiceprint_path)

    def boom(_samples):
        raise RuntimeError("model exploded")

    with pytest.warns(RuntimeWarning, match="embedding failed"):
        verdict = _gate(tmp_path, embedder=boom).verify_samples(
            np.ones(16000, dtype=np.float32)
        )

    assert verdict is None


def test_a_failed_embed_leaves_no_stale_score(tmp_path):
    _voiceprint(_gate(tmp_path).voiceprint_path)
    gate = _gate(tmp_path)  # scores a clean match first
    assert gate.verify_samples(np.ones(16000, dtype=np.float32)) is True
    assert gate.last_score is not None

    def boom(_samples):
        raise RuntimeError("model exploded")

    gate._embedder = boom  # simulate backend death mid-session
    with pytest.warns(RuntimeWarning, match="embedding failed"):
        assert gate.verify_samples(np.ones(16000, dtype=np.float32)) is None

    assert gate.last_score is None


def test_audio_too_short_to_hold_a_phrase_is_not_judged(tmp_path):
    _voiceprint(_gate(tmp_path).voiceprint_path)
    gate = _gate(tmp_path)

    assert gate.verify_samples(np.ones(100, dtype=np.float32)) is None
    assert gate.last_score is None


def test_verify_file_reads_the_recorded_segment(tmp_path):
    _voiceprint(_gate(tmp_path).voiceprint_path)
    loud = _loud_wav(tmp_path / "seg.wav")
    # The embedder keys off what was actually read back from disk (a sine's
    # mean is ~0 — use loudness, which only the real segment has).
    gate = _gate(tmp_path, embedder=lambda s: USER if np.abs(s).mean() > 0.1 else OTHER)

    assert gate.verify_file(loud) is True
    assert gate.last_score == pytest.approx(1.0, abs=1e-6)


def test_verify_file_judges_the_voice_not_the_silence_padding(tmp_path):
    """A probe that is half VAD trailing-quiet must still score as the voice
    (field round 10: silence padding dragged real wakes under the bar)."""
    _voiceprint(_gate(tmp_path).voiceprint_path)
    voice = 0.3 * np.sin(2 * np.pi * 220 * np.arange(8000) / 16000)
    padded = np.concatenate([voice, np.zeros(48000)])  # 0.5s voice + 3s quiet
    path = tmp_path / "padded.wav"
    wavfile.write(path, 16000, (padded * 32767).astype(np.int16))
    gate = _gate(tmp_path, embedder=lambda s: USER if np.abs(s).mean() > 0.1 else OTHER)

    assert gate.verify_file(path) is True  # trimmed to the voiced part first


def test_verify_file_rejects_a_phrase_match_in_too_little_audio(tmp_path):
    """A full "hey dude" needs ~0.6s — sub-0.25s audio is a prompt echo."""
    _voiceprint(_gate(tmp_path).voiceprint_path)
    tiny = tmp_path / "tiny.wav"
    wavfile.write(tiny, 16000, np.full(1000, 0.5, dtype=np.int16))

    assert _gate(tmp_path).verify_file(tiny) is False


def test_verify_file_is_open_without_a_voiceprint(tmp_path):
    loud = _loud_wav(tmp_path / "seg.wav")

    assert _gate(tmp_path).verify_file(loud) is None


# --- trim_silence / carve_windows (enrollment helpers) ----------------------


def test_trim_silence_drops_the_quiet_edges_keeps_the_voice():
    samples = np.concatenate(
        [np.zeros(8000), np.full(8000, 0.5, dtype=np.float32), np.zeros(8000)]
    )

    trimmed = trim_silence(samples, margin_samples=1600)

    assert trimmed.size == 8000 + 2 * 1600
    assert trimmed[0] == 0.0  # margin kept
    assert trimmed[1600] == pytest.approx(0.5)


def test_trim_silence_passes_through_all_quiet_or_empty_audio():
    assert trim_silence(np.zeros(1000, dtype=np.float32)).size == 1000
    assert trim_silence(np.empty(0, dtype=np.float32)).size == 0


def test_carve_windows_splits_a_clip_into_probe_length_pieces():
    clip = np.arange(16000 * 5, dtype=np.float32)  # 5s

    windows = carve_windows(clip, 16000, window_seconds=1.5)

    assert len(windows) == 3  # 4.5s used, the 0.5s tail dropped
    assert all(w.size == 24000 for w in windows)
    assert windows[0][0] == 0.0
    assert windows[1][0] == 24000.0


def test_carve_windows_keeps_a_short_clip_whole():
    short = np.ones(8000, dtype=np.float32)
    assert carve_windows(short, 16000) == [short]
    assert carve_windows(np.empty(0, dtype=np.float32), 16000) == []


# --- agent integration ------------------------------------------------------


class FakeProcessor:
    def __init__(self, transcripts):
        self.stt = object()  # gates transcript wake on in the agent
        self._transcripts = iter(transcripts)
        self.transcribed = []
        self.pending_confirmation = None
        self.pending_text = ""
        self.timings = {}

    def transcribe(self, audio_path, prompt=None, hotwords=None, command=False):
        self.transcribed.append(audio_path)
        return next(self._transcripts)


class FakeHUD:
    @staticmethod
    def set_state(_state, _message=""):
        pass

    @staticmethod
    def show_transcript(_text):
        pass


class NeverWake:
    def __init__(self):
        self.resets = 0

    def process_chunk(self, _chunk):
        return False

    def reset(self):
        self.resets += 1


class AlwaysWake(NeverWake):
    def process_chunk(self, _chunk):
        return True


class IdleVAD:
    def process_chunk(self, _chunk):
        return None

    def reset(self):
        pass


class FakeGate:
    """VoiceGate with a scripted verdict, recording what was judged."""

    engaged = True
    threshold = 0.4

    def __init__(self, verdict):
        self.verdict = verdict
        self.checked = []
        # The score the console line will print for a rejection.
        self.last_score = None if verdict is None else 0.21

    @staticmethod
    def status() -> str:
        return "test gate"

    def verify_file(self, path):
        self.checked.append(path)
        return self.verdict

    def verify_samples(self, samples):
        self.checked.append(np.asarray(samples))
        return self.verdict


def _agent(transcripts, **kwargs):
    return NovaAgent(
        wake_engine=NeverWake(),
        vad_reader=IdleVAD(),
        processor=FakeProcessor(transcripts),
        hud=FakeHUD(),
        **kwargs,
    )


def test_transcript_wake_is_ignored_on_a_voice_mismatch(tmp_path, capsys):
    segment = _loud_wav(tmp_path / "seg.wav")
    gate = FakeGate(False)
    agent = _agent(["hey dude"], voice_gate=gate)

    assert agent._wake_from_transcript(segment) is None
    out = capsys.readouterr().out
    assert "[wake] ignored (voice mismatch 0.21 < 0.4)" in out
    assert "'hey dude'" in out  # the rejected text stays auditable
    assert agent.listening is False
    assert not segment.exists()  # probes are cleaned up either way
    assert gate.checked == [segment]


def test_transcript_wake_still_works_when_the_gate_is_open(tmp_path, capsys):
    segment = _loud_wav(tmp_path / "seg.wav")
    agent = _agent(["hey dude"], voice_gate=FakeGate(None))

    assert agent._wake_from_transcript(segment) == "wake"
    assert agent.listening is True
    assert "[wake] matched: 'hey dude'" in capsys.readouterr().out


def test_accepted_wake_prints_its_voice_score(tmp_path, capsys):
    segment = _loud_wav(tmp_path / "seg.wav")
    agent = _agent(["hey dude"], voice_gate=FakeGate(True))

    assert agent._wake_from_transcript(segment) == "wake"
    assert "(voice 0.21)" in capsys.readouterr().out


def test_audio_model_wake_is_ignored_on_a_voice_mismatch(capsys):
    wake = AlwaysWake()
    agent = NovaAgent(
        wake_engine=wake,
        vad_reader=IdleVAD(),
        processor=FakeProcessor([]),
        hud=FakeHUD(),
        voice_gate=FakeGate(False),
    )
    resets_before = wake.resets

    assert agent.process_chunk(np.zeros(160, dtype=np.float32)) is None
    out = capsys.readouterr().out
    assert "voice mismatch 0.21 < 0.4, audio model" in out
    assert agent.listening is False
    # Re-armed: the next real phrase can fire, and no embedding spam follows.
    assert wake.resets == resets_before + 1


def test_audio_model_wake_proceeds_when_the_gate_is_open():
    agent = NovaAgent(
        wake_engine=AlwaysWake(),
        vad_reader=IdleVAD(),
        processor=FakeProcessor([]),
        hud=FakeHUD(),
        voice_gate=FakeGate(None),
    )

    assert agent.process_chunk(np.zeros(160, dtype=np.float32)) == "wake"
    assert agent.listening is True


def test_recent_audio_window_collects_and_caps_frames():
    agent = _agent([])
    frame = np.full((160, 1), 0.5, dtype=np.float32)

    for _ in range(3):
        agent._on_audio(frame, 160, None, None)
    assert agent._recent_audio_window().size == 3 * 160

    for _ in range(100):
        agent._on_audio(np.zeros((160, 1), dtype=np.float32), 160, None, None)
    # Capped at ~1.5s of sample_block frames — a rolling window, not a history.
    assert agent._recent_audio_window().size == agent._recent_audio.maxlen * 160


def test_voice_gate_settings_defaults():
    settings = Settings()
    assert settings.voice_gate is True
    # Confident-mismatch line: ECAPA different-speaker scores sit below ~0.3;
    # 0.4 still vetoes them while tolerating short same-voice probes.
    assert settings.voice_threshold == 0.4
