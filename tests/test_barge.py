"""Barge-in (Wave 1a): the user can talk over the reply and cut it off.

Three layers, tested bottom-up:

* ``BargeGate`` — the pure echo-adaptive energy gate. A steady playback bleed
  must never trip it; a person talking over the reply (sustained, well above
  that floor) must; a short transient must not; and the baseline must not
  ratchet upward while someone talks (or they could never exceed it).
* ``TTSEngine`` — playback is chunked and interruptible, so ``stop()``/the
  barge gate halt it between ~50ms slices without touching the mic stream.
* ``NovaAgent`` — the wiring: flush+reset on playback start, drain and feed
  while it plays, and call ``tts.stop()`` when the gate trips.

No microphone or audio hardware: ``sounddevice`` is faked for playback and the
gate is fed synthetic frames.
"""

import sys
import types

import numpy as np

from nova_agent.core.barge import BargeGate
from nova_agent.core.context_engine import ContextEngine
from nova_agent.core.tts import TTSEngine

# --- the gate -----------------------------------------------------------------


def _frame(value: float) -> np.ndarray:
    return np.full(320, value, dtype=np.float32)


def test_a_steady_echo_never_trips():
    gate = BargeGate()

    for _ in range(100):  # ~2s of the assistant's own voice bleeding back in
        assert gate.feed(_frame(0.05)) is False

    assert not gate.tripped


def test_a_person_talking_over_the_reply_trips():
    gate = BargeGate()
    gate.feed(_frame(0.05))  # calibrate the floor to the echo

    tripped = False
    for _ in range(20):  # sustained voice, well above the floor
        tripped = gate.feed(_frame(0.5))

    assert tripped
    assert gate.tripped


def test_a_short_transient_does_not_trip():
    gate = BargeGate()
    gate.feed(_frame(0.05))

    for _ in range(5):  # 0.1s loud — under the 0.3s hold
        assert gate.feed(_frame(0.5)) is False
    assert gate.feed(_frame(0.05)) is False  # a quiet frame ends the run

    assert not gate.tripped


def test_the_baseline_does_not_ratchet_up_when_someone_talks():
    gate = BargeGate()
    gate.feed(_frame(0.05))
    start = gate.baseline

    for _ in range(10):
        gate.feed(_frame(0.5))  # loud but under the hold, so not tripped

    # Loud frames must leave the floor alone, or a raised voice would raise the
    # bar against itself and barge-in could never fire.
    assert gate.baseline == start


def test_the_threshold_is_floored_so_near_silence_cannot_arm_it():
    gate = BargeGate(floor=0.05, multiplier=3.0)

    assert gate.threshold == 0.05  # unprimed: the floor, not zero
    gate.feed(_frame(0.01))
    assert gate.threshold == 0.05  # a near-silent prime still floors it


def test_reset_forgets_the_floor_and_the_trip():
    gate = BargeGate()
    gate.feed(_frame(0.05))
    for _ in range(20):
        gate.feed(_frame(0.5))
    assert gate.tripped

    gate.reset()

    assert not gate.tripped
    assert not gate._primed
    assert gate.baseline == 0.0


# --- playback -----------------------------------------------------------------


class _FakeOutputStream:
    def __init__(self, recorder, **_kw):
        self._recorder = recorder

    def start(self):
        pass

    def write(self, data):
        self._recorder["writes"].append(int(np.asarray(data).shape[0]))

    def stop(self):
        pass

    def close(self):
        pass


def _engine_with_cache(monkeypatch, tmp_path, seconds=1.0, samplerate=24_000):
    """A TTSEngine whose cached reply is a known silent array (no synthesis)."""
    engine = TTSEngine(cache_dir=tmp_path)
    audio = np.zeros(int(samplerate * seconds), dtype=np.float32)
    engine._read_wav = lambda _path: (samplerate, audio)  # pretend it's cached

    recorder = {"writes": []}
    fake_sd = types.SimpleNamespace(
        OutputStream=lambda **kw: _FakeOutputStream(recorder, **kw)
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    return engine, recorder


def test_speak_plays_the_whole_reply_when_nobody_interrupts(monkeypatch, tmp_path):
    engine, recorder = _engine_with_cache(monkeypatch, tmp_path)

    engine.speak("hello there")

    # 1s at 24kHz, 50ms chunks -> 20 slices, all played.
    assert len(recorder["writes"]) == 20
    assert not engine._interrupt.is_set()


def test_the_barge_gate_cuts_playback_short(monkeypatch, tmp_path):
    engine, recorder = _engine_with_cache(monkeypatch, tmp_path)
    begins = []
    calls = {"n": 0}

    def check():
        calls["n"] += 1
        return calls["n"] >= 2  # the user talks over it after the first chunk

    engine.barge_begin = lambda: begins.append(1)
    engine.barge_check = check

    engine.speak("hello there")

    assert begins == [1]  # calibrator fired once, before the first slice
    assert len(recorder["writes"]) == 2  # stopped early, not 20
    assert engine._interrupt.is_set()


def test_stop_raises_the_interrupt_flag():
    engine = TTSEngine(cache_dir="unused")

    engine.stop()

    assert engine._interrupt.is_set()


def test_each_fresh_reply_starts_uninterruptible(monkeypatch, tmp_path):
    # A stop raised while we were idle must not swallow the *next* reply.
    engine, recorder = _engine_with_cache(monkeypatch, tmp_path, seconds=0.1)
    engine.stop()

    engine.speak("hi")

    assert recorder["writes"]  # it still played
    assert not engine._interrupt.is_set()


# --- NovaAgent wiring ---------------------------------------------------------


class _FakeTTS:
    def __init__(self):
        self.barge_begin = None
        self.barge_check = None
        self.stopped = 0

    def stop(self):
        self.stopped += 1

    def speak(self, message):
        pass


class _FakeProcessor:
    def __init__(self):
        self.tts = _FakeTTS()


def _agent(tmp_path):
    from nova_agent.main import NovaAgent

    return NovaAgent(
        wake_engine=object(),
        vad_reader=object(),
        processor=_FakeProcessor(),
        hud=object(),
        monitor=object(),
        context=ContextEngine(tmp_path / "nova.db"),
    )


def test_playback_begin_flushes_the_queue_and_resets_the_gate(tmp_path):
    agent = _agent(tmp_path)
    for _ in range(5):  # frames queued while we were still thinking
        agent.audio_queue.put(np.zeros(160, dtype=np.float32))

    agent._barge_begin()

    assert agent.audio_queue.empty()
    assert not agent.barge_gate._primed
    assert not agent.barge_gate.tripped


def test_a_sustained_voice_over_the_reply_stops_playback(tmp_path):
    agent = _agent(tmp_path)
    agent._barge_begin()
    agent.audio_queue.put(_frame(0.05))  # first slice's echo calibrates the floor
    assert agent._barge_check() is False

    for _ in range(20):  # the user talks over the reply
        agent.audio_queue.put(_frame(0.5))

    assert agent._barge_check() is True
    assert agent.processor.tts.stopped == 1
