"""Field round 8: false wakes from noise/echo and their STT cost.

The soak's complaints — waking without the wake phrase, then acting on
garbage — traced to three gaps the guards here close: the stand-in sound
model's 0.5 default, no reverb-tail guard after our own playback, and idle
segments buffering toward 30s (the stt budget overrun came from those).
"""

import time

import numpy as np

from nova_agent.config.settings import Settings
from nova_agent.core.wake_word import WakeWordEngine
from nova_agent.main import NovaAgent


class _Wake:
    def process_chunk(self, _chunk):
        return False

    def reset(self):
        pass


class _VAD:
    """Records the segment cap in force whenever an idle probe arrives."""

    def __init__(self):
        self.caps = []

    def process_chunk(self, _chunk):
        self.caps.append(getattr(self, "max_seconds", None))

    def reset(self):
        pass


class _HUD:
    def set_state(self, state, message=""):
        pass

    def show_transcript(self, text):
        pass


class _Processor:
    stt = object()  # gates transcript wake on

    def __init__(self):
        self.pending_confirmation = None
        self.pending_text = ""
        self.timings = {}


def _agent() -> NovaAgent:
    return NovaAgent(
        wake_engine=_Wake(),
        vad_reader=_VAD(),
        processor=_Processor(),
        hud=_HUD(),
    )


def test_wake_line_names_the_triggering_path(capsys):
    agent = _agent()

    agent._begin_listening("audio model")

    assert "Wake detected (audio model) - listening" in capsys.readouterr().out

    other = _agent()
    other._begin_listening("transcript: 'hey dude open chrome'")

    out = capsys.readouterr().out
    assert "Wake detected (transcript) - listening" in out
    # The full utterance stays --debug detail; the loud line is one line.
    assert "hey dude open chrome" not in out


def test_reverb_tail_is_dropped_at_the_callback():
    agent = _agent()
    frame = np.zeros((160, 1), dtype=np.float32)

    # A fresh agent has never spoken: the tail guard starts open.
    agent._on_audio(frame, 160, None, None)
    assert agent.audio_queue.qsize() == 1

    # Playback just ended: the room's ringing tail must not be queued...
    agent._finish_speaking()
    agent._on_audio(frame, 160, None, None)
    assert agent.is_speaking is False
    assert agent.audio_queue.qsize() == 1

    # ...until the cooldown window has passed.
    agent._speaking_ended_at = time.monotonic() - Settings().wake_cooldown - 0.01
    agent._on_audio(frame, 160, None, None)
    assert agent.audio_queue.qsize() == 2


def test_idle_probes_carry_the_short_cap_and_listening_restores_the_full_one():
    agent = _agent()
    settings = Settings()

    agent.process_chunk(np.zeros(160, dtype=np.float32))  # one idle probe

    assert agent.vad_reader.caps == [settings.idle_segment_max]
    assert settings.idle_segment_max < settings.vad_max_seconds

    agent._begin_listening("test")

    assert agent.vad_reader.max_seconds == settings.vad_max_seconds


def test_round_8_guardrail_defaults():
    settings = Settings()

    assert settings.wake_cooldown == 0.5
    assert settings.idle_segment_max == 10.0
    # The stand-in model fired on background chatter at 0.5 (field round 8).
    assert WakeWordEngine.resolve_threshold(None, using_model=True) == 0.65
