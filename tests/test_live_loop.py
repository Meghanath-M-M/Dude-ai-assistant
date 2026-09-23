"""The microphone callback must never block on transcription or playback."""

import numpy as np

from nova_agent.main import NovaAgent


class FakeWakeEngine:
    def __init__(self):
        self.resets = 0

    def process_chunk(self, _chunk):
        return True

    def reset(self):
        self.resets += 1


class FakeVADRecorder:
    def __init__(self):
        self.calls = 0
        self.resets = 0

    def process_chunk(self, _chunk):
        self.calls += 1
        return "command.wav" if self.calls == 1 else None

    def reset(self):
        self.resets += 1


class FakeProcessor:
    def __init__(self):
        self.path_seen = []
        self.pending_confirmation = None
        self.pending_text = ""
        self.timings = {}

    def transcribe(self, audio_path, prompt=None, hotwords=None, command=False):
        self.path_seen.append(audio_path)
        return "open chrome"

    def observe(self, text):
        return {"text": text, "intent": {"action": "open_app", "target": "chrome"}, "score": 0.95}

    def run_observation(self, _observation, confirmed=False):
        return "Would open chrome"


class FakeHUD:
    def __init__(self):
        self.states = []
        self.transcripts = []

    def set_state(self, state, message=""):
        self.states.append(state)

    def show_transcript(self, text):
        self.transcripts.append(text)


def _frame(value: float = 0.0):
    return np.full((160, 1), value, dtype=np.float32)


def _build_agent(processor=None, queue_size: int = 64):
    return NovaAgent(
        wake_engine=FakeWakeEngine(),
        vad_reader=FakeVADRecorder(),
        processor=processor or FakeProcessor(),
        hud=FakeHUD(),
        queue_size=queue_size,
    )


def test_audio_callback_only_queues_frames():
    agent = _build_agent()

    agent._on_audio(_frame(), 160, None, None)

    assert agent.audio_queue.qsize() == 1
    assert agent.processor.path_seen == []


def test_worker_thread_drains_the_queue_and_runs_the_command():
    agent = _build_agent()
    agent._on_audio(_frame(), 160, None, None)  # wake word
    agent._on_audio(_frame(0.4), 160, None, None)  # command audio

    assert agent.process_pending() == 2
    assert agent.processor.path_seen == ["command.wav"]


def test_frames_arriving_while_speaking_are_dropped():
    agent = _build_agent()
    agent.is_speaking = True
    agent._on_audio(_frame(0.9), 160, None, None)

    assert agent.process_pending() == 0
    assert agent.processor.path_seen == []
    assert agent.audio_queue.qsize() == 0


def test_audio_queue_drops_frames_instead_of_blocking():
    agent = _build_agent(queue_size=1)
    agent._on_audio(_frame(), 160, None, None)
    agent._on_audio(_frame(), 160, None, None)

    assert agent.audio_queue.qsize() == 1


def test_nova_never_transcribes_its_own_playback():
    class SelfListeningProcessor:
        def __init__(self):
            self.agent = None
            self.seen = []
            self.pending_confirmation = None
            self.pending_text = ""
            self.timings = {}

        def transcribe(self, audio_path, prompt=None, hotwords=None, command=False):
            self.seen.append(audio_path)
            return "open chrome"

        def observe(self, text):
            return {
                "text": text,
                "intent": {"action": "open_app", "target": "chrome"},
                "score": 0.9,
            }

        def run_observation(self, _observation, confirmed=False):
            # Simulate Kokoro output being picked up by the open microphone.
            self.agent._on_audio(_frame(0.6), 160, None, None)
            assert self.agent.is_speaking is True
            return "Would open chrome"

    processor = SelfListeningProcessor()
    agent = _build_agent(processor=processor)
    processor.agent = agent

    agent._on_audio(_frame(), 160, None, None)  # wake word
    agent._on_audio(_frame(0.4), 160, None, None)  # command audio
    agent.process_pending()

    assert processor.seen == ["command.wav"]
    assert agent.is_speaking is False
    assert agent.audio_queue.qsize() == 0
