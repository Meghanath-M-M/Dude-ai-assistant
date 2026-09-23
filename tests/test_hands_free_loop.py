import numpy as np

from nova_agent.main import NovaAgent


class FakeWakeWordEngine:
    def process_chunk(self, _chunk):
        return True

    def reset(self):
        pass


class FakeVADRecorder:
    def __init__(self):
        self.calls = 0

    def process_chunk(self, _chunk):
        self.calls += 1
        return "command.wav" if self.calls == 1 else None

    def reset(self):
        pass


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


def test_nova_agent_transitions_from_wake_to_command():
    processor = FakeProcessor()
    agent = NovaAgent(
        wake_engine=FakeWakeWordEngine(),
        vad_reader=FakeVADRecorder(),
        processor=processor,
    )

    first = agent.process_chunk(np.zeros(160, dtype=np.float32))
    second = agent.process_chunk(np.zeros(160, dtype=np.float32))

    assert first == "wake"
    assert second == "command.wav"
    assert agent.listening is False
    assert processor.path_seen == ["command.wav"]
