"""Transcript wake: idle speech is transcribed and matched against the phrase.

The ONNX sound model can only ever fire on the phrase it was trained for, so
"hey dude" is recognized from what was actually *said* instead.
"""

import numpy as np
import pytest

from nova_agent.core.wake_phrase import normalize, wake_phrase_hits
from nova_agent.main import NovaAgent

# --- matcher unit tests ----------------------------------------------------


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize("Hey,  dude!") == "hey dude"
    assert normalize("") == ""
    assert normalize(None) == ""


def test_wake_phrase_hits_the_exact_phrase():
    matched, remainder = wake_phrase_hits("hey dude", "hey dude")

    assert matched is True
    assert remainder == ""


def test_wake_phrase_hits_are_case_and_punctuation_insensitive():
    matched, remainder = wake_phrase_hits("Hey Dude, what time is it?", "hey dude")

    assert matched is True
    assert remainder == "what time is it"


def test_wake_phrase_remainder_holds_the_one_shot_command():
    matched, remainder = wake_phrase_hits("hey dude open chrome", "hey dude")

    assert matched is True
    assert remainder == "open chrome"


def test_wake_phrase_requires_word_boundaries():
    # "hey dudes" contains the phrase as a substring but is not the phrase.
    assert wake_phrase_hits("hey dudes are cool", "hey dude") == (False, "")
    # The first occurrence fails the boundary check; a later one must still hit.
    assert wake_phrase_hits("hey dudes. hey dude.", "hey dude") == (True, "")


def test_wake_phrase_ignores_unrelated_speech():
    assert wake_phrase_hits("what should we have for dinner", "hey dude") == (False, "")
    assert wake_phrase_hits("the dude abides", "hey dude") == (False, "")
    assert wake_phrase_hits("", "hey dude") == (False, "")


# --- agent integration -----------------------------------------------------


class NeverWakeModel:
    """The sound model never fires (it was trained for another phrase)."""

    def __init__(self):
        self.resets = 0

    def process_chunk(self, _chunk):
        return False

    def reset(self):
        self.resets += 1


class ScriptedIdleVAD:
    """Returns an idle segment exactly when told to."""

    def __init__(self, finalize_on_call: int | None = None):
        self.calls = 0
        self.finalize_on_call = finalize_on_call
        self.resets = 0

    def process_chunk(self, _chunk):
        self.calls += 1
        if self.calls == self.finalize_on_call:
            return "idle_segment.wav"
        return None

    def reset(self):
        self.resets += 1


class FakeProcessor:
    def __init__(self, transcripts):
        self.stt = object()  # gates transcript wake on in the agent
        self._transcripts = iter(transcripts)
        self.transcribed = []
        self.prompts = []
        self.observed = []
        self.responses = []
        self.pending_confirmation = None
        self.pending_text = ""
        self.timings = {}

    def transcribe(self, audio_path, prompt=None, hotwords=None, command=False):
        self.transcribed.append(audio_path)
        self.prompts.append(prompt)
        return next(self._transcripts)

    def observe(self, text):
        self.observed.append(text)
        return {"text": text, "intent": {"action": "greet"}, "score": 0.9}

    def run_observation(self, _observation, confirmed=False):
        self.responses.append("Hello there")
        return "Hello there"


class FakeHUD:
    def __init__(self):
        self.states = []

    def set_state(self, state, message=""):
        self.states.append(state)

    def show_transcript(self, text):
        pass


def _agent(transcripts, finalize_on_call=2, **kwargs):
    return NovaAgent(
        wake_engine=NeverWakeModel(),
        vad_reader=ScriptedIdleVAD(finalize_on_call=finalize_on_call),
        processor=FakeProcessor(transcripts),
        hud=FakeHUD(),
        **kwargs,
    )


def _frame():
    return np.zeros(160, dtype=np.float32)


def test_idle_speech_containing_the_wake_phrase_wakes_the_assistant():
    agent = _agent(["hey dude"])
    agent.processor._transcripts = iter(["hey dude"])

    assert agent.process_chunk(_frame()) is None  # segment not finished yet
    assert agent.process_chunk(_frame()) == "wake"
    assert agent.listening is True
    assert agent.vad_reader.resets >= 1


def test_idle_speech_without_the_phrase_keeps_the_assistant_idle():
    agent = _agent(["what's for dinner"])

    assert agent.process_chunk(_frame()) is None
    assert agent.process_chunk(_frame()) is None
    assert agent.listening is False
    assert agent.processor.observed == []


def test_one_shot_wake_phrase_sentence_runs_its_tail_as_the_command():
    agent = _agent(["hey dude, open the project folder"])

    assert agent.process_chunk(_frame()) is None
    assert agent.process_chunk(_frame()) == "wake"
    # The tail after the wake phrase was dispatched without a second capture.
    assert agent.processor.observed == ["open the project folder"]
    assert agent.listening is False
    assert agent.processor.responses == ["Hello there"]


def test_transcript_wake_can_be_disabled():
    agent = _agent(["hey dude"], transcript_wake=False)

    assert agent.process_chunk(_frame()) is None
    assert agent.process_chunk(_frame()) is None
    assert agent.vad_reader.calls == 0  # idle audio never reached the VAD


def test_processor_without_stt_does_not_segment_idle_audio():
    class STTLessProcessor(FakeProcessor):
        def __init__(self):
            super().__init__([])
            del self.stt

    agent = NovaAgent(
        wake_engine=NeverWakeModel(),
        vad_reader=ScriptedIdleVAD(finalize_on_call=1),
        processor=STTLessProcessor(),
        hud=FakeHUD(),
    )

    assert agent.process_chunk(_frame()) is None
    assert agent.vad_reader.calls == 0


def test_an_failing_idle_transcription_does_not_kill_the_loop():
    class ExplodingSTT:
        pass

    class ExplodingProcessor(FakeProcessor):
        def __init__(self):
            super().__init__([])
            self.stt = ExplodingSTT()

        def transcribe(self, audio_path, prompt=None, hotwords=None, command=False):
            raise RuntimeError("whisper exploded")

    agent = NovaAgent(
        wake_engine=NeverWakeModel(),
        vad_reader=ScriptedIdleVAD(finalize_on_call=1),
        processor=ExplodingProcessor(),
        hud=FakeHUD(),
    )

    assert agent.process_chunk(_frame()) is None
    assert agent.listening is False


def test_the_default_wake_word_is_the_dude_phrase():
    from nova_agent.config.settings import Settings

    assert Settings().wake_word == "hey dude"

    agent = NovaAgent(
        wake_engine=NeverWakeModel(),
        vad_reader=ScriptedIdleVAD(),
        processor=FakeProcessor([]),
        hud=FakeHUD(),
    )
    assert agent.wake_word == "hey dude"


def test_wake_phrase_flag_overrides_the_settings_default():
    agent = NovaAgent(
        wake_word="computer",
        wake_engine=NeverWakeModel(),
        vad_reader=ScriptedIdleVAD(),
        processor=FakeProcessor([]),
        hud=FakeHUD(),
    )
    assert agent.wake_word == "computer"


def test_vad_max_seconds_closes_a_segment_that_never_finds_a_pause(tmp_path):
    from nova_agent.core.vad import VADRecorder

    recorder = VADRecorder(
        max_silence=9,
        vad_model_path=None,
        max_seconds=0.2,
        command_path=tmp_path / "command.wav",
    )
    segment = None
    # Continuous speech (never a silent frame): the cap must still close it.
    for _ in range(40):
        result = recorder.process_chunk(np.full(160, 0.5, dtype=np.float32))
        if result is not None:
            segment = result
            break

    assert segment is not None
    assert segment.parent == tmp_path
    assert recorder.is_recording is False
    assert recorder.buffered_seconds == 0.0


@pytest.mark.parametrize("value", [True, False])
def test_transcript_wake_setting_follows_the_flag(value):
    agent = NovaAgent(
        wake_engine=NeverWakeModel(),
        vad_reader=ScriptedIdleVAD(),
        processor=FakeProcessor([]),
        hud=FakeHUD(),
        transcript_wake=value,
    )
    assert agent.transcript_wake is value


def test_wake_prompt_for_primes_the_decoding_prior():
    from nova_agent.main import wake_prompt_for

    assert wake_prompt_for("hey dude") == "Hey dude. Hey dude."
    assert wake_prompt_for("") is None


def test_quiet_idle_segments_never_reach_stt(tmp_path):
    from scipy.io import wavfile

    quiet = tmp_path / "quiet.wav"
    wavfile.write(quiet, 16000, np.zeros(16000, dtype=np.int16))
    agent = _agent(["hey dude"])

    assert agent._wake_from_transcript(quiet) is None
    assert agent.processor.transcribed == []


def test_voiced_idle_segments_are_transcribed_with_the_wake_prompt(tmp_path):
    from scipy.io import wavfile

    loud = tmp_path / "loud.wav"
    audio = (0.1 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)
    wavfile.write(loud, 16000, (audio * 32767).astype(np.int16))
    agent = _agent(["hey dude"])

    assert agent._wake_from_transcript(loud) == "wake"
    assert agent.processor.transcribed == [loud]
    assert agent.processor.prompts == ["Hey dude. Hey dude."]
    # Idle segments are probes: the file is cleaned up after the check.
    assert not loud.exists()


def test_stt_engine_forwards_the_prompt_to_whisper():
    from nova_agent.core.stt import STTEngine

    stt = object.__new__(STTEngine)
    stt.device = "cpu"
    captured = {}

    class _Segment:
        def __init__(self, text):
            self.text = text

    class _Model:
        def transcribe(self, _audio, **kwargs):
            captured.update(kwargs)
            return [_Segment(" hey dude.")], None

    stt.model = _Model()
    heard = STTEngine.transcribe(stt, "clip.wav", prompt="Hey dude. Hey dude.")

    assert captured["initial_prompt"] == "Hey dude. Hey dude."
    assert heard == "hey dude."


def test_consumer_thread_survives_a_failing_command(capsys):
    import threading

    agent = _agent([])
    handled = []

    def flaky(_chunk):
        handled.append(1)
        if len(handled) == 1:
            raise RuntimeError("boom")
        agent._stopping = True

    agent.process_chunk = flaky
    worker = threading.Thread(target=agent._consume_loop, daemon=True)
    worker.start()
    for _ in range(2):
        agent.audio_queue.put(np.zeros(160, dtype=np.float32))
    worker.join(timeout=5.0)

    assert not worker.is_alive()  # exited normally, did not hang
    assert len(handled) == 2  # the first failure did not kill the consumer
    assert "Command failed (RuntimeError): boom" in capsys.readouterr().out


def test_a_miss_after_the_window_returns_to_idle():
    import time

    from nova_agent.main import MISS_RESPONSE

    agent = _agent([""], finalize_on_call=1)
    agent.processor.run_observation = lambda *_args, **_kwargs: MISS_RESPONSE
    agent.listening = True
    agent._turn_started = time.monotonic() - 45  # beyond command_max_turn

    agent.process_chunk(_frame())

    assert agent.listening is False


def test_a_missed_capture_keeps_the_turn(capsys):
    import time

    from nova_agent.main import MISS_RESPONSE

    agent = _agent([""], finalize_on_call=1)
    agent.processor.run_observation = lambda *_args, **_kwargs: MISS_RESPONSE
    agent.listening = True
    agent._turn_started = time.monotonic()

    assert agent.process_chunk(_frame()) == "idle_segment.wav"
    assert agent.listening is True  # the miss did not end the turn
    assert "Still listening for your command" in capsys.readouterr().out


def test_a_miss_survives_slow_pipeline_latency(capsys):
    """Command #1 once took 13.33s (cold encoder) against a wake-time window.

    The window must be measured from the miss, not from the wake, or our own
    latency silently eats the retry before it can run.
    """
    import time

    from nova_agent.main import MISS_RESPONSE

    agent = _agent([""], finalize_on_call=1)
    agent.processor.run_observation = lambda *_args, **_kwargs: MISS_RESPONSE
    agent.listening = True
    agent._turn_started = time.monotonic() - 20  # processing already ate 20s

    agent.process_chunk(_frame())

    assert agent.listening is True
    assert "Still listening for your command" in capsys.readouterr().out


def test_announced_wake_opens_the_command_window():
    import time

    agent = _agent(["hey dude"])

    agent._begin_listening("test")

    assert agent.listening is True
    assert time.monotonic() - agent._turn_started < 5


def test_on_audio_copies_out_of_portaudio_reusable_buffer():
    # indata is PortAudio's ring-buffer memory, overwritten by the next
    # callback; a view would let the VAD finalize segments from silent audio.
    agent = _agent([])
    indata = np.zeros((1280, 1), dtype=np.float32)
    indata[0, 0] = 0.5
    agent._on_audio(indata, 1280, None, None)
    indata[0, 0] = -0.9  # what the next callback writes into the same memory

    queued = agent.audio_queue.get_nowait()
    assert queued.shape == (1280,)
    assert queued[0] == np.float32(0.5)


def test_debug_keeps_unmatched_idle_segments_for_replay(tmp_path):
    from scipy.io import wavfile

    heard = tmp_path / "command_deadbeef.wav"
    audio = (0.1 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)
    wavfile.write(heard, 16000, (audio * 32767).astype(np.int16))
    agent = _agent(["what are you doing"], debug=True)

    assert agent._wake_from_transcript(heard) is None
    # Renamed and kept so the real live audio can be replayed offline.
    assert heard.with_name(f"idle_{heard.name}").exists()
    assert not heard.exists()


def test_wake_announces_itself_even_without_debug(capsys):
    class WakeNow(NeverWakeModel):
        def process_chunk(self, _chunk):
            return True

    agent = NovaAgent(
        wake_engine=WakeNow(),
        vad_reader=ScriptedIdleVAD(),
        processor=FakeProcessor([]),
        hud=FakeHUD(),
    )

    assert agent.process_chunk(_frame()) == "wake"
    output = capsys.readouterr().out
    assert "Wake detected" in output
    assert "listening for your command" in output
