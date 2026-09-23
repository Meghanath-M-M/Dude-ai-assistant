"""The confirmation window is multi-turn: a reply can be a new command.

Nova asks "Say confirm to proceed" for destructive actions. The user may
answer yes, answer no — or ignore the question and say something else
entirely. That third reply must be treated as a *new* command that replaces
the pending one, not thrown away.
"""

from pathlib import Path

from nova_agent.core.context_engine import ContextEngine
from nova_agent.main import MAX_COMMAND_SWAPS, CommandProcessor, NovaAgent, speakable

CLOSE = {"action": "close_window", "safe": False}
LOCK = {"action": "lock_workstation", "safe": False}
OPEN = {"action": "open_app", "target": "chrome", "safe": True}


class FakeSTT:
    def __init__(self, text):
        self.text = text

    def transcribe(self, _audio_path, prompt=None, hotwords=None):
        return self.text


class ScriptedRouter:
    def __init__(self, script):
        self.script = script

    def match(self, text):
        return self.script.get(text, (None, 0.1))


class FakeTTS:
    def __init__(self):
        self.messages = []

    def speak(self, message):
        self.messages.append(message)


class FakeHUD:
    def __init__(self):
        self.transcripts = []

    def set_state(self, _state, _message=""):
        pass

    def show_transcript(self, text):
        self.transcripts.append(text)


class FakeWake:
    def process_chunk(self, _chunk):
        return True

    def reset(self):
        pass


class FakeVAD:
    def process_chunk(self, _chunk):
        return "command.wav"

    def reset(self):
        pass


_SCRIPT = {
    "close the window": (CLOSE, 0.99),
    "lock the computer": (LOCK, 0.99),
    "open chrome": (OPEN, 0.95),
}


def _agent(tmp_path, first_text, replies):
    processor = CommandProcessor(
        FakeSTT(first_text),
        ScriptedRouter(_SCRIPT),
        FakeTTS(),
        dry_run=True,
        context=ContextEngine(Path(tmp_path) / "nova.db"),
    )
    agent = NovaAgent(
        wake_engine=FakeWake(),
        vad_reader=FakeVAD(),
        processor=processor,
        hud=FakeHUD(),
    )
    scripted = iter(replies)
    agent._await_confirmation = lambda: next(scripted, None)
    return agent, processor


def test_reply_that_is_a_new_command_replaces_the_pending_one(tmp_path):
    agent, processor = _agent(tmp_path, "close the window", replies=["open chrome"])

    response = agent.handle_command("command.wav")

    # The destructive intent never ran; the replacement did.
    assert processor.pending_confirmation is None
    assert "chrome" in response.lower()
    # The confirmation prompt was spoken first, the replacement's reply last.
    # Speech gets the path-stripped form; the full response keeps the path.
    assert processor.tts.messages[0].startswith("About to close")
    assert processor.tts.messages[-1] == speakable(response) == "Would open chrome"


def test_a_clear_no_cancels_the_pending_command(tmp_path):
    agent, processor = _agent(tmp_path, "close the window", replies=["no"])

    response = agent.handle_command("command.wav")

    assert response == "Cancelled"
    assert processor.pending_confirmation is None
    assert not any("chrome" in message for message in processor.tts.messages)


def test_yes_still_runs_the_pending_command(tmp_path):
    agent, processor = _agent(tmp_path, "close the window", replies=["yes"])

    response = agent.handle_command("command.wav")

    assert processor.pending_confirmation is None
    assert response != "Cancelled"
    assert "close" in response.lower()
    assert processor.tts.messages[-1] == speakable(response)


def test_an_unreadable_reply_still_cancels(tmp_path):
    # "hmm" is neither yes/no nor a confidently routable command.
    agent, processor = _agent(tmp_path, "close the window", replies=["hmm"])

    response = agent.handle_command("command.wav")

    assert response == "Cancelled"
    assert processor.pending_confirmation is None


def test_silence_times_out_and_cancels(tmp_path):
    agent, processor = _agent(tmp_path, "close the window", replies=[])

    response = agent.handle_command("command.wav")

    assert response == "Cancelled"
    assert processor.pending_confirmation is None


def test_swaps_are_capped_so_prompts_cannot_ping_pong(tmp_path):
    agent, processor = _agent(
        tmp_path, "close the window", replies=["lock the computer"] * 5
    )

    response = agent.handle_command("command.wav")

    assert response == "Cancelled"
    prompts = [m for m in processor.tts.messages if m.startswith("About to")]
    assert len(prompts) == MAX_COMMAND_SWAPS + 1  # initial + capped swaps


def test_missed_commands_report_what_happened(capsys):
    processor = CommandProcessor(FakeSTT(""), ScriptedRouter(_SCRIPT), FakeTTS())

    # Empty transcription: nothing to route, and the console must say so —
    # a silent miss looks exactly like a dead loop (and a cached reply
    # emits no synthesis warnings either).
    processor.run_observation(processor.observe(""))
    out = capsys.readouterr().out
    assert "Heard: (nothing intelligible)" in out
    assert "Response: I didn't catch that" in out

    # Confidently un-routable text: show what was heard and why it missed.
    processor.run_observation(processor.observe("frobnicate the flux"))
    out = capsys.readouterr().out
    assert "Heard: frobnicate the flux" in out
    assert "Intent: none" in out
    assert "Response: I didn't catch that" in out
