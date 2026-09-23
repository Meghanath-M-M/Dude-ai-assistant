"""Compound commands: "X and Y" must do both — and long names must arrive.

Field logs:
  * ``increase brightness and open microsoft edge`` — the brightness claim
    took the whole sentence, brightness ran, and the launch was silently
    dropped.
  * "you need to add a long name" — multi-word Start Menu names ("microsoft
    edge", "file explorer") must reach the command decoder, so they get
    biased into the hotwords (single-word stems stay out of that budget).
"""

from nova_agent.config.settings import INTENTS_PATH
from nova_agent.core.intent_router import IntentRouter
from nova_agent.core.query_extractor import split_compound
from nova_agent.main import CommandProcessor, join_replies


class FakeSTT:
    def __init__(self, text):
        self.text = text

    def transcribe(self, _audio_path, prompt=None, hotwords=None):
        return self.text


class FakeRouter:
    def __init__(self, intent):
        self.intent = intent

    def match(self, _text):
        return dict(self.intent), 0.9


class FakeTTS:
    def __init__(self):
        self.messages = []

    def speak(self, message):
        self.messages.append(message)


def _fake_start_menu(tmp_path, *shortcut_names):
    programs = tmp_path / "Microsoft/Windows/Start Menu/Programs"
    programs.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in shortcut_names:
        shortcut = programs / name
        shortcut.write_bytes(b"")
        paths.append(shortcut)
    return paths


def _real_processor(stt_text=""):
    return CommandProcessor(
        FakeSTT(stt_text),
        IntentRouter(intents_path=INTENTS_PATH),
        FakeTTS(),
    )


def test_split_compound_finds_the_command_clauses():
    assert split_compound("increase brightness and open microsoft edge") == [
        "increase brightness",
        "open microsoft edge",
    ]
    assert split_compound("what time is it") == ["what time is it"]
    assert split_compound("search for rust and ownership") == ["search for rust", "ownership"]
    assert split_compound("") == []
    # Conjunction-stuffed phrases are capped, never queued as a storm.
    assert len(split_compound("a and b and c and d and e")) == 3


def test_join_replies_reads_as_one_response():
    assert (
        join_replies("Turned the brightness up", "Would open chrome")
        == "Turned the brightness up. Would open chrome"
    )
    # The final clause keeps its own stop; only the joins get normalized.
    assert join_replies("Brightness set to 50 percent.", "Opening chrome.") == (
        "Brightness set to 50 percent. Opening chrome."
    )
    assert join_replies("", "Opening chrome.") == "Opening chrome."


def test_observe_queues_the_follow_up_clauses_of_a_compound():
    processor = _real_processor()

    head = processor.observe("increase brightness and open microsoft edge")

    assert head["text"] == "increase brightness"
    assert head["intent"]["target"] == "brightness"
    assert [part["text"] for part in processor.pending_compound] == ["open microsoft edge"]
    assert processor.pending_compound[0]["intent"]["action"] == "open_app"


def test_one_matching_clause_falls_back_to_whole_phrase_routing():
    processor = _real_processor()

    head = processor.observe("blah blah and open chrome")

    assert head["text"] == "blah blah and open chrome"  # routed as one command
    assert head["intent"] is not None
    assert processor.pending_compound == []


def test_compound_runs_both_clauses_dry_run(tmp_path, monkeypatch):
    (shortcut,) = _fake_start_menu(tmp_path, "Microsoft Edge.lnk")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "not-there"))
    tts = FakeTTS()
    processor = CommandProcessor(
        FakeSTT("increase brightness and open microsoft edge"),
        IntentRouter(intents_path=INTENTS_PATH),
        tts,
    )

    response = processor.process("command.wav")

    assert response == f"Would turn the brightness up. Would open microsoft edge ({shortcut})"
    # Speech gets the prose: the resolved path stays a console detail.
    assert tts.messages == ["Would turn the brightness up. Would open microsoft edge"]


def test_compound_waits_for_confirmation_then_runs_the_rest(tmp_path, monkeypatch):
    (shortcut,) = _fake_start_menu(tmp_path, "Gallery.lnk")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "not-there"))
    tts = FakeTTS()
    processor = CommandProcessor(FakeSTT(""), FakeRouter({"action": "time_check"}), tts)
    risky = {
        "text": "lock the workstation",
        "intent": {"action": "lock_workstation", "safe": False},
        "score": 0.9,
    }
    follow_up = {
        "text": "open gallery",
        "intent": {"action": "open_app", "safe": True},
        "score": 0.9,
    }
    processor.pending_compound = [follow_up]

    prompt = processor.run_observation(risky)

    assert "Say confirm to proceed" in prompt
    assert processor.pending_confirmation is not None
    assert processor.pending_compound == [follow_up]  # waits its turn

    confirmed = processor.run_observation(
        {"text": processor.pending_text, "intent": processor.pending_confirmation},
        confirmed=True,
    )

    assert confirmed == f"Would lock the workstation. Would open gallery ({shortcut})"
    assert processor.pending_compound == []
    assert len(tts.messages) == 2  # prompt, then the joined confirmed reply


def test_long_shortcut_names_reach_the_command_decode(tmp_path, monkeypatch):
    _fake_start_menu(tmp_path, "Microsoft Edge.lnk", "File Explorer.lnk", "Notepad.lnk")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "not-there"))
    processor = CommandProcessor(FakeSTT(""), FakeRouter({"action": "time_check"}), FakeTTS())

    hotwords = processor.stt_hotwords()

    assert "microsoft edge" in hotwords
    assert "file explorer" in hotwords
    assert "notepad" not in hotwords  # single-word stems stay out of the budget
