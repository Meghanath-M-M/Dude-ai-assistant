"""Understanding fixes for the failure modes seen in a live dry-run session.

The field log that motivated this file:

    Heard: open.               -> Intent: open_app (0.75) -> "Would open chrome"
    Heard: so, i run the time. -> Intent: none (best score 0.51)
    Heard: name.               -> Intent: none (best score 0.44)

The first is the dangerous one. ``_lexical_fallback``'s example loop matches
whole words both ways, so a bare "open" claimed the example "open chrome" — any
transcript that collapsed to a launch verb launched the browser. Whisper cannot
guess a name it has never seen, and the honest answer to an *incomplete*
sentence is a question ("Open what?"), which is also what a user can answer
without saying the wake word again.

The reference implementations agree: Home Assistant's Whisper server biases
decoding with the names the user actually says, Hassil treats an edit distance
as a cost that must stay inside a budget, and isair/jarvis refuses to act on a
query that is not self-contained.
"""

import time
from pathlib import Path

from nova_agent.config.settings import INTENTS_PATH, Settings
from nova_agent.core.context_engine import ContextEngine
from nova_agent.core.intent_router import IntentRouter
from nova_agent.main import (
    CANCELLED_RESPONSE,
    IDENTITY_RESPONSE,
    MISS_RESPONSE,
    OPEN_WHAT_RESPONSE,
    CommandProcessor,
    NovaAgent,
    build_registry,
)
from nova_agent.tools import app_controller
from nova_agent.tools.screen_reader import TESSERACT_MISSING_MESSAGE
from nova_agent.ui.overlay import NovaHUD

# --- fakes -----------------------------------------------------------------


class FakeSTT:
    def __init__(self, text=""):
        self.text = text
        self.hotwords = []

    def transcribe(self, _audio_path, prompt=None, hotwords=None):
        self.hotwords.append(hotwords)
        return self.text


class ScriptedRouter:
    """A router with an explicit script, plus the runner-up a miss reports."""

    def __init__(self, script=None, default=(None, 0.1)):
        self.script = script or {}
        self.default = default
        self.last_best_label = None

    def match(self, text):
        return self.script.get(text, self.default)


class FakeTTS:
    def __init__(self):
        self.messages = []

    def speak(self, message):
        self.messages.append(message)


class NeverWakeModel:
    def process_chunk(self, _chunk):
        return False

    def reset(self):
        pass


class NullVAD:
    def process_chunk(self, _chunk):
        return None

    def reset(self):
        pass


class FakeHUD:
    def __init__(self):
        self.states = []

    def set_state(self, state, message=""):
        self.states.append(state)

    def show_transcript(self, _text):
        pass

    def start(self):
        return False

    def stop(self):
        pass


def _agent(tmp_path, **kwargs):
    return NovaAgent(
        wake_engine=NeverWakeModel(),
        vad_reader=NullVAD(),
        hud=FakeHUD(),
        context=ContextEngine(tmp_path / "nova.db"),
        **kwargs,
    )


def _fake_start_menu(tmp_path, *shortcut_names):
    programs = tmp_path / "Microsoft/Windows/Start Menu/Programs"
    programs.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in shortcut_names:
        shortcut = programs / name
        shortcut.write_bytes(b"")
        paths.append(shortcut)
    return paths


def _isolated_apps(tmp_path, monkeypatch, *shortcut_names):
    """Point the resolver at a fake Start Menu and an empty PATH."""
    paths = _fake_start_menu(tmp_path, *shortcut_names)
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "not-there"))
    monkeypatch.setattr(app_controller.shutil, "which", lambda *_args, **_kwargs: None)
    return paths


# --- a launch verb with no app asks instead of guessing --------------------


def test_bare_launch_verb_claims_open_app_without_a_target():
    router = IntentRouter(intents_path=INTENTS_PATH)

    for uttered in ("open", "launch", "open.", "Open,"):
        intent, score = router._lexical_fallback(uttered)

        assert intent is not None, uttered
        assert intent["action"] == "open_app"
        # No target at all: nothing was named, so there is nothing to default to.
        assert "target" not in intent, uttered
        assert score == 0.75


def test_a_launch_verb_with_no_app_asks_instead_of_opening_chrome():
    router = ScriptedRouter({"open.": ({"action": "open_app", "safe": True}, 0.75)})
    processor = CommandProcessor(FakeSTT("open."), router, FakeTTS())

    response = processor.process("command.wav")

    assert response == OPEN_WHAT_RESPONSE
    assert processor.pending_clarification == "open_app"
    assert processor.tts.messages == [OPEN_WHAT_RESPONSE]


def test_the_answer_to_open_what_is_read_as_the_missing_target(tmp_path, monkeypatch):
    (shortcut,) = _isolated_apps(tmp_path, monkeypatch, "Gallery.lnk")
    stt = FakeSTT("open.")
    router = ScriptedRouter({"open.": ({"action": "open_app", "safe": True}, 0.75)})
    processor = CommandProcessor(stt, router, FakeTTS())

    assert processor.process("command.wav") == OPEN_WHAT_RESPONSE

    # "gallery" is not a command on its own — it is the answer to the question.
    stt.text = "gallery"
    response = processor.process("command.wav")

    assert response == f"Would open gallery ({shortcut})"
    assert processor.pending_clarification is None


def test_a_new_command_supersedes_the_open_question():
    stt = FakeSTT("open.")
    router = ScriptedRouter({"open.": ({"action": "open_app", "safe": True}, 0.75)})
    processor = CommandProcessor(stt, router, FakeTTS())

    processor.process("command.wav")
    assert processor.pending_clarification == "open_app"

    router.script["what time is it"] = ({"action": "time_check"}, 0.95)
    stt.text = "what time is it"

    assert processor.process("command.wav").startswith("The current time is")
    assert processor.pending_clarification is None


def test_open_what_keeps_the_command_turn_open(tmp_path, capsys):
    agent = _agent(tmp_path)
    agent.listening = True
    agent._turn_started = time.monotonic()

    agent._finish_turn(OPEN_WHAT_RESPONSE)

    assert agent.listening is True
    assert "Still listening for your command" in capsys.readouterr().out


# --- a time word is enough to mean a time question -------------------------


def test_a_lone_time_word_claims_the_time_intent():
    router = IntentRouter(intents_path=INTENTS_PATH)

    intent, score = router._lexical_fallback("so, i run the time.")

    assert intent is not None
    assert intent["action"] == "time_check"
    assert score == 0.75


def test_an_example_shaped_phrase_keeps_its_own_intent_over_the_time_word():
    """\"it's time to open chrome\" is an open command, not a clock check."""
    router = IntentRouter(intents_path=INTENTS_PATH)

    intent, _score = router._lexical_fallback("it's time to open chrome")

    assert intent["action"] == "open_app"


# --- identity and help ------------------------------------------------------


def test_asking_about_the_name_routes_to_identity():
    router = IntentRouter(intents_path=INTENTS_PATH)

    intent, _score = router._lexical_fallback("name")

    assert intent is not None
    assert intent["action"] == "identity"


def test_identity_and_help_answer_from_the_capability_registry():
    router = ScriptedRouter(
        {
            "what is your name": ({"action": "identity"}, 0.9),
            "help": ({"action": "help"}, 0.9),
        }
    )
    processor = CommandProcessor(
        FakeSTT("what is your name"),
        router,
        FakeTTS(),
        capabilities=build_registry(Settings()),
    )

    assert processor.process("command.wav") == IDENTITY_RESPONSE

    processor.stt.text = "help"
    response = processor.process("command.wav")

    assert response.startswith("I can ")
    assert "open apps" in response
    # Destructive actions are off by default: never advertised either.
    assert "lock the workstation" not in response
    assert "close windows" not in response


# --- the miss line names the runner-up -------------------------------------


def test_a_miss_prints_the_intent_it_almost_matched(capsys):
    router = ScriptedRouter({"frobnicate the flux": (None, 0.51)})
    router.last_best_label = "time_check"
    processor = CommandProcessor(FakeSTT("frobnicate the flux"), router, FakeTTS())

    response = processor.process("command.wav")

    assert response == MISS_RESPONSE
    out = capsys.readouterr().out
    assert "Heard: frobnicate the flux" in out
    assert "Intent: none (best score 0.51 for time_check" in out


# --- the command decode gets the names it should expect ---------------------


def test_stt_hotwords_hold_the_names_this_install_understands(tmp_path):
    context = ContextEngine(tmp_path / "nova.db")
    context.set_alias("browser", "edge")
    processor = CommandProcessor(FakeSTT(), ScriptedRouter(), FakeTTS(), context=context)

    hotwords = processor.stt_hotwords()

    assert "chrome" in hotwords
    assert "browser" in hotwords  # the alias name itself
    assert "edge" in hotwords  # taught with "set browser to edge"


def test_hotwords_are_used_for_the_command_slot_only():
    stt = FakeSTT("open chrome")
    processor = CommandProcessor(stt, ScriptedRouter(), FakeTTS())

    processor.process("command.wav")
    # Wake segments and confirmation replies must not be biased toward app
    # names: a "yes" that decodes as "notepad" cancels the confirmation.
    processor.transcribe("wake.wav", prompt="Hey dude.")

    assert stt.hotwords[0] is not None
    assert stt.hotwords[1] is None


# --- shutdown ---------------------------------------------------------------


def test_hud_stop_without_a_running_overlay_is_harmless():
    hud = NovaHUD(enabled=False)

    assert hud.start() is False
    hud.stop()
    hud.stop()  # a second stop must not raise either


# --- the resolver tolerates a misheard name, but still never guesses --------


def test_resolve_app_tolerates_a_misheard_shortcut_name(tmp_path, monkeypatch):
    (shortcut,) = _isolated_apps(tmp_path, monkeypatch, "Notepad.lnk")

    # Whisper-small drops trailing consonants: "notepad" -> "notepadd".
    assert app_controller.resolve_app("notepadd") == shortcut


def test_fuzzy_resolution_matches_a_word_inside_a_longer_name(tmp_path, monkeypatch):
    (shortcut,) = _isolated_apps(tmp_path, monkeypatch, "Microsoft Edge.lnk")

    assert app_controller.resolve_app("edgge") == shortcut


def test_fuzzy_resolution_does_not_stretch_to_unrelated_names(tmp_path, monkeypatch):
    _isolated_apps(tmp_path, monkeypatch, "Notepad.lnk")

    assert app_controller.resolve_app("notes") is None


def test_a_missing_app_is_still_refused_honestly(tmp_path, monkeypatch):
    """Fuzzy matching must not turn a real name into a launched guess."""
    _isolated_apps(tmp_path, monkeypatch, "Notepad.lnk")
    processor = CommandProcessor(
        FakeSTT("open zzz no such app ever"),
        ScriptedRouter({"open zzz no such app ever": ({"action": "open_app"}, 0.75)}),
        FakeTTS(),
    )

    response = processor.process("command.wav")

    assert response == "I don't know how to open zzz no such app ever yet."
    assert processor.pending_clarification is None


def test_open_question_then_answer_flow_through_a_whole_turn(tmp_path, monkeypatch):
    """End to end: "open." -> question -> answer -> the app it named."""
    (shortcut,) = _isolated_apps(tmp_path, monkeypatch, "Gallery.lnk")
    processor = CommandProcessor(
        FakeSTT("open."),
        ScriptedRouter({"open.": ({"action": "open_app", "safe": True}, 0.75)}),
        FakeTTS(),
        context=ContextEngine(tmp_path / "nova.db"),
        settings=Settings(),
    )

    first = processor.run_observation(
        {"text": "open.", "intent": {"action": "open_app", "safe": True}, "score": 0.75}
    )
    processor.stt.text = "gallery"
    second = processor.run_observation(processor.observe("gallery"))

    assert first == OPEN_WHAT_RESPONSE
    assert second == f"Would open gallery ({shortcut})"
    assert processor.last_intent["action"] == "open_app"
    assert Path(shortcut).exists()


# --- the TTS phrasebook (why the tts budget is met) -------------------------


def test_the_phrasebook_holds_the_fixed_replies(tmp_path):
    processor = CommandProcessor(
        FakeSTT(), ScriptedRouter(), FakeTTS(), capabilities=build_registry(Settings())
    )

    replies = processor.canned_replies()

    assert MISS_RESPONSE in replies
    assert OPEN_WHAT_RESPONSE in replies
    assert IDENTITY_RESPONSE in replies
    assert CANCELLED_RESPONSE in replies
    assert any(reply.startswith("I can ") for reply in replies)  # the help list
    assert all(reply.strip() for reply in replies)


def test_the_screen_reading_failure_is_preloaded_verbatim(monkeypatch):
    """That 15-word reply cost 2669ms of synthesis on *every* use, so it is
    preloaded — which only works if the dispatcher's text and the hint in
    tools/screen_reader.py stay character-for-character identical."""
    from nova_agent.tools import screen_reader

    def missing(**_kwargs):
        raise RuntimeError(TESSERACT_MISSING_MESSAGE)

    monkeypatch.setattr(screen_reader, "read_screen", missing)
    processor = CommandProcessor(
        FakeSTT("read the screen"),
        ScriptedRouter({"read the screen": ({"action": "screen_ocr"}, 0.9)}),
        FakeTTS(),
    )

    response = processor.process("command.wav")

    assert response == f"Screen reading is unavailable. {TESSERACT_MISSING_MESSAGE}"
    assert response in processor.canned_replies()
