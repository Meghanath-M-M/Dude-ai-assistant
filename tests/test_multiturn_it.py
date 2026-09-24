"""Multi-turn "it": a follow-up pronoun reopens what we just opened.

Wave 1b of the Jarvis roadmap. Previously "open it" always fell through to the
"Open what?" clarification (a pronoun names no app). Now it resolves against
``last_open`` — the last app *or* project actually launched — so a follow-up
never repeats the name.

``last_open`` is deliberately not ``last_text``: the pronoun command itself
would overwrite ``last_text`` with "open it" and lose the target after one use.
"""

from pathlib import Path

from nova_agent.core.context_engine import ContextEngine
from nova_agent.main import OPEN_WHAT_RESPONSE, CommandProcessor


class FakeSTT:
    def __init__(self, text):
        self.text = text

    def transcribe(self, _audio_path, prompt=None, hotwords=None):
        return self.text


class FakeRouter:
    def __init__(self, intent=None):
        self.intent = intent or {"action": "open_app", "safe": True}

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


def _processor(tmp_path):
    return CommandProcessor(
        FakeSTT(""),
        FakeRouter(),
        FakeTTS(),
        context=ContextEngine(tmp_path / "nova.db"),
    )


def _dispatch(processor, text, action="open_app", **intent_extra):
    intent = {"action": action, "safe": True, **intent_extra}
    return processor.run_observation(
        {"text": text, "intent": intent, "score": 0.9}
    )


def test_open_it_reopens_the_last_opened_app(tmp_path, monkeypatch):
    (shortcut,) = _fake_start_menu(tmp_path, "Gallery.lnk")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "not-there"))
    processor = _processor(tmp_path)

    first = _dispatch(processor, "open gallery")
    second = _dispatch(processor, "open it")

    assert first == f"Would open gallery ({shortcut})"
    assert second == first  # "it" resolved to the app we just opened


def test_open_it_is_reusable(tmp_path, monkeypatch):
    # last_open (not last_text) is the referent, so the pronoun survives being
    # used twice: last_text becomes "open it" after the first follow-up.
    (shortcut,) = _fake_start_menu(tmp_path, "Gallery.lnk")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "not-there"))
    processor = _processor(tmp_path)
    expected = f"Would open gallery ({shortcut})"

    _dispatch(processor, "open gallery")
    second = _dispatch(processor, "open it")
    third = _dispatch(processor, "open it")

    assert second == expected
    assert third == expected


def test_open_it_without_history_still_asks(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path))
    processor = _processor(tmp_path)

    response = _dispatch(processor, "open it")

    assert response == OPEN_WHAT_RESPONSE
    assert processor.pending_clarification == "open_app"


def test_open_it_points_at_the_most_recent_open(tmp_path, monkeypatch):
    (gallery, edge) = _fake_start_menu(tmp_path, "Gallery.lnk", "Microsoft Edge.lnk")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "not-there"))
    processor = _processor(tmp_path)

    _dispatch(processor, "open gallery")
    _dispatch(processor, "open edge")
    follow_up = _dispatch(processor, "open it")

    assert follow_up == f"Would open edge ({edge})"
    assert Path(edge).exists()
    assert Path(gallery).exists()


def test_open_it_reopens_the_last_project(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "not-there"))
    project_dir = tmp_path / "ml"
    project_dir.mkdir()
    processor = _processor(tmp_path)
    processor.context.set_preference("default_project", str(project_dir))

    # The first command opens a project; the follow-up arrives as open_app
    # (as the router would classify "open it") and resolves back to the
    # project via the recorded referent.
    first = _dispatch(processor, "open my project", action="context_open")
    second = _dispatch(processor, "open it")

    assert first == f"Would open project default at {project_dir}"
    assert second == first


def test_referent_survives_an_unrelated_command(tmp_path, monkeypatch):
    (shortcut,) = _fake_start_menu(tmp_path, "Gallery.lnk")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "not-there"))
    processor = _processor(tmp_path)
    expected = f"Would open gallery ({shortcut})"

    _dispatch(processor, "open gallery")
    # An unrelated command overwrites last_intent/last_text but must not
    # disturb the open referent.
    _dispatch(processor, "mute the volume", action="system_control", target="volume")
    assert processor.last_intent["action"] == "system_control"

    follow_up = _dispatch(processor, "open it")

    assert follow_up == expected
