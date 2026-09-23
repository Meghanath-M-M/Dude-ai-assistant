"""Out-of-the-box app opening: resolve spoken names without a configured path.

``APP_PATHS`` only knows chrome and code. "open gallery"/"open notepad" must
still dispatch (lexical claim + phrase fallback) and resolve through PATH or
the Start Menu — or answer honestly, never crash, never launch a guess.
"""

from pathlib import Path

from nova_agent.core.context_engine import ContextEngine
from nova_agent.main import CommandProcessor, speakable
from nova_agent.tools import app_controller


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


def test_resolve_app_finds_a_start_menu_shortcut(tmp_path, monkeypatch):
    (shortcut,) = _fake_start_menu(tmp_path, "Gallery.lnk")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "not-there"))

    assert app_controller.resolve_app("gallery") == shortcut


def test_resolve_app_matches_a_whole_word_inside_a_longer_name(tmp_path, monkeypatch):
    (shortcut,) = _fake_start_menu(tmp_path, "Microsoft Edge.lnk")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "not-there"))

    assert app_controller.resolve_app("edge") == shortcut


def test_resolve_app_returns_none_for_an_unknown_name(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path))

    assert app_controller.resolve_app("zzq_no_such_app_ever") is None


def test_open_app_dispatch_resolves_and_stays_dry_run(tmp_path, monkeypatch):
    (shortcut,) = _fake_start_menu(tmp_path, "Gallery.lnk")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "not-there"))
    tts = FakeTTS()
    processor = CommandProcessor(
        FakeSTT("open gallery"),
        FakeRouter({"action": "open_app", "safe": True}),
        tts,
    )

    response = processor.process("command.wav")

    assert response == f"Would open gallery ({shortcut})"
    # The path is console detail: speech gets the prose form.
    assert tts.messages == [speakable(response)] == ["Would open gallery"]


def test_resolve_app_maps_spoken_names_to_the_real_executable(tmp_path, monkeypatch):
    """"open calculator": no Calculator.lnk and calc.exe != 'calculator' —
    a spoken-name alias is the only way to resolve it."""
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "not-there"))
    seen = []

    def fake_which(command, path=None):
        seen.append(command)
        if command in {"calc", "calc.exe"}:
            return r"C:\Windows\system32\calc.exe"
        return None

    monkeypatch.setattr(app_controller.shutil, "which", fake_which)

    assert app_controller.resolve_app("calculator") == Path(
        r"C:\Windows\system32\calc.exe"
    )
    assert seen == ["calculator", "calculator.exe", "calc"]


def test_open_calculator_resolves_out_of_the_box(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "not-there"))
    monkeypatch.setattr(
        app_controller.shutil,
        "which",
        lambda command, path=None: r"C:\Windows\system32\calc.exe"
        if command in {"calc", "calc.exe"}
        else None,
    )
    tts = FakeTTS()
    processor = CommandProcessor(
        FakeSTT("open calculator"),
        FakeRouter({"action": "open_app", "safe": True}),
        tts,
    )

    response = processor.process("command.wav")

    assert response == r"Would open calculator (C:\Windows\system32\calc.exe)"
    assert tts.messages == ["Would open calculator"]


def test_open_app_dispatch_answers_honestly_when_nothing_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path))
    processor = CommandProcessor(
        FakeSTT("open zzz no such app ever"),
        FakeRouter({"action": "open_app", "safe": True}),
        FakeTTS(),
    )

    response = processor.process("command.wav")

    assert response == "I don't know how to open zzz no such app ever yet."


def test_set_browser_accepts_an_app_the_resolver_can_find(tmp_path, monkeypatch):
    """"set browser to edge" used to be refused (only chrome/code configured),
    even though opening resolves aliases through PATH/Start Menu at use time."""
    _fake_start_menu(tmp_path, "Microsoft Edge.lnk")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "not-there"))
    context = ContextEngine(tmp_path / "nova.db")
    processor = CommandProcessor(
        FakeSTT("set browser to edge"),
        FakeRouter({"action": "set_browser"}),
        FakeTTS(),
        context=context,
    )

    response = processor.process("command.wav")

    assert response == "Browser set to edge."
    assert context.get_alias("browser") == "edge"


def test_set_browser_still_refuses_an_app_nothing_can_resolve(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path))
    processor = CommandProcessor(
        FakeSTT("set browser to zzz no such browser"),
        FakeRouter({"action": "set_browser"}),
        FakeTTS(),
        context=ContextEngine(tmp_path / "nova.db"),
    )

    response = processor.process("command.wav")

    assert response == "I don't know how to open zzz no such browser yet."


# --- launching: CreateProcess cannot execute a shortcut ----------------------


def _explode_popen(_args):
    raise AssertionError("Popen must not be called for a shortcut")


def test_shortcut_launch_goes_through_the_shell(tmp_path, monkeypatch):
    """Live field log: 'open microsoft edge' resolved Microsoft Edge.lnk and
    died in CreateProcess with WinError 193 "%1 is not a valid Win32
    application". Shortcuts must launch via the shell (os.startfile)."""
    shortcut = tmp_path / "Microsoft Edge.lnk"
    shortcut.write_bytes(b"")
    launched = []
    monkeypatch.setattr(app_controller.os, "startfile", launched.append)
    monkeypatch.setattr(app_controller.subprocess, "Popen", _explode_popen)

    response = app_controller.open_path("microsoft edge", shortcut, dry_run=False)

    assert response == "Opening microsoft edge"
    assert launched == [shortcut]


def test_directory_launch_goes_through_the_shell(tmp_path, monkeypatch):
    folder = tmp_path / "SomeProject"
    folder.mkdir()
    launched = []
    monkeypatch.setattr(app_controller.os, "startfile", launched.append)
    monkeypatch.setattr(app_controller.subprocess, "Popen", _explode_popen)

    app_controller.open_path("SomeProject", folder, dry_run=False)

    assert launched == [folder]


def test_executable_launch_keeps_createprocess(tmp_path, monkeypatch):
    """The field-proven path for notepad & friends must not change shape."""
    exe = tmp_path / "tool.exe"
    exe.write_bytes(b"")
    launched = []
    monkeypatch.setattr(app_controller.subprocess, "Popen", launched.append)

    def refuse_shell(_path):
        raise AssertionError("os.startfile must not be needed for a plain exe")

    monkeypatch.setattr(app_controller.os, "startfile", refuse_shell)

    assert app_controller.open_path("tool", exe, dry_run=False) == "Opening tool"
    assert launched == [[str(exe)]]


def test_createprocess_refusal_falls_back_to_the_shell(tmp_path, monkeypatch):
    """An unusual file (WinError 193 from CreateProcess) still launches."""
    odd = tmp_path / "thing.url"
    odd.write_bytes(b"")
    launched = []

    def refuse_popen(_args):
        raise OSError(193, "%1 is not a valid Win32 application")

    monkeypatch.setattr(app_controller.subprocess, "Popen", refuse_popen)
    monkeypatch.setattr(app_controller.os, "startfile", launched.append)

    assert app_controller.open_path("thing", odd, dry_run=False) == "Opening thing"
    assert launched == [odd]


def test_total_launch_failure_answers_honestly(monkeypatch, tmp_path):
    """No traceback into the listen loop: the spoken reply admits failure."""
    exe = tmp_path / "tool.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(
        app_controller.subprocess,
        "Popen",
        lambda _args: (_ for _ in ()).throw(OSError(193, "nope")),
    )
    monkeypatch.setattr(
        app_controller.os,
        "startfile",
        lambda _path: (_ for _ in ()).throw(OSError(2, "nope")),
    )

    assert (
        app_controller.open_path("tool", exe, dry_run=False) == "I couldn't open tool."
    )
