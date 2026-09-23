"""Brightness control through the monitor's ``root\\WMI`` interface.

The field log ``Heard: increase the brightness`` scored 0.42 for
system_volume — rightly refused, but for the wrong reason:
``parse_volume_command("increase the brightness")`` returns ``VolumeCommand
("up")``, because the word "increase" alone was enough. Both parsers now
guard against each other, intents.json carries the phrase, and the live tests
run against a fake monitor; a test must never move the machine's real
brightness (the single real-set smoke lives outside the suite).
"""

from types import SimpleNamespace

import pytest

from nova_agent.config.settings import INTENTS_PATH
from nova_agent.core.intent_router import IntentRouter
from nova_agent.main import CommandProcessor
from nova_agent.tools.system import (
    VolumeCommand,
    apply_brightness,
    brightness_backend,
    parse_brightness_command,
    parse_volume_command,
)


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


class FakeMonitor:
    """Stands in for the WmiMonitorBrightness* WMI instances."""

    def __init__(self, level=44):
        self.level = level
        self.set_calls: list[dict] = []

    def WmiMonitorBrightness(self):
        return [SimpleNamespace(CurrentBrightness=self.level)]

    def WmiMonitorBrightnessMethods(self):
        return [self]

    def WmiSetBrightness(self, **kwargs):
        # Keywords only: this machine's MOF declares (Brightness, Timeout) —
        # the reverse of the order community recipes pass positionally, and the
        # wrong guess blanks the screen to 1%. A positional call lands here as
        # a TypeError, fails the production setter's guard, and breaks the
        # assertions below instead of the field.
        self.set_calls.append(kwargs)
        self.level = kwargs["Brightness"]


def test_the_field_phrase_parses_as_brightness_and_never_as_volume():
    """The exact log line: up for brightness, and refused for volume."""
    assert parse_brightness_command("increase the brightness") == VolumeCommand("up")
    assert parse_volume_command("increase the brightness") is None
    assert parse_volume_command("dim the screen") is None


def test_levels_directions_and_honest_nones():
    assert parse_brightness_command("set brightness to fifty percent") == VolumeCommand(
        "level", 50
    )
    assert parse_brightness_command("brightness to 30") == VolumeCommand("level", 30)
    assert parse_brightness_command("dim the screen") == VolumeCommand("down")
    assert parse_brightness_command("lower the brightness") == VolumeCommand("down")
    assert parse_brightness_command("brightness up") == VolumeCommand("up")
    # A token with no direction is a clarification, not a guess; a volume
    # phrase can never parse as brightness (the token gate refuses it).
    assert parse_brightness_command("brightness") is None
    assert parse_brightness_command("turn up the volume") is None
    assert parse_brightness_command("what time is it") is None
    assert parse_brightness_command("") is None


def test_intent_routes_the_field_phrase_to_brightness():
    router = IntentRouter(intents_path=INTENTS_PATH)

    intent, score = router.match("increase the brightness")
    assert intent["action"] == "system_control"
    assert intent.get("target") == "brightness"
    assert score >= 0.75

    intent, _score = router.match("dim the screen")
    assert intent.get("target") == "brightness"


def test_paraphrases_below_the_embedding_band_claim_by_word():
    """Field log: both heard at 0.77/0.78 — the right intent, under the 0.82
    embedding band, with no verbatim example inside the phrase, so the example
    loop had nothing to claim. A brightness word now claims outright at 0.75,
    exactly like VOLUME_WORDS does for volume."""
    router = IntentRouter(intents_path=INTENTS_PATH)

    for phrase in ("turn brightness down to 50", "decrease the brightness to 50"):
        intent, score = router.match(phrase)
        assert intent is not None, phrase
        assert intent.get("target") == "brightness", phrase
        assert score >= 0.75, phrase


def test_processor_end_to_end_paraphrase_sets_the_spoken_level():
    """Real router + parse + dispatch: the level wins over the direction."""
    processor = CommandProcessor(
        FakeSTT("decrease the brightness to 50"),
        IntentRouter(intents_path=INTENTS_PATH),
        FakeTTS(),
    )
    assert processor.process("command.wav") == "Would set the brightness to 50 percent"


def test_brightness_backend_reports_a_known_path():
    assert brightness_backend() in {"wmi", "unsupported"}


# --- dry run: describe the intent, touch nothing -----------------------------


def test_dry_run_replies_describe_the_intent():
    assert (
        apply_brightness(VolumeCommand("up"), dry_run=True)
        == "Would turn the brightness up"
    )
    assert (
        apply_brightness(VolumeCommand("down"), dry_run=True)
        == "Would turn the brightness down"
    )
    assert (
        apply_brightness(VolumeCommand("level", 50), dry_run=True)
        == "Would set the brightness to 50 percent"
    )


def test_dry_run_never_probes_the_backend(monkeypatch):
    """Dry run must not touch the machine's real brightness — even to read it."""
    from nova_agent.tools import system

    def explode():
        raise AssertionError("dry run touched the backend")

    monkeypatch.setattr(system, "_brightness_backend", explode)

    apply_brightness(VolumeCommand("up"), dry_run=True)
    apply_brightness(VolumeCommand("level", 50), dry_run=True)


def test_processor_end_to_end_in_dry_run():
    processor = CommandProcessor(
        FakeSTT("increase the brightness"),
        FakeRouter({"action": "system_control", "target": "brightness"}),
        FakeTTS(),
    )
    assert processor.process("command.wav") == "Would turn the brightness up"


def test_processor_asks_when_no_brightness_action_was_recognized():
    processor = CommandProcessor(
        FakeSTT("brightness"),
        FakeRouter({"action": "system_control", "target": "brightness"}),
        FakeTTS(),
    )
    assert (
        processor.process("command.wav")
        == "Do you want the brightness up, down, or to a percent?"
    )


def test_system_control_without_a_target_stays_volume():
    """The target dispatch must not strand intents that predate brightness."""
    processor = CommandProcessor(
        FakeSTT("mute the volume"),
        FakeRouter({"action": "system_control"}),
        FakeTTS(),
    )
    assert processor.process("command.wav") == "Would mute the volume"


# --- live: every call lands on a fake monitor, never the real one ------------


def test_live_set_reads_back_the_applied_level(monkeypatch):
    from nova_agent.tools import system

    monitor = FakeMonitor(level=44)
    monkeypatch.setattr(system, "_brightness_backend", lambda: monitor)

    response = system.apply_brightness(VolumeCommand("level", 50), dry_run=False)

    assert monitor.level == 50
    assert monitor.set_calls == [{"Brightness": 50, "Timeout": 1}]  # keywords!
    assert response == "Brightness set to 50 percent."  # the honest, read-back number


def test_live_nudge_steps_ten_percent_and_clamps(monkeypatch):
    from nova_agent.tools import system

    monitor = FakeMonitor(level=95)
    monkeypatch.setattr(system, "_brightness_backend", lambda: monitor)

    assert (
        system.apply_brightness(VolumeCommand("up"), dry_run=False)
        == "Turned the brightness up"
    )
    assert monitor.level == 100  # clamped at the top, never past it

    assert (
        system.apply_brightness(VolumeCommand("down"), dry_run=False)
        == "Turned the brightness down"
    )
    assert monitor.level == 90  # one 10% step back down


def test_without_a_controllable_display_the_refusal_is_honest(monkeypatch):
    from nova_agent.tools import system

    monkeypatch.setattr(system, "_brightness_backend", lambda: None)

    assert (
        system.apply_brightness(VolumeCommand("up"), dry_run=False)
        == "I can't change the brightness on this display."
    )
    assert (
        system.apply_brightness(VolumeCommand("level", 50), dry_run=False)
        == "I can't change the brightness on this display."
    )


def test_a_display_that_refuses_the_set_says_so(monkeypatch):
    from nova_agent.tools import system

    class Refusing:
        def WmiMonitorBrightnessMethods(self):
            return []

        def WmiMonitorBrightness(self):
            return [SimpleNamespace(CurrentBrightness=44)]

    monkeypatch.setattr(system, "_brightness_backend", lambda: Refusing())

    assert (
        system.apply_brightness(VolumeCommand("level", 50), dry_run=False)
        == "I couldn't change the brightness on this display."
    )


def test_pythoncom_is_available_for_the_live_path():
    """The worker-thread COM guard imports pythoncom — requirements say it is there."""
    pytest.importorskip("pythoncom")
