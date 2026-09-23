"""Volume control through Core Audio (pycaw), with the media-key fallback.

The old implementation only nudged multimedia keys: "unmute" *toggled* a mute
that was not on, and "set volume to thirty percent" did nothing but turn the
sound up — an example intents.json had carried since Phase 1. The live tests
here run against a fake endpoint; a test must never move the machine's real
volume.
"""

from nova_agent.main import CommandProcessor
from nova_agent.tools.system import (
    VolumeCommand,
    apply_volume,
    parse_volume_command,
    volume_backend,
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


class FakeEndpoint:
    """Stands in for IAudioEndpointVolume, recording what was asked."""

    def __init__(self, level=0.9):
        self.level = level
        self.mute = False

    def GetMasterVolumeLevelScalar(self):
        return self.level

    def SetMasterVolumeLevelScalar(self, value, _guid):
        self.level = value

    def SetMute(self, state, _guid):
        self.mute = state


def test_volume_backend_reports_a_known_path():
    assert volume_backend() in {"core audio", "media keys"}


# --- parsing: the shape of what was said ------------------------------------


def test_unmute_is_matched_before_mute():
    # "unmute" contains "mute": collapsing both into one toggle was the bug.
    assert parse_volume_command("unmute the volume") == VolumeCommand("unmute")
    assert parse_volume_command("mute the volume") == VolumeCommand("mute")


def test_levels_from_digits_and_spoken_words():
    assert parse_volume_command("set volume to 30") == VolumeCommand("level", 30)
    assert (
        parse_volume_command("set volume to thirty percent")
        == VolumeCommand("level", 30)
    )
    assert (
        parse_volume_command("set volume to fifty percent")
        == VolumeCommand("level", 50)
    )
    assert parse_volume_command("set volume to 300") == VolumeCommand("level", 100)


def test_directions_and_an_honest_none():
    assert parse_volume_command("turn up the sound") == VolumeCommand("up")
    assert parse_volume_command("lower my volume") == VolumeCommand("down")
    assert parse_volume_command("what time is it") is None
    assert parse_volume_command("") is None


def test_the_field_phrases_resolve_to_levels():
    """The exact live phrases that exposed the gap: a level, not a nudge."""
    assert parse_volume_command("increase volume to 100") == VolumeCommand("level", 100)
    assert parse_volume_command("make the volume to 30") == VolumeCommand("level", 30)
    assert parse_volume_command("volume to half") == VolumeCommand("level", 50)
    assert parse_volume_command("turn the volume up to 80") == VolumeCommand("level", 80)


# --- dry run: describe the intent, touch nothing -----------------------------


def test_dry_run_replies_describe_the_intent():
    assert apply_volume(VolumeCommand("mute"), dry_run=True) == "Would mute the volume"
    assert (
        apply_volume(VolumeCommand("unmute"), dry_run=True)
        == "Would unmute the volume"
    )
    assert (
        apply_volume(VolumeCommand("level", 30), dry_run=True)
        == "Would set the volume to 30 percent"
    )
    assert (
        apply_volume(VolumeCommand("up"), dry_run=True)
        == "Would turn the volume up"
    )
    assert (
        apply_volume(VolumeCommand("down"), dry_run=True)
        == "Would turn the volume down"
    )


def test_dry_run_never_probes_the_backend(monkeypatch):
    """Dry run must not touch the machine's real volume — even to read it."""
    from nova_agent.tools import system

    touched: list[str] = []
    monkeypatch.setattr(system, "_endpoint", lambda: touched.append("endpoint"))
    monkeypatch.setattr(system, "_press_keys", lambda key, count: touched.append(key))

    apply_volume(VolumeCommand("level", 30), dry_run=True)
    apply_volume(VolumeCommand("mute"), dry_run=True)
    apply_volume(VolumeCommand("up"), dry_run=True)

    assert touched == []


def test_processor_end_to_end_sets_a_spoken_level_in_dry_run():
    processor = CommandProcessor(
        FakeSTT("set volume to fifty percent"),
        FakeRouter({"action": "system_control", "target": "volume"}),
        FakeTTS(),
    )

    assert processor.process("command.wav") == "Would set the volume to 50 percent"


def test_processor_asks_when_no_volume_action_was_recognized():
    processor = CommandProcessor(
        FakeSTT("volume"),
        FakeRouter({"action": "system_control", "target": "volume"}),
        FakeTTS(),
    )

    assert (
        processor.process("command.wav")
        == "Do you want the volume up, down, or muted?"
    )


# --- live: every call lands on a fake endpoint, never the real one -----------


def test_live_set_level_reads_back_what_the_endpoint_reports(monkeypatch):
    from nova_agent.tools import system

    endpoint = FakeEndpoint(level=0.2)
    monkeypatch.setattr(system, "_endpoint", lambda: endpoint)

    response = system.apply_volume(VolumeCommand("level", 50), dry_run=False)

    assert endpoint.level == 0.5
    assert response == "Volume set to 50 percent."  # the honest, read-back number


def test_live_mute_and_unmute_set_real_state_not_a_toggle(monkeypatch):
    from nova_agent.tools import system

    endpoint = FakeEndpoint()
    monkeypatch.setattr(system, "_endpoint", lambda: endpoint)

    assert (
        system.apply_volume(VolumeCommand("mute"), dry_run=False)
        == "Muted the volume."
    )
    assert endpoint.mute is True
    assert (
        system.apply_volume(VolumeCommand("unmute"), dry_run=False)
        == "Unmuted the volume."
    )
    assert endpoint.mute is False  # unmute on an unmuted machine stays unmuted


def test_live_nudge_steps_ten_percent_and_clamps(monkeypatch):
    from nova_agent.tools import system

    endpoint = FakeEndpoint(level=0.95)
    monkeypatch.setattr(system, "_endpoint", lambda: endpoint)

    system.apply_volume(VolumeCommand("up"), dry_run=False)
    assert endpoint.level == 1.0  # clamped at the top, never past it

    system.apply_volume(VolumeCommand("down"), dry_run=False)
    assert round(endpoint.level * 100) == 90  # one 10% step back down


def test_without_pycaw_the_media_key_fallback_is_honest(monkeypatch):
    from nova_agent.tools import system

    pressed: list[tuple[str, int]] = []
    monkeypatch.setattr(system, "_endpoint", lambda: None)
    monkeypatch.setattr(
        system, "_press_keys", lambda key, presses: pressed.append((key, presses))
    )

    assert (
        system.apply_volume(VolumeCommand("level", 30), dry_run=False)
        == "I can't set an exact volume level without pycaw installed."
    )
    assert system.apply_volume(VolumeCommand("mute"), dry_run=False) == "Toggled mute"
    assert system.apply_volume(VolumeCommand("up"), dry_run=False) == "Turned the volume up"
    assert pressed == [("volumemute", 1), ("volumeup", 5)]
