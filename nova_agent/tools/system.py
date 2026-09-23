"""Volume control for Windows.

Absolute levels and the *state* of mute both need the Windows Core Audio API
(``pycaw``): pyautogui's multimedia keys can only nudge the level and toggle
mute. That asymmetry is why "unmute the volume" used to mute an already-unmuted
machine, and why "set volume to 30" did nothing but turn the sound up — an
example the intent file had carried since Phase 1. Core Audio is used whenever
pycaw imports; the media keys stay as the fallback so a bare install keeps
coarse volume control, and that fallback says honestly what it cannot do.

Brightness shares the module because it is the same shape of request (a
direction or an absolute level) on a different bus: levels go through Core
Audio, brightness through the monitor's ``root\\WMI`` class via the ``wmi``
package. Each parser refuses the other's phrases — the field phrase
"increase the brightness" parses as volume-up without that guard (the word
"increase" is enough), one routing change away from raising the sound when
the screen was meant.
"""

from __future__ import annotations

import re
import time
from typing import NamedTuple

# Five multimedia key presses is what "turn the volume up" used to do (Windows
# steps 2% per press), and it is the step Core Audio applies now.
DEFAULT_PRESSES = 5
VOLUME_STEP_PERCENT = 10
BRIGHTNESS_STEP_PERCENT = 10  # the same step, so both devices feel alike

VOLUME_KEYS = {"up": "volumeup", "down": "volumedown"}
MUTE_KEY = "volumemute"

# Checked before MUTE_WORDS: "unmute" contains "mute", and collapsing both into
# a single toggle was the bug this file used to have.
UNMUTE_WORDS = ("unmute", "un mute", "un-mute", "restore the sound")
MUTE_WORDS = ("mute", "silence")
DOWN_WORDS = (
    "turn down",
    "down the volume",
    "volume down",
    "quieter",
    "lower",
    "decrease",
    "reduce",
)
UP_WORDS = (
    "turn up",
    "up the volume",
    "volume up",
    "louder",
    "increase",
    "raise",
)

# Brightness parsing is gated on one of these tokens, and parse_volume_command
# refuses them in return: without both guards "increase the brightness" parses
# as volume-up ("increase" alone was enough).
BRIGHTNESS_TOKENS = ("bright", "dim", "luminance")
DIM_WORDS = ("dim", "darker", "less bright", "lower", "decrease", "reduce", "down")
BRIGHTEN_WORDS = ("brighter", "increase", "raise", "up", "more", "turn up")

# Spoken levels, so "set volume to fifty percent" (an example in
# config/intents.json) finally works.
LEVEL_WORDS = {
    "zero": 0,
    "ten": 10,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "half": 50,
    "full": 100,
    "max": 100,
    "maximum": 100,
    "one hundred": 100,
    "hundred": 100,
}
LEVEL_PATTERN = re.compile(r"(?P<value>\d{1,3})\s*(?:%|percent|per cent)?")
LEVEL_WORD_PATTERN = re.compile(
    r"\b(?P<word>"
    + "|".join(re.escape(word) for word in sorted(LEVEL_WORDS, key=len, reverse=True))
    + r")\b"
)


class VolumeCommand(NamedTuple):
    """A parsed request: an action, plus a 0-100 level for ``"level"``."""

    action: str
    percent: int | None = None


def _clamp(percent: int) -> int:
    return max(0, min(100, percent))


def _extract_percent(text: str) -> int | None:
    """Pull an absolute level out of spoken text ("30", "fifty percent")."""
    digits = LEVEL_PATTERN.search(text)
    if digits:
        return _clamp(int(digits.group("value")))
    word = LEVEL_WORD_PATTERN.search(text)
    if word:
        return LEVEL_WORDS[word.group("word")]
    return None


def parse_volume_command(text: str) -> VolumeCommand | None:
    """Parse a spoken volume request, or ``None`` when nothing was recognized.

    Order carries meaning: an explicit level beats a direction ("turn the volume
    up to 80" is a level), and unmute is matched before mute.
    """
    lowered = " ".join((text or "").lower().split())
    if not lowered:
        return None
    # Brightness phrases belong to parse_brightness_command; without this
    # "increase the brightness" parses as volume-up ("increase" is enough).
    if any(token in lowered for token in BRIGHTNESS_TOKENS):
        return None
    if any(word in lowered for word in UNMUTE_WORDS):
        return VolumeCommand("unmute")
    if any(word in lowered for word in MUTE_WORDS):
        return VolumeCommand("mute")
    percent = _extract_percent(lowered)
    if percent is not None:
        return VolumeCommand("level", percent)
    if any(word in lowered for word in DOWN_WORDS):
        return VolumeCommand("down")
    if any(word in lowered for word in UP_WORDS):
        return VolumeCommand("up")
    return None


def parse_brightness_command(text: str) -> VolumeCommand | None:
    """Parse a spoken brightness request, or ``None`` when none was meant.

    Same contract as :func:`parse_volume_command` — an explicit level beats a
    direction — but gated on a brightness token so a volume phrase can never
    land here (the mirror of the guard inside ``parse_volume_command``).
    """
    lowered = " ".join((text or "").lower().split())
    if not lowered or not any(token in lowered for token in BRIGHTNESS_TOKENS):
        return None
    percent = _extract_percent(lowered)
    if percent is not None:
        return VolumeCommand("level", percent)
    if any(word in lowered for word in DIM_WORDS):
        return VolumeCommand("down")
    if any(word in lowered for word in BRIGHTEN_WORDS):
        return VolumeCommand("up")
    return None


def _endpoint():
    """The default output endpoint's volume interface, or ``None``.

    ``None`` means pycaw is missing or the machine exposes no audio endpoint, so
    every caller degrades to the media keys instead of failing.

    Windows COM is *per thread*, and commands run on the worker thread — without
    an explicit ``CoInitialize`` pycaw's ``GetSpeakers`` fails there with
    "CoInitialize has not been called" (found on the second live test).
    """
    try:
        import comtypes

        comtypes.CoInitialize()
        from pycaw.pycaw import AudioUtilities
    except Exception:  # noqa: BLE001 -- pycaw absent, or COM refused: use the keys
        return None
    try:
        return AudioUtilities.GetSpeakers().EndpointVolume
    except Exception:  # noqa: BLE001 -- no endpoint must not kill the command
        return None


def volume_backend() -> str:
    """Which backend volume control will use — reported by ``--check``."""
    return "core audio" if _endpoint() is not None else "media keys"


def _press_keys(key: str, presses: int) -> None:
    try:
        import pyautogui
    except ImportError as exc:  # pragma: no cover - depends on the local install.
        raise RuntimeError("Media-key volume control requires pyautogui.") from exc

    pyautogui.press(key, presses=presses)


def apply_volume(command: VolumeCommand, dry_run: bool = True) -> str:
    """Run a parsed volume request and return the spoken confirmation.

    The reply describes the *intent*, so it reads the same whichever backend is
    live. The one place the fallback shows through is mute: a media-key press
    can only toggle, and saying "Toggled mute" is more honest than claiming a
    state that was never verified.
    """
    if command.action == "level":
        return _set_level(command.percent, dry_run)
    if command.action in {"mute", "unmute"}:
        return _set_mute(command.action, dry_run)
    return _nudge(command.action, dry_run)


def _set_level(percent: int | None, dry_run: bool) -> str:
    if percent is None:  # pragma: no cover - parse_volume_command always fills it
        return "I couldn't tell which volume level you wanted."
    if dry_run:
        return f"Would set the volume to {percent} percent"
    endpoint = _endpoint()
    if endpoint is None:
        return "I can't set an exact volume level without pycaw installed."
    endpoint.SetMasterVolumeLevelScalar(percent / 100, None)
    # Read it back: Windows is free to round, and the honest number is the one
    # the endpoint reports, not the one that was asked for.
    actual = round(endpoint.GetMasterVolumeLevelScalar() * 100)
    return f"Volume set to {actual} percent."


def _set_mute(action: str, dry_run: bool) -> str:
    wanted = action == "mute"
    if dry_run:
        return "Would mute the volume" if wanted else "Would unmute the volume"
    endpoint = _endpoint()
    if endpoint is None:
        _press_keys(MUTE_KEY, 1)
        return "Toggled mute"
    endpoint.SetMute(wanted, None)
    return "Muted the volume." if wanted else "Unmuted the volume."


def _nudge(action: str, dry_run: bool) -> str:
    if action not in VOLUME_KEYS:
        return f"Unknown volume action: {action}"
    if dry_run:
        return f"Would turn the volume {'up' if action == 'up' else 'down'}"
    endpoint = _endpoint()
    if endpoint is None:
        _press_keys(VOLUME_KEYS[action], DEFAULT_PRESSES)
    else:
        step = VOLUME_STEP_PERCENT / 100
        current = endpoint.GetMasterVolumeLevelScalar()
        target = current + step if action == "up" else current - step
        endpoint.SetMasterVolumeLevelScalar(min(1.0, max(0.0, target)), None)
    return "Turned the volume up" if action == "up" else "Turned the volume down"


# --- brightness: the same shapes, over the monitor's root\WMI interface ------


def _brightness_backend():
    """A ``root\\WMI`` connection with monitor support, or ``None``.

    ``None`` means the wmi package is missing or the machine exposes no
    controllable display (a desktop driving external monitors only) — callers
    must then say so honestly instead of pretending.
    """
    try:
        import wmi
    except Exception:  # noqa: BLE001 -- package absent on a bare install
        return None
    try:
        return wmi.WMI(namespace="root\\WMI")
    except Exception:  # noqa: BLE001 -- no WMI must not kill the command
        return None


def brightness_backend() -> str:
    """Which backend brightness control will use — reported by ``--check``."""
    try:
        import pythoncom
    except ImportError:  # pragma: no cover - depends on the local install.
        return "unsupported"
    pythoncom.CoInitialize()  # COM is per thread; --check runs on the main one
    try:
        backend = _brightness_backend()
        if backend is None or _wmi_brightness(backend) is None:
            return "unsupported"
        return "wmi"
    finally:
        pythoncom.CoUninitialize()


def apply_brightness(command: VolumeCommand, dry_run: bool = True) -> str:
    """Run a parsed brightness request and return the spoken confirmation.

    The same contract as :func:`apply_volume`: dry-run answers "Would ..."
    *before* COM is initialised (a dry run never touches an endpoint), a live
    run reports the level Windows actually applied, and a machine with no
    controllable display says so instead of pretending.
    """
    if command.action == "level":
        if command.percent is None:  # pragma: no cover - parse fills it in
            return "I couldn't tell which brightness level you wanted."
        if dry_run:
            return f"Would set the brightness to {command.percent} percent"
    elif command.action in {"up", "down"}:
        if dry_run:
            edge = "up" if command.action == "up" else "down"
            return f"Would turn the brightness {edge}"
    else:
        return f"Unknown brightness action: {command.action}"
    return _apply_brightness_live(command)


def _apply_brightness_live(command: VolumeCommand) -> str:
    """The live half: apply the request, then report what Windows did."""
    import pythoncom

    # Windows COM is per thread and commands run on the worker thread: without
    # this, wmi's Dispatch fails exactly like pycaw's "CoInitialize has not
    # been called" did — the same lesson, learned on a second bus.
    pythoncom.CoInitialize()
    try:
        backend = _brightness_backend()
        if backend is None:
            return "I can't change the brightness on this display."
        target: int
        if command.action == "level":
            target = int(command.percent)  # None was answered before we got here
        else:
            current = _wmi_brightness(backend)
            if current is None:
                return "I can't read the brightness on this display."
            step = BRIGHTNESS_STEP_PERCENT
            if command.action == "down":
                step = -step
            target = _clamp(current + step)
        if not _set_wmi_brightness(backend, target):
            return "I couldn't change the brightness on this display."
        if command.action != "level":
            edge = "up" if command.action == "up" else "down"
            return f"Turned the brightness {edge}"
        # Read it back — the spoken number must be the one Windows applied.
        return f"Brightness set to {_settle_brightness(backend, target)} percent."
    finally:
        pythoncom.CoUninitialize()


def _set_wmi_brightness(backend, percent: int) -> bool:
    """Apply a level; ``False`` means the driver refused (or is not there)."""
    try:
        methods = backend.WmiMonitorBrightnessMethods()
        if not methods:
            return False
        # Keywords, not positionals: this machine's MOF declares
        # (Brightness, Timeout) — the reverse of the (Timeout, Brightness)
        # order that community recipes assume, and the wrong guess blanks the
        # screen to 1%. The wmi wrapper maps keywords by name, so order is moot.
        methods[0].WmiSetBrightness(Brightness=int(percent), Timeout=1)
        return True
    except Exception:  # noqa: BLE001 -- a refusing driver must not kill the reply
        return False


def _wmi_brightness(backend) -> int | None:
    """The current level as Windows reports it, or ``None`` if unreadable."""
    try:
        readings = backend.WmiMonitorBrightness()
        if not readings:
            return None
        return int(readings[0].CurrentBrightness)
    except Exception:  # noqa: BLE001
        return None


def _settle_brightness(backend, expected: int) -> int:
    """Wait briefly for the driver to apply the level, then report it.

    ``WmiSetBrightness`` takes a Timeout whose meaning varies by driver; the
    number about to be spoken must be the number the panel shows, so poll for
    up to a second instead of racing it.
    """
    deadline = time.monotonic() + 1.0
    while True:
        actual = _wmi_brightness(backend)
        if actual is None or actual == expected or time.monotonic() >= deadline:
            return expected if actual is None else actual
        time.sleep(0.1)
