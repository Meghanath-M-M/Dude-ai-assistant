"""Volume control for Windows.

pyautogui maps the multimedia keys to virtual key codes 173-175 on Windows, so
volume control needs no extra dependency beyond the existing pyautogui pin.
"""

from __future__ import annotations

VOLUME_KEYS = {
    "up": "volumeup",
    "down": "volumedown",
    "mute": "volumemute",
}

MUTE_WORDS = ("mute", "unmute", "silence")
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


def detect_volume_action(text: str) -> str | None:
    """Map a spoken volume request onto ``up``, ``down``, or ``mute``."""
    lowered = text.lower()
    if any(word in lowered for word in MUTE_WORDS):
        return "mute"
    if any(word in lowered for word in DOWN_WORDS):
        return "down"
    if any(word in lowered for word in UP_WORDS):
        return "up"
    return None


def adjust_volume(action: str, presses: int = 5, dry_run: bool = True) -> str:
    key = VOLUME_KEYS.get(action)
    if key is None:
        return f"Unknown volume action: {action}"

    if dry_run:
        return f"Would press {key} {presses} times"

    try:
        import pyautogui
    except ImportError as exc:  # pragma: no cover - depends on the local install.
        raise RuntimeError("Volume control requires pyautogui.") from exc

    pyautogui.press(key, presses=presses)
    description = {
        "up": "Turned the volume up",
        "down": "Turned the volume down",
        "mute": "Toggled mute",
    }
    return description[action]
