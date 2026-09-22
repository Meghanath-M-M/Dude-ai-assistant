from __future__ import annotations


class CapabilityRegistry:
    """Tracks which actions are enabled and whether they require confirmation."""

    def __init__(self):
        self._actions: dict[str, dict[str, bool]] = {}

    def register(self, name: str, *, enabled: bool = True, safe: bool = True) -> None:
        self._actions[name] = {"enabled": enabled, "safe": safe}

    def is_enabled(self, name: str) -> bool:
        return bool(self._actions.get(name, {}).get("enabled", False))

    def is_safe(self, name: str) -> bool:
        return bool(self._actions.get(name, {}).get("safe", True))

    def list_enabled(self) -> list[str]:
        return [name for name, details in self._actions.items() if details.get("enabled", False)]
