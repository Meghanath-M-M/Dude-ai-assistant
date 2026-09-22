from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class CapabilityRegistry:
    """The gate every action passes through before it runs.

    Marking an action ``safe=False`` is what forces a spoken confirmation, so a
    new destructive action cannot reach the dispatcher without one.
    """

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

    def list_destructive(self) -> list[str]:
        return [
            name
            for name, details in self._actions.items()
            if details.get("enabled", False) and not details.get("safe", True)
        ]

    def describe(self) -> dict[str, dict[str, bool]]:
        return {name: dict(details) for name, details in self._actions.items()}

    @classmethod
    def from_actions(
        cls,
        actions: Iterable[str],
        destructive: Iterable[str] = (),
        *,
        allow_destructive: bool = True,
    ) -> CapabilityRegistry:
        """Build a registry from the implemented actions and the risky ones.

        Destructive actions stay unregistered unless the user opts in, which is
        what keeps them off the dispatcher entirely by default.
        """
        registry = cls()
        risky = set(destructive)
        for action in actions:
            if action in risky:
                registry.register(action, enabled=allow_destructive, safe=False)
            else:
                registry.register(action, enabled=True, safe=True)
        return registry


def requires_confirmation(config: dict[str, Any]) -> bool:
    return config.get("safe", True) is False
