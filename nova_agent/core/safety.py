from typing import Any


def requires_confirmation(config: dict[str, Any]) -> bool:
    """Return true unless an intent is explicitly marked safe."""
    return not config.get("safe", False)


def confirmation_prompt(config: dict[str, Any]) -> str:
    action = config.get("action", "this action").replace("_", " ")
    target = config.get("target")
    suffix = f" {target}" if target else ""
    return f"About to {action}{suffix}. Say confirm to proceed."
