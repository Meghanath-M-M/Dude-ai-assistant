from typing import Any


def requires_confirmation(config: dict[str, Any]) -> bool:
    """Default to safe; only explicitly unsafe actions require confirmation."""
    return config.get("safe", True) is False


def confirmation_prompt(config: dict[str, Any]) -> str:
    action = config.get("action", "this action").replace("_", " ")
    target = config.get("target")
    suffix = f" {target}" if target else ""
    return f"About to {action}{suffix}. Say confirm to proceed."


def safe_intent(config: dict[str, Any]) -> bool:
    return not requires_confirmation(config)
