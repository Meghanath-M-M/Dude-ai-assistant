from typing import Any

# Spoken replies to a confirmation prompt. Rejections are checked first: a reply
# like "no, don't do it" contains both kinds of word and must not confirm.
CONFIRMATION_WORDS = (
    "yes",
    "yeah",
    "yep",
    "confirm",
    "confirmed",
    "proceed",
    "do it",
    "go ahead",
    "sure",
    "okay",
    "ok",
)
REJECTION_WORDS = (
    "no",
    "nope",
    "cancel",
    "stop",
    "abort",
    "never mind",
    "nevermind",
    "don't",
    "dont",
    "forget it",
)
CONFIRM = "confirm"
REJECT = "reject"
UNCLEAR = "unclear"


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


def confirmation_decision(text: str) -> str:
    """Classify a spoken reply as ``confirm``, ``reject``, or ``unclear``.

    ``unclear`` matters: an unheard or ambiguous reply must cancel rather than
    let something irreversible slip through.
    """
    lowered = (text or "").lower()
    if not lowered.strip():
        return UNCLEAR
    if any(word in lowered for word in REJECTION_WORDS):
        return REJECT
    if any(word in lowered for word in CONFIRMATION_WORDS):
        return CONFIRM
    return UNCLEAR
