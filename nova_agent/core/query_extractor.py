"""Pull the argument out of a spoken command.

The intent router only needs a label ("this is a web search"), but the tools
need the payload that followed the trigger phrase:

    "search the web for rust ownership" -> "rust ownership"
    "open my ml project"                -> "ml"
"""

from __future__ import annotations

import re

from nova_agent.core.intent_router import IntentRouter

# Longest triggers are tried first so "search the web for" wins over "search".
SEARCH_TRIGGERS = (
    "search the web for",
    "search the web",
    "search online for",
    "search for",
    "google for",
    "google this",
    "look up",
    "lookup",
    "find this online",
    "find online",
    "google",
    "search",
)

PROJECT_TRIGGERS = (
    "launch my workspace",
    "open my workspace",
    "open the project",
    "open my project",
    "open the folder",
    "open project",
    "open workspace",
    "open folder",
    "open my",
    "open the",
    "launch my",
    "launch the",
    "launch",
    "open",
    "show",
    "start",
)

GENERIC_PROJECT_WORDS = {
    "project",
    "projects",
    "folder",
    "folders",
    "workspace",
    "workspaces",
    "directory",
    "directories",
    "repo",
    "repository",
}

APP_TRIGGERS = ("open", "launch", "start", "run", "show me", "show", "switch to")
APP_WORDS = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "browser": "chrome",
    "vscode": "code",
    "vs code": "code",
    "code": "code",
    "visual studio code": "code",
    "editor": "code",
}

SET_PROJECT_PATTERNS = (
    re.compile(r"set (?:the )?(?:project|folder) (?P<name>.+?) to (?P<path>.+)$"),
    re.compile(r"set (?P<name>.+?) project to (?P<path>.+)$"),
    re.compile(r"remember (?:this )?(?:folder|path) (?P<path>.+) as (?P<name>.+)$"),
)

SET_BROWSER_PATTERNS = (
    re.compile(r"set (?:my )?(?:default )?browser to (?P<app>.+)$"),
    re.compile(r"use (?P<app>.+?) as (?:my )?(?:default )?browser$"),
)

WAKE_THRESHOLD_PATTERN = re.compile(
    r"wake (?:word )?threshold (?:to )?(?P<value>\d+(?:\.\d+)?)"
)
DRY_RUN_ON_PATTERNS = (
    re.compile(r"(?:enable|turn on|switch to|start|use) dry ?run"),
)
DRY_RUN_OFF_PATTERNS = (
    re.compile(r"(?:disable|turn off|stop|quit) dry ?run"),
    re.compile(r"\bgo live\b"),
    re.compile(r"switch to live(?: mode)?"),
)
VOICE_PATTERNS = (
    re.compile(r"set (?:the |my )?voice to (?P<value>.+)$"),
    re.compile(r"switch (?:my |the )?voice to (?P<value>.+)$"),
    re.compile(r"change (?:my |the )?voice to (?P<value>.+)$"),
    re.compile(r"use (?:the |my )?voice (?P<value>.+)$"),
)

_TRAILING_PUNCTUATION = " ,.!?;:-"
_LEADING_STOPWORDS = ("for", "about", "the", "me", "on")


def _strip_fillers(text: str) -> str:
    return " ".join(
        token for token in text.split() if token.lower() not in IntentRouter.FILLER_WORDS
    )


def _strip_leading_stopwords(text: str) -> str:
    tokens = text.split()
    while tokens and tokens[0].lower() in _LEADING_STOPWORDS:
        tokens.pop(0)
    return " ".join(tokens)


def extract_search_query(text: str) -> str:
    """Return the part of ``text`` that should be searched for.

    The longest matching trigger wins, so "search the web for rust" resolves to
    "rust" instead of losing the phrase to the shorter "search" trigger.
    """
    stripped = text.strip()
    if not stripped:
        return ""

    lowered = stripped.lower()
    for trigger in sorted(SEARCH_TRIGGERS, key=len, reverse=True):
        index = lowered.find(trigger)
        if index == -1:
            continue
        remainder = stripped[index + len(trigger) :].strip(_TRAILING_PUNCTUATION)
        # An empty remainder means the user only said the trigger phrase.
        return _strip_leading_stopwords(remainder)

    # No trigger phrase at all: strip the filler words and use what is left.
    return _strip_fillers(stripped.strip(_TRAILING_PUNCTUATION))


def extract_project_name(text: str) -> str:
    """Return the likely project name from a context-open command.

    Returns an empty string for generic requests such as "open my project",
    which callers resolve with the stored default project instead.
    """
    stripped = text.strip()
    if not stripped:
        return ""

    lowered = stripped.lower()
    for trigger in sorted(PROJECT_TRIGGERS, key=len, reverse=True):
        index = lowered.find(trigger)
        if index == -1:
            continue
        stripped = stripped[index + len(trigger) :]
        break

    tokens = [
        token
        for token in stripped.strip(_TRAILING_PUNCTUATION).split()
        if token.lower() not in GENERIC_PROJECT_WORDS
        and token.lower() not in IntentRouter.FILLER_WORDS
    ]
    return " ".join(tokens)


def looks_like_open_command(text: str) -> bool:
    """Whether the user asked to open *something*, recognised or not.

    This separates "open notepad" (an app the assistant cannot launch) from a bare
    "chrome", so the first gets an honest refusal instead of opening the browser.
    """
    lowered = " ".join(text.strip().lower().split())
    return any(
        lowered == trigger or lowered.startswith(f"{trigger} ")
        for trigger in APP_TRIGGERS
        if trigger != "switch to"
    )


def extract_app_phrase(text: str) -> str | None:
    """Return the app phrase as spoken ("vs code"), or ``None``.

    The raw phrase is kept so an alias such as "browser" can be mapped to the
    user's own preferred browser instead of the built-in default.
    """
    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        return None
    # Longest phrase first so "vs code" wins over the bare "code".
    for phrase in sorted(APP_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", normalized):
            return phrase
    return None


def extract_app_name(text: str) -> str | None:
    """Return the configured app key mentioned in an open/launch command."""
    phrase = extract_app_phrase(text)
    return APP_WORDS[phrase] if phrase else None


def parse_set_project(text: str) -> tuple[str | None, str | None]:
    """Parse "set project ml to C:/work/ml" into ``("ml", "C:/work/ml")``."""
    normalized = " ".join(text.strip().lower().split())
    for pattern in SET_PROJECT_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        name = match.group("name").strip(_TRAILING_PUNCTUATION)
        path = match.group("path").strip(_TRAILING_PUNCTUATION)
        if name and path:
            return name, path
    return None, None


def parse_set_browser(text: str) -> str | None:
    """Parse "set browser to chrome" into ``"chrome"``."""
    normalized = " ".join(text.strip().lower().split())
    for pattern in SET_BROWSER_PATTERNS:
        match = pattern.search(normalized)
        if match:
            app = match.group("app").strip(_TRAILING_PUNCTUATION)
            if app:
                return app
    return None


def parse_set_preference(text: str) -> tuple[str, str] | None:
    """Parse a spoken settings change into ``(key, value)``.

    Recognises the wake threshold (a 0..1 number), dry-run on/off phrasing,
    and a TTS voice name ("af heart" -> ``af_heart``). Returns None when
    nothing recognizable was said.
    """
    normalized = " ".join(text.strip().lower().split()).rstrip(_TRAILING_PUNCTUATION)

    for pattern in DRY_RUN_ON_PATTERNS:
        if pattern.search(normalized):
            return "dry_run", "1"
    for pattern in DRY_RUN_OFF_PATTERNS:
        if pattern.search(normalized):
            return "dry_run", "0"

    threshold = WAKE_THRESHOLD_PATTERN.search(normalized)
    if threshold:
        return "wake_threshold", threshold.group("value")

    for pattern in VOICE_PATTERNS:
        match = pattern.search(normalized)
        if match:
            name = match.group("value").strip(_TRAILING_PUNCTUATION)
            if name:
                # Spoken as two words: "af heart" -> "af_heart".
                return "tts_voice", "_".join(name.split())
    return None
