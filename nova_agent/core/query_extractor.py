"""Pull the argument out of a spoken command.

The intent router only needs a label ("this is a web search"), but the tools
need the payload that followed the trigger phrase:

    "search the web for rust ownership" -> "rust ownership"
    "open my ml project"                -> "ml"
"""

from __future__ import annotations

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
