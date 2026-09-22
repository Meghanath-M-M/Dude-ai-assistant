"""Match the wake phrase inside an idle transcript.

Transcript wake lets the assistant answer to a phrase the ONNX sound model
was never trained for: while idle, speech is segmented by the VAD,
transcribed, and checked here. Matching is word-boundary aware so "hey dudes"
cannot wake the assistant, and any words after the phrase are handed back so
"hey dude, open chrome" still runs in a single breath.
"""

from __future__ import annotations

import re

_NON_WORD = re.compile(r"[^a-z0-9\s]+")


def normalize(text: str) -> str:
    """Lowercase and strip punctuation so transcripts compare cleanly."""
    lowered = _NON_WORD.sub(" ", (text or "").lower())
    return " ".join(lowered.split())


def wake_phrase_hits(text: str, phrase: str) -> tuple[bool, str]:
    """Whether ``text`` contains ``phrase`` at word boundaries.

    Returns ``(matched, remainder)`` where ``remainder`` is the normalized
    text after the phrase (``"hey dude, open chrome" -> "open chrome"``) and
    is empty when the phrase closes the utterance.
    """
    transcript = normalize(text)
    target = normalize(phrase)
    if not transcript or not target:
        return False, ""

    search = 0
    while True:
        start = transcript.find(target, search)
        if start < 0:
            return False, ""
        end = start + len(target)
        before_ok = start == 0 or transcript[start - 1] == " "
        after_ok = end == len(transcript) or transcript[end] == " "
        if before_ok and after_ok:
            return True, transcript[end:].strip()
        search = start + 1
