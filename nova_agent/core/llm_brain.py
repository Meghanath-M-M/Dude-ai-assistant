"""Wave 2: the LLM tool-calling brain (``NOVA_LLM_BRAIN=1``).

Where Tier 2 (``llm_fallback.py``) *parses a pseudo-function string* out of free
text and may only ever propose three allow-listed actions, the brain speaks
Ollama's native tool-calling protocol:

* **Tools are derived from the dispatcher itself** — one tool per intent in
  ``intents.json`` whose ``action`` is in ``IMPLEMENTED_ACTIONS`` — so the model
  can only ever select an action the dispatcher can actually run.
* A tool call resolves to that intent's own config (exactly what Tier 1 would
  have returned), gated again by the ``CapabilityRegistry`` before it is
  handed back. The tool takes **no arguments on purpose**: every action reads
  its concrete details (app name, query, level, reminder text) from the
  transcript through field-hardened parsers, so trusting model-supplied
  arguments would only weaken them.
* When the model calls **no** tool, its content is a conversational reply,
  streamed **sentence by sentence** through ``on_sentence`` — each sentence is
  a ``tts.speak`` call, so the Wave 1a chunked, bargeable playback applies and
  you can talk over an answer mid-thought.

The brain is a catch-all, not a band: it only runs when Tier 1 resolved
nothing (no confident match, no lexical claim). Failures — no Ollama, a
timeout, a hallucinated tool — all degrade to an honest miss, never a guess.
"""

import re
from dataclasses import dataclass
from typing import Any

try:
    import ollama
except ImportError:  # pragma: no cover - optional dependency for the brain path.
    ollama = None

# Flush a sentence to speech once the buffer grows past this without terminal
# punctuation, so a rambling model still gets spoken in audible bites.
MAX_SENTENCE_CHARS = 160

SYSTEM_PROMPT = (
    "You are Dude, a local desktop voice assistant. When the user asks you to "
    "DO something on the computer, call exactly one matching tool and add no "
    "prose. For anything else — questions, chat, explanations — reply "
    "conversationally in one or two short sentences, as if spoken aloud."
)


@dataclass
class Decision:
    """The outcome of one brain turn: a tool to run, a reply to speak, or neither."""

    tool_intent: dict | None = None
    reply_text: str | None = None
    score: float = 0.0

    @property
    def is_miss(self) -> bool:
        return self.tool_intent is None and self.reply_text is None


def take_sentences(buffer: str, force: bool) -> tuple[list[str], str]:
    """Split accumulated stream text into whole sentences.

    Returns ``(sentences, remaining)``. ``force`` flushes a trailing fragment
    that never saw its period; without it, a long punctuated-less run is cut at
    the last space so speech is never starved (``MAX_SENTENCE_CHARS``).
    """
    sentences: list[str] = []
    while True:
        match = re.search(r"[.!?][\"')\]]?(?=\s|$)", buffer)
        if match:
            end = match.end()
            sentences.append(buffer[:end].strip())
            buffer = buffer[end:].lstrip()
        elif force and buffer.strip():
            sentences.append(buffer.strip())
            buffer = ""
        elif not force and len(buffer) >= MAX_SENTENCE_CHARS:
            cut = buffer.rfind(" ", 0, MAX_SENTENCE_CHARS)
            if cut <= 0:
                break
            sentences.append(buffer[:cut].strip())
            buffer = buffer[cut:].lstrip()
        else:
            break
    return sentences, buffer


def build_tool_schemas(intents: dict[str, dict], implemented: Any) -> list[dict]:
    """One OpenAI-style tool per intent the dispatcher can actually run.

    The tool's *name* is the intent label (labels are unique; actions are not —
    ``system_volume`` and ``system_brightness`` share ``system_control``), so a
    call maps straight back to ``intents[label]``. Parameters are empty by
    design: the transcript carries the arguments (see the module docstring).
    """
    implemented = set(implemented)
    schemas: list[dict] = []
    for label, config in intents.items():
        if config.get("action") not in implemented:
            continue
        examples = config.get("examples") or []
        if examples:
            description = f"{label.replace('_', ' ')} — e.g. {examples[0]!r}"
        else:
            description = label.replace("_", " ")
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": label,
                    "description": description[:512],
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )
    return schemas


class LLMBrain:
    def __init__(
        self,
        model: str = "phi4-mini",
        timeout: float = 3.0,
        capabilities=None,
        intents: dict[str, dict] | None = None,
        implemented: Any = (),
    ):
        self.model = model
        self.timeout = timeout
        # The dispatcher's own gate: never even propose an action it would refuse.
        self.capabilities = capabilities
        self.intents = intents or {}
        self.tools = build_tool_schemas(self.intents, implemented)

    def decide(self, text: str, on_sentence=None) -> Decision:
        """Resolve one utterance: a tool to run, a reply to speak, or a miss.

        ``on_sentence`` is called with each whole sentence of a conversational
        reply as it completes, so the caller can speak it immediately (the Wave
        1a chunked TTS makes each sentence interruptible). A tool call wins over
        any content; every failure path returns a miss.
        """
        if ollama is None or not self.tools:
            return Decision()

        try:
            stream = self._stream(text)
        except Exception:  # noqa: BLE001 - a slow or missing Ollama must not break the loop
            return Decision()

        tool_fn: dict | None = None
        parts: list[str] = []
        buffer = ""
        for chunk in stream:
            message = chunk.get("message") or {}
            calls = message.get("tool_calls") or []
            if calls and tool_fn is None:
                tool_fn = calls[0].get("function") or {}
            content = message.get("content") or ""
            # Speak reply content only while no tool has claimed the turn.
            if content and tool_fn is None:
                parts.append(content)
                buffer += content
                if on_sentence is not None:
                    sentences, buffer = take_sentences(buffer, force=False)
                    for sentence in sentences:
                        on_sentence(sentence)

        if tool_fn is not None:
            return self._tool_decision(tool_fn)

        reply = "".join(parts).strip()
        if reply:
            if on_sentence is not None:
                leftover, buffer = take_sentences(buffer, force=True)
                for sentence in leftover:
                    on_sentence(sentence)
            return Decision(reply_text=reply, score=0.9)
        return Decision()

    def _tool_decision(self, tool_fn: dict) -> Decision:
        """Turn a model tool call into an intent — or a miss if it can't run."""
        label = tool_fn.get("name")
        config = self.intents.get(label)
        if not isinstance(config, dict):
            return Decision()  # hallucinated tool name
        action = config.get("action")
        if action is None:
            return Decision()
        if self.capabilities is not None and not self.capabilities.is_enabled(action):
            # A silent capability (destructive flag off) must stay a miss rather
            # than a proposal the dispatcher would reject.
            return Decision()
        intent = dict(config)
        intent["source"] = "brain"
        return Decision(tool_intent=intent, score=0.9)

    def _stream(self, text: str):
        # The timeout is applied to the client (a cold model once stalled the
        # intent stage for its whole load — see llm_fallback._chat).
        return ollama.Client(timeout=self.timeout).chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            tools=self.tools,
            stream=True,
            options={"temperature": 0.0},
            keep_alive="5m",
        )
