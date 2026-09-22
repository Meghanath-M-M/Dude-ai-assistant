import re
from typing import ClassVar

try:
    import ollama
except ImportError:  # pragma: no cover - optional dependency for fallback path.
    ollama = None


class LLMFallback:
    UNKNOWN: ClassVar[dict] = {"action": "unknown", "safe": True, "confidence": 0.0}

    # Only actions the dispatcher can actually run may come back from the model.
    ALLOWED_ACTIONS: ClassVar[set[str]] = {"open_app", "browser_search", "screen_ocr"}

    # The LLM may propose an app, but only these may be launched.
    ALLOWED_TARGETS: ClassVar[set[str]] = {"chrome", "code"}

    def __init__(self, model: str = "phi4-mini", timeout: float = 3.0, capabilities=None):
        self.model = model
        self.timeout = timeout
        # When attached, this is the dispatcher's own gate: never even propose
        # an action it would refuse to run.
        self.capabilities = capabilities

    def classify(self, text: str) -> dict:
        """Ask the model to pick an action, and refuse anything unexpected.

        A local model can hallucinate an action name or an app that does not
        exist, so the reply is validated against the dispatcher's allow list
        and, when a CapabilityRegistry is attached, against the dispatcher's
        own capability gate before it is handed back.
        """
        if ollama is None:
            return dict(self.UNKNOWN)

        try:
            response = self._chat(text)
        except Exception:  # noqa: BLE001
            # A slow or missing Ollama must not break the voice loop.
            return dict(self.UNKNOWN)

        raw = response["message"]["content"].strip()
        match = re.search(r"(open_app|browser_search|screen_ocr|unknown)\(([^)]*)\)", raw)
        if not match:
            return dict(self.UNKNOWN)

        action = match.group(1)
        params = match.group(2).strip().strip('"').strip("'")
        if action not in self.ALLOWED_ACTIONS:
            return dict(self.UNKNOWN)
        if self.capabilities is not None and not self.capabilities.is_enabled(action):
            # A silent capability (destructive flag off, unknown action) must
            # stay a miss rather than a proposal the dispatcher rejects.
            return dict(self.UNKNOWN)

        intent = {"action": action, "safe": True, "source": "llm", "confidence": 0.7}
        if action == "open_app":
            target = params.lower()
            if target not in self.ALLOWED_TARGETS:
                return dict(self.UNKNOWN)
            intent["target"] = target
        elif action == "browser_search":
            if not params:
                return dict(self.UNKNOWN)
            intent["query"] = params
        return intent

    def _chat(self, text: str) -> dict:
        # The timeout used to be stored but never applied — a cold phi4-mini
        # could stall the intent stage for its whole model load.
        return ollama.Client(timeout=self.timeout).chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a desktop assistant. Respond with a function call that matches the "
                        "user request. Use exactly one of: open_app(chrome), open_app(code), "
                        "browser_search(query), screen_ocr(), or unknown(). Answer with the function "
                        "call only and nothing else."
                    ),
                },
                {"role": "user", "content": text},
            ],
            options={"temperature": 0.0},
            keep_alive="5m",
        )
