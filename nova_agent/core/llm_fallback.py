import re

try:
    import ollama
except ImportError:  # pragma: no cover - optional dependency for fallback path.
    ollama = None


class LLMFallback:
    def __init__(self, model: str = "qwen2.5:3b"):
        self.model = model

    UNKNOWN = {"action": "unknown", "safe": True, "confidence": 0.0}

    def classify(self, text: str) -> dict:
        if ollama is None:
            return dict(self.UNKNOWN)

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a desktop assistant. Respond with a function call that matches the user request. "
                        "Use formats like open_app(chrome), browser_search(query), screen_ocr(), or unknown(). "
                        "Answer with the function call only."
                    ),
                },
                {"role": "user", "content": text},
            ],
            options={"temperature": 0.0},
        )
        raw = response["message"]["content"].strip()
        match = re.search(r"(open_app|browser_search|screen_ocr|unknown)\(([^)]*)\)", raw)
        if not match:
            return dict(self.UNKNOWN)

        action = match.group(1)
        params = match.group(2).strip().strip('"')
        if action == "unknown":
            return dict(self.UNKNOWN)

        intent = {"action": action, "safe": True, "source": "llm", "confidence": 0.7}
        if action == "open_app":
            intent["target"] = params or "chrome"
        elif action == "browser_search":
            intent["query"] = params
        return intent
