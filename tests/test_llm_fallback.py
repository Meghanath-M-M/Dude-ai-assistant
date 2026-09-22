import numpy as np

from nova_agent.core.intent_router import IntentRouter


class FakeFallback:
    def __init__(self):
        self.calls = 0

    def classify(self, text):
        self.calls += 1
        return {"action": "open_app", "target": "chrome", "safe": True, "source": "fallback"}


class FakeEncoder:
    def __init__(self, values):
        self.values = values

    def encode(self, texts, normalize_embeddings=True):
        if len(texts) == 1:
            return np.array([[0.2]])
        return np.array([[1.0], [0.1], [0.4]])


def test_router_uses_fallback_on_low_confidence(monkeypatch):
    router = IntentRouter(threshold=0.82, fallback=FakeFallback())
    router._encoder = FakeEncoder(["open chrome", "open vscode", "search the web"])
    router._embedding_matrix = np.array([[1.0], [0.1], [0.4]])
    router._labels = ["open_chrome", "open_vscode", "search_web"]

    monkeypatch.setattr(router, "_prepare_embeddings", lambda: None)

    intent, score = router.match("open browser")

    assert intent["action"] == "open_app"
    assert intent["target"] == "chrome"
    assert score == 0.75


def test_fallback_classifies_unknown_intent(monkeypatch):
    import nova_agent.core.llm_fallback as llm_fallback

    class FakeOllama:
        @staticmethod
        def chat(*args, **kwargs):
            return {"message": {"content": "open_app(chrome)"}}

    monkeypatch.setattr(llm_fallback, "ollama", FakeOllama)
    fallback = llm_fallback.LLMFallback()

    result = fallback.classify("open chrome")

    assert result["action"] == "open_app"
    assert result["target"] == "chrome"
