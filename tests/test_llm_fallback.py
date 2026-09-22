import numpy as np

from nova_agent.config.settings import Settings
from nova_agent.core.capabilities import CapabilityRegistry
from nova_agent.core.intent_router import IntentRouter
from nova_agent.main import CommandProcessor, build_router


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
    from nova_agent.core import llm_fallback

    class FakeOllama:
        class Client:
            def __init__(self, **_kwargs):
                pass

            def chat(self, *_args, **_kwargs):
                return {"message": {"content": "open_app(chrome)"}}

    monkeypatch.setattr(llm_fallback, "ollama", FakeOllama)
    fallback = llm_fallback.LLMFallback()

    result = fallback.classify("open chrome")

    assert result["action"] == "open_app"
    assert result["target"] == "chrome"


def test_the_llm_timeout_reaches_the_client(monkeypatch):
    """The timeout used to be stored but never applied to the Ollama client."""
    from nova_agent.core import llm_fallback

    seen = {}

    class FakeOllama:
        class Client:
            def __init__(self, **kwargs):
                seen.update(kwargs)

            def chat(self, *_args, **_kwargs):
                return {"message": {"content": "unknown()"}}

    monkeypatch.setattr(llm_fallback, "ollama", FakeOllama)
    fallback = llm_fallback.LLMFallback(timeout=1.5)

    fallback._chat("hi")

    assert seen["timeout"] == 1.5


class ScoreEncoder:
    """Encode any utterance to a fixed vector so Tier 1's score is controllable."""

    def __init__(self, score):
        self.score = score

    def encode(self, texts, normalize_embeddings=True):
        if len(texts) == 1:
            return np.array([[self.score]])
        return np.array([[1.0], [0.1], [0.4]])


def _band_router(monkeypatch, score, fallback):
    """A router whose best embedding score is exactly ``score`` (matrix row0 * 1)."""
    router = IntentRouter(threshold=0.82, fallback=fallback, fallback_floor=0.65)
    router._encoder = ScoreEncoder(score)
    router._embedding_matrix = np.array([[1.0], [0.1], [0.4]])
    router._labels = ["open_chrome", "open_vscode", "search_web"]
    monkeypatch.setattr(router, "_prepare_embeddings", lambda: None)
    return router


def test_fallback_only_runs_inside_the_confidence_band(monkeypatch):
    fallback = FakeFallback()
    router = _band_router(monkeypatch, 0.70, fallback)

    intent, score = router.match("wumble the frobnitz")

    assert fallback.calls == 1
    assert intent["source"] == "fallback"
    assert score == 0.75


def test_fallback_stays_silent_below_the_floor(monkeypatch):
    fallback = FakeFallback()
    router = _band_router(monkeypatch, 0.40, fallback)

    intent, score = router.match("wumble the frobnitz")

    assert fallback.calls == 0
    assert intent is None
    assert score == 0.40


def test_fallback_stays_silent_when_tier1_is_already_confident(monkeypatch):
    fallback = FakeFallback()
    router = _band_router(monkeypatch, 0.90, fallback)

    intent, score = router.match("wumble the frobnitz")

    assert fallback.calls == 0
    assert intent["action"] == "open_app"
    assert score == 0.90


def test_lexical_claims_still_preempt_the_fallback(monkeypatch):
    fallback = FakeFallback()
    router = _band_router(monkeypatch, 0.70, fallback)

    intent, score = router.match("open browser")

    assert fallback.calls == 0
    assert intent["action"] == "open_app"
    assert score == 0.75


class FakeSTT:
    def __init__(self, text):
        self.text = text

    def transcribe(self, _audio_path, prompt=None):
        return self.text


class FakeTTS:
    def __init__(self):
        self.messages = []

    def speak(self, message):
        self.messages.append(message)


def test_a_below_band_miss_becomes_a_reask(monkeypatch):
    fallback = FakeFallback()
    router = _band_router(monkeypatch, 0.40, fallback)
    processor = CommandProcessor(FakeSTT("wumble the frobnitz"), router, FakeTTS())

    response = processor.process("command.wav")

    assert response == "I didn't catch that"
    assert fallback.calls == 0


def test_fallback_refuses_a_disabled_action(monkeypatch):
    from nova_agent.core import llm_fallback

    class FakeOllama:
        class Client:
            def __init__(self, **_kwargs):
                pass

            def chat(self, *_args, **_kwargs):
                return {"message": {"content": 'open_app("chrome")'}}

    monkeypatch.setattr(llm_fallback, "ollama", FakeOllama)
    fallback = llm_fallback.LLMFallback(capabilities=CapabilityRegistry())

    result = fallback.classify("open chrome")

    assert result["action"] == "unknown"


def test_fallback_accepts_an_enabled_action(monkeypatch):
    from nova_agent.core import llm_fallback

    class FakeOllama:
        class Client:
            def __init__(self, **_kwargs):
                pass

            def chat(self, *_args, **_kwargs):
                return {"message": {"content": 'open_app("chrome")'}}

    monkeypatch.setattr(llm_fallback, "ollama", FakeOllama)
    registry = CapabilityRegistry()
    registry.register("open_app", enabled=True, safe=True)
    fallback = llm_fallback.LLMFallback(capabilities=registry)

    result = fallback.classify("open chrome")

    assert result["action"] == "open_app"
    assert result["target"] == "chrome"


def test_build_router_passes_band_model_and_registry_to_the_fallback():
    settings = Settings(llm_fallback=True)
    registry = CapabilityRegistry()
    registry.register("open_app", enabled=True, safe=True)

    router = build_router(settings, registry)

    assert router.fallback is not None
    assert router.fallback.capabilities is registry
    assert router.fallback.model == settings.llm_model
    assert router.fallback_floor == settings.llm_min_score
    # The band must be non-empty: floor below Tier 1's claim threshold.
    assert settings.llm_min_score < settings.intent_threshold
