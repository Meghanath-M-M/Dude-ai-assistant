"""Voice-settable preferences: wake threshold, dry-run, and TTS voice."""

from pathlib import Path

from nova_agent.config.settings import INTENTS_PATH, Settings
from nova_agent.core.context_engine import ContextEngine
from nova_agent.core.intent_router import IntentRouter
from nova_agent.core.query_extractor import parse_set_preference
from nova_agent.main import CommandProcessor, resolve_dry_run, stored_preference_float


def test_parse_set_preference_understands_spoken_settings():
    assert parse_set_preference("set wake threshold to 0.6") == ("wake_threshold", "0.6")
    assert parse_set_preference("wake threshold 0.42") == ("wake_threshold", "0.42")
    assert parse_set_preference("enable dry run") == ("dry_run", "1")
    assert parse_set_preference("switch to dry run") == ("dry_run", "1")
    assert parse_set_preference("disable dry run") == ("dry_run", "0")
    assert parse_set_preference("go live") == ("dry_run", "0")
    assert parse_set_preference("switch to live mode") == ("dry_run", "0")
    assert parse_set_preference("set voice to af heart") == ("tts_voice", "af_heart")
    assert parse_set_preference("use voice am_michael") == ("tts_voice", "am_michael")


def test_parse_set_preference_returns_none_for_unrelated_speech():
    assert parse_set_preference("what time is it") is None
    assert parse_set_preference("open chrome") is None
    assert parse_set_preference("set wake threshold to soon") is None


def test_router_sends_preference_phrases_to_the_new_intent():
    router = IntentRouter(intents_path=INTENTS_PATH)
    phrases = (
        "enable dry run",
        "go live",
        "set wake threshold to 0.6",
        "set voice to af heart",
    )

    for phrase in phrases:
        intent, score = router.match(phrase)
        assert intent is not None, phrase
        assert intent["action"] == "set_preference", phrase
        assert score >= 0.75, phrase


class FakeSTT:
    def transcribe(self, _audio_path, prompt=None, hotwords=None):
        return ""


class FakeRouter:
    def match(self, _text):
        return None, 0.0


class FakeTTS:
    def __init__(self):
        self.voice = "af_heart"
        self.messages = []

    def speak(self, message):
        self.messages.append(message)


class FakeWake:
    def __init__(self):
        self.threshold = 0.5


def _processor(tmp_path, **kwargs):
    context = ContextEngine(Path(tmp_path) / "nova.db")
    processor = CommandProcessor(FakeSTT(), FakeRouter(), FakeTTS(), context=context, **kwargs)
    return processor, context


def test_dry_run_preference_is_applied_and_persisted(tmp_path):
    processor, context = _processor(tmp_path, dry_run=False)

    response = processor._set_preference("enable dry run")
    assert processor.dry_run is True
    assert context.get_preference("dry_run") == "1"
    assert "Dry run is on" in response

    response = processor._set_preference("go live")
    assert processor.dry_run is False
    assert context.get_preference("dry_run") == "0"
    assert "act for real" in response


def test_wake_threshold_preference_applies_live_with_a_wake_engine(tmp_path):
    processor, context = _processor(tmp_path, wake_engine=FakeWake())

    response = processor._set_preference("set wake threshold to 0.6")

    assert processor.wake_engine.threshold == 0.6
    assert context.get_preference("wake_threshold") == "0.6"
    assert "0.6" in response
    assert "next start" not in response


def test_wake_threshold_preference_without_engine_is_saved_for_next_start(tmp_path):
    processor, context = _processor(tmp_path)

    response = processor._set_preference("set wake threshold to 0.6")

    assert context.get_preference("wake_threshold") == "0.6"
    assert "next start" in response


def test_out_of_range_wake_threshold_is_refused(tmp_path):
    processor, context = _processor(tmp_path, wake_engine=FakeWake())

    response = processor._set_preference("set wake threshold to 1.5")

    assert "above 0 and at most 1" in response
    assert context.get_preference("wake_threshold") is None
    assert processor.wake_engine.threshold == 0.5


def test_tts_voice_preference_is_applied_and_persisted(tmp_path):
    processor, context = _processor(tmp_path)

    response = processor._set_preference("set voice to af heart")

    assert processor.tts.voice == "af_heart"
    assert context.get_preference("tts_voice") == "af_heart"
    assert "af_heart" in response


def test_changing_the_voice_recaches_the_fixed_replies(tmp_path):
    """The cache is keyed by phrase, not voice — without a recache the old
    voice would keep saying every canned reply, phrasebook included."""
    import threading

    from nova_agent.main import speakable

    class CacheTTS(FakeTTS):
        def __init__(self):
            super().__init__()
            self.cleared = 0
            self.preloaded: list[str] = []
            self.done = threading.Event()

        def clear_cache(self):
            self.cleared += 1
            return 0

        def preload(self, phrases):
            self.preloaded = list(phrases)
            self.done.set()
            return len(phrases)

    processor, _context = _processor(tmp_path)
    tts = CacheTTS()
    processor.tts = tts

    response = processor._set_preference("set voice to af heart")

    assert response == "Voice set to af_heart."
    assert tts.cleared == 1  # the previous voice's clips are gone...
    assert tts.done.wait(timeout=5.0)  # ...and the phrasebook returns, re-recorded
    assert tts.preloaded == [speakable(reply) for reply in processor.canned_replies()]


def test_unknown_voice_is_refused(tmp_path):
    processor, context = _processor(tmp_path)

    response = processor._set_preference("set voice to chrome")

    assert "don't know that voice" in response
    assert context.get_preference("tts_voice") is None
    assert processor.tts.voice == "af_heart"


def test_unparseable_preference_says_so(tmp_path):
    processor, _context = _processor(tmp_path)

    assert "couldn't tell" in processor._set_preference("something odd")


def test_stored_preference_float_tolerates_missing_and_garbage(tmp_path):
    context = ContextEngine(Path(tmp_path) / "nova.db")

    assert stored_preference_float(context, "wake_threshold") is None
    context.set_preference("wake_threshold", "0.6")
    assert stored_preference_float(context, "wake_threshold") == 0.6
    context.set_preference("wake_threshold", "banana")
    assert stored_preference_float(context, "wake_threshold") is None
    assert stored_preference_float(None, "wake_threshold") is None


def test_resolve_dry_run_precedence(tmp_path):
    settings = Settings()
    context = ContextEngine(Path(tmp_path) / "nova.db")

    # An explicit CLI choice beats everything.
    assert resolve_dry_run(True, context, settings) is True
    assert resolve_dry_run(False, context, settings) is False

    # With no preference stored, the env/settings default decides...
    assert resolve_dry_run(None, context, settings) == bool(settings.dry_run)
    # ...but a spoken preference beats that default.
    context.set_preference("dry_run", "0")
    assert resolve_dry_run(None, context, settings) is False
    context.set_preference("dry_run", "1")
    assert resolve_dry_run(None, context, settings) is True
    # And --live still overrides a stored "dry run" preference.
    assert resolve_dry_run(False, context, settings) is False
