from pathlib import Path

import numpy as np
from scipy.io import wavfile

from nova_agent.config.settings import INTENTS_PATH
from nova_agent.core.context_engine import ContextEngine
from nova_agent.core.intent_router import IntentRouter
from nova_agent.core.safety import requires_confirmation
from nova_agent.core.stt import STTEngine
from nova_agent.core.tts import TTSEngine
from nova_agent.main import CommandProcessor, speakable


def test_intents_load():
    router = IntentRouter(intents_path=INTENTS_PATH)

    assert len(router.intents) == 18
    assert "read_screen" in router.intents
    assert "greeting" in router.intents
    assert "time_check" in router.intents
    assert "repeat_last" in router.intents
    assert "identity" in router.intents
    assert "help" in router.intents
    assert "set_project" in router.intents
    assert "set_browser" in router.intents
    assert "set_preference" in router.intents
    assert "close_window" in router.intents
    assert "system_brightness" in router.intents


def test_safe_intents_do_not_require_confirmation():
    """Only the three destructive tasks may demand a spoken confirmation."""
    router = IntentRouter(intents_path=INTENTS_PATH)

    safe = {name: config for name, config in router.intents.items() if config.get("safe", True)}
    risky = {name: config for name, config in router.intents.items() if not config.get("safe", True)}

    assert all(not requires_confirmation(config) for config in safe.values())
    assert all(requires_confirmation(config) for config in risky.values())
    assert set(risky) == {"close_window", "tidy_downloads", "lock_workstation"}


def test_intent_router_matches_phrase_with_filler_words():
    router = IntentRouter(intents_path=INTENTS_PATH)
    intent, score = router.match("hello open chrome")

    assert intent is not None
    assert intent["action"] == "open_app"
    assert score >= 0.75


def test_intent_router_matches_greeting():
    router = IntentRouter(intents_path=INTENTS_PATH)
    intent, score = router.match("hello")

    assert intent is not None
    assert intent["action"] == "greet"
    assert score >= 0.75


def test_warm_up_loads_the_encoder_before_the_first_match():
    router = IntentRouter(intents_path=INTENTS_PATH)

    elapsed = router.warm_up()

    assert router._embedding_matrix is not None
    assert elapsed >= 0.0


def test_prepare_embeddings_loads_once_under_concurrent_first_use(monkeypatch):
    """The background warm thread and the first live match can race in."""
    import sys
    import threading
    import time as time_module
    import types

    constructions: list[str] = []

    class SlowEncoder:
        def __init__(self, _name):
            constructions.append(_name)
            time_module.sleep(0.05)  # long enough for racing threads to enter

        def encode(self, texts, normalize_embeddings=True):
            return np.ones((len(texts), 3))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=SlowEncoder),
    )
    router = IntentRouter(intents_path=INTENTS_PATH)

    threads = [threading.Thread(target=router._prepare_embeddings) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(constructions) == 1


def test_tts_warm_up_prepays_the_first_synthesis(monkeypatch, tmp_path):
    engine = TTSEngine(cache_dir=tmp_path)
    synthesized: list[str] = []
    monkeypatch.setattr(engine, "_synthesize", lambda text: synthesized.append(text))

    elapsed = engine.warm_up()

    assert synthesized  # the import + model-load + JIT tax is paid here...
    assert elapsed >= 0.0  # ...so the first real speak() doesn't pay it


def test_tts_preload_caches_a_reply_the_policy_would_reject(monkeypatch, tmp_path):
    """The long fixed replies are exactly the ones `should_cache` skips."""
    from nova_agent.tools.screen_reader import TESSERACT_MISSING_MESSAGE

    engine = TTSEngine(cache_dir=tmp_path)
    synthesized: list[str] = []

    def fake_synthesize(text):
        synthesized.append(text)
        return np.ones(8, dtype=np.float32)

    monkeypatch.setattr(engine, "_synthesize", fake_synthesize)
    reply = f"Screen reading is unavailable. {TESSERACT_MISSING_MESSAGE}"

    assert engine.should_cache(reply) is False  # >6 words: never cached on use
    assert engine.preload([reply, "", reply]) == 1  # de-duplicated, blanks skipped

    assert engine._cache_path(reply).exists()
    assert synthesized == [reply]

    # A second boot must not re-synthesise the whole phrasebook.
    assert engine.preload([reply]) == 0
    assert engine.last_preload_count == 0


def test_a_preloaded_reply_is_served_from_the_cache(monkeypatch, tmp_path):
    """The tts-budget fix: a preloaded reply is a file read, not a synthesis."""
    import sys
    import types

    from nova_agent.tools.screen_reader import TESSERACT_MISSING_MESSAGE

    engine = TTSEngine(cache_dir=tmp_path)
    synthesized: list[str] = []

    def fake_synthesize(text):
        synthesized.append(text)
        return np.ones(8, dtype=np.float32)

    monkeypatch.setattr(engine, "_synthesize", fake_synthesize)
    played: list[int] = []
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(
            play=lambda data, rate: played.append(len(data)),
            wait=lambda: None,
        ),
    )
    reply = f"Screen reading is unavailable. {TESSERACT_MISSING_MESSAGE}"
    engine.preload([reply])

    engine.speak(reply)

    assert played  # audio came out of the cache
    assert engine.last_synthesis_seconds == 0.0  # so the tts stage reports ~0ms
    assert synthesized == [reply]  # and nothing was synthesised twice


def test_warm_up_preloads_the_phrasebook(monkeypatch, tmp_path):
    engine = TTSEngine(cache_dir=tmp_path)
    synthesized: list[str] = []

    def fake_synthesize(text):
        synthesized.append(text)
        return np.ones(8, dtype=np.float32)

    monkeypatch.setattr(engine, "_synthesize", fake_synthesize)

    engine.warm_up(phrases=["Hello there.", "Hello there."])

    assert synthesized == ["Warm up.", "Hello there."]
    assert engine.last_preload_count == 1


def test_clear_cache_drops_every_cached_clip(monkeypatch, tmp_path):
    """The cache is keyed by phrase, not voice: changing the voice must purge
    clips synthesised with the old one, phrasebook included."""
    engine = TTSEngine(cache_dir=tmp_path)
    monkeypatch.setattr(
        engine, "_synthesize", lambda _text: np.ones(8, dtype=np.float32)
    )
    engine.preload(["Hello there.", "Goodbye. Talk to you soon."])

    assert engine.clear_cache() == 2
    assert list(tmp_path.glob("*.wav")) == []
    assert engine.clear_cache() == 0  # idempotent: a second purge finds nothing


def test_command_captures_are_transcribed_with_the_command_prompt():
    from nova_agent.main import COMMAND_PROMPT

    prompts: list = []

    class RecordingSTT:
        def transcribe(self, _audio_path, prompt=None, hotwords=None):
            prompts.append(prompt)
            return "open chrome"

    class NeverRouter:
        def match(self, _text):
            return None, 0.0

    class QuietTTS:
        def speak(self, _message):
            pass

    processor = CommandProcessor(RecordingSTT(), NeverRouter(), QuietTTS())
    processor.process("command.wav")

    assert prompts == [COMMAND_PROMPT]


def test_the_kokoro_pipeline_builds_once_under_concurrent_use(monkeypatch, tmp_path):
    """The background warm thread and the first live reply must not double-build."""
    import sys
    import threading
    import time as time_module
    import types

    constructions: list = []

    class SlowPipeline:
        def __init__(self, lang_code):
            constructions.append(lang_code)
            time_module.sleep(0.05)  # long enough for racing threads to enter

        def __call__(self, text, voice=None):
            yield None, None, np.ones(8, dtype=np.float32)

    monkeypatch.setitem(
        sys.modules, "kokoro", types.SimpleNamespace(KPipeline=SlowPipeline)
    )
    engine = TTSEngine(cache_dir=tmp_path)

    threads = [threading.Thread(target=engine._synthesize, args=("hi",)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(constructions) == 1
    assert engine.last_synthesis_seconds >= 0.0


def test_intent_router_matches_time_queries():
    router = IntentRouter(intents_path=INTENTS_PATH)
    intent, score = router.match("what time is it")

    assert intent is not None
    assert intent["action"] == "time_check"
    assert score >= 0.75


class FakeSTT:
    def __init__(self, text):
        self.text = text

    def transcribe(self, _audio_path, prompt=None, hotwords=None):
        return self.text


class FakeRouter:
    def __init__(self, intent):
        self.intent = intent

    def match(self, _text):
        return self.intent, 0.91


class FakeTTS:
    def __init__(self):
        self.messages = []

    def speak(self, message):
        self.messages.append(message)


def test_phase1_open_app_pipeline_stays_dry_run():
    tts = FakeTTS()
    processor = CommandProcessor(
        FakeSTT("open chrome"),
        FakeRouter({"action": "open_app", "target": "chrome"}),
        tts,
    )

    response = processor.process("command.wav")

    assert response.startswith("Would open chrome")
    assert tts.messages == [speakable(response)]  # full detail printed, path never spoken


def test_phase1_search_pipeline_stays_dry_run():
    tts = FakeTTS()
    processor = CommandProcessor(
        FakeSTT("search the web for local python projects"),
        FakeRouter({"action": "browser_search"}),
        tts,
    )

    response = processor.process("command.wav")

    assert response == "Would search for local python projects"
    assert tts.messages == [response]


def test_stt_falls_back_to_cpu_when_cuda_library_is_missing(monkeypatch):
    class FakeSegment:
        text = "open chrome"

    class FakeModel:
        def __init__(self, _name, device, compute_type):
            self.device = device

        def transcribe(self, _audio_path, **_kwargs):
            if self.device == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
            return iter([FakeSegment()]), None

    engine = STTEngine.__new__(STTEngine)
    engine._whisper_model = FakeModel
    engine.model_name = "small"
    engine.device = "cuda"
    engine.compute_type = "int8"
    engine.model = FakeModel("small", "cuda", "int8")

    with monkeypatch.context() as patch:
        patch.setattr("warnings.warn", lambda *args, **kwargs: None)
        assert engine.transcribe("command.wav") == "open chrome"
    assert engine.device == "cpu"


def test_tts_normalizes_tuple_tensor_like_audio():
    class FakeTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return [-1.5, 0.25, 1.5]

    audio = TTSEngine._normalize_audio((FakeTensor(), 24_000))

    assert audio.tolist() == [-1.0, 0.25, 1.0]


def test_tts_cache_writer_creates_readable_wav(tmp_path):
    cache_path = tmp_path / "response.wav"

    TTSEngine._write_cache(wavfile, cache_path, np.array([0.0, 0.25], dtype=np.float32))

    sample_rate, audio = wavfile.read(cache_path)
    assert sample_rate == 24_000
    assert len(audio) == 2


def test_tts_cache_path_is_windows_safe(tmp_path):
    engine = TTSEngine(cache_dir=tmp_path)

    cache_path = engine._cache_path("Hello! How can I help?")

    assert cache_path.name == "hello_how_can_i_help.wav"
    assert all(character not in cache_path.name for character in "<>:/\\|?*")


def test_intent_router_routes_project_phrases():
    router = IntentRouter(intents_path=INTENTS_PATH)
    intent, score = router.match("open my ml project")

    assert intent is not None
    assert intent["action"] == "context_open"
    assert score >= 0.75


def test_intent_router_routes_volume_phrases():
    router = IntentRouter(intents_path=INTENTS_PATH)
    intent, score = router.match("mute the volume")

    assert intent is not None
    assert intent["action"] == "system_control"
    assert score >= 0.75


def test_command_processor_opens_a_stored_project(tmp_path):
    context = ContextEngine(Path(tmp_path) / "nova.db")
    context.set_project("ML Projects", "C:/work/ml")
    processor = CommandProcessor(
        FakeSTT("open my ml project"),
        FakeRouter({"action": "context_open"}),
        FakeTTS(),
        context=context,
    )

    assert processor.process("command.wav") == "Would open project ml at C:/work/ml"


def test_command_processor_asks_when_no_project_is_known(tmp_path):
    context = ContextEngine(Path(tmp_path) / "nova.db")
    processor = CommandProcessor(
        FakeSTT("open my rust project"),
        FakeRouter({"action": "context_open"}),
        FakeTTS(),
        context=context,
    )

    response = processor.process("command.wav")

    assert response.startswith("I don't know where rust is")


def test_command_processor_controls_volume_in_dry_run():
    processor = CommandProcessor(
        FakeSTT("mute the volume"),
        FakeRouter({"action": "system_control", "target": "volume"}),
        FakeTTS(),
    )

    assert processor.process("command.wav") == "Would mute the volume"


def test_command_processor_reports_a_missing_tesseract_binary(monkeypatch):
    from nova_agent.tools import screen_reader

    def missing_binary(**_kwargs):
        raise RuntimeError("Tesseract OCR is not installed.")

    monkeypatch.setattr(screen_reader, "read_screen", missing_binary)
    processor = CommandProcessor(
        FakeSTT("read the screen"),
        FakeRouter({"action": "screen_ocr"}),
        FakeTTS(),
    )

    response = processor.process("command.wav")

    assert response == "Screen reading is unavailable. Tesseract OCR is not installed."


def test_lexical_fallback_refuses_one_character_fragments():
    """A bare "i" must never claim an example — field run opened VS Code on it."""
    router = IntentRouter(intents_path=INTENTS_PATH)

    intent, score = router._lexical_fallback("i")

    assert intent is None
    assert score == 0.0


def test_lexical_fallback_claims_launch_verb_plus_unknown_app():
    router = IntentRouter(intents_path=INTENTS_PATH)

    intent, score = router._lexical_fallback("open gallery")

    assert intent["action"] == "open_app"
    assert score == 0.75


def test_lexical_fallback_still_prefers_project_shape_over_generic_open():
    router = IntentRouter(intents_path=INTENTS_PATH)

    intent, _score = router._lexical_fallback("open my new project")

    assert intent["action"] == "context_open"


def test_intent_router_matches_farewell():
    router = IntentRouter(intents_path=INTENTS_PATH)

    intent, score = router.match("bye")

    assert intent is not None
    assert intent["action"] == "greet"
    assert score >= 0.75


def test_greet_replies_with_a_farewell_for_bye():
    processor = CommandProcessor(FakeSTT("bye"), FakeRouter({"action": "greet"}), FakeTTS())

    assert processor.process("command.wav") == "Goodbye. Talk to you soon."


def test_speakable_drops_paths_from_spoken_replies():
    assert (
        speakable(r"Would open chrome (C:\Program Files\Google\Chrome\chrome.exe)")
        == "Would open chrome"
    )
    assert speakable("Would open project ml at C:/work/ml") == "Would open project ml"
    assert speakable("I didn't catch that") == "I didn't catch that"
    assert speakable("Good evening. How can I help?") == "Good evening. How can I help?"
