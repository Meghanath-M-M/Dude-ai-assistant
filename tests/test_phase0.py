from pathlib import Path

from nova_agent.config.settings import INTENTS_PATH
from nova_agent.core.context_engine import ContextEngine
from nova_agent.core.intent_router import IntentRouter
from nova_agent.core.safety import requires_confirmation
from nova_agent.main import CommandProcessor
from nova_agent.core.stt import STTEngine
from nova_agent.core.tts import TTSEngine
from scipy.io import wavfile
import numpy as np


def test_intents_load():
    router = IntentRouter(intents_path=INTENTS_PATH)
    assert len(router.intents) == 8
    assert "read_screen" in router.intents
    assert "greeting" in router.intents
    assert "time_check" in router.intents


def test_safe_intents_do_not_require_confirmation():
    router = IntentRouter(intents_path=INTENTS_PATH)
    assert all(not requires_confirmation(config) for config in router.intents.values())


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


def test_intent_router_matches_time_queries():
    router = IntentRouter(intents_path=INTENTS_PATH)
    intent, score = router.match("what time is it")

    assert intent is not None
    assert intent["action"] == "time_check"
    assert score >= 0.75


class FakeSTT:
    def __init__(self, text):
        self.text = text

    def transcribe(self, _audio_path):
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
    assert tts.messages == [response]


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

    assert processor.process("command.wav") == "Would press volumemute 5 times"


def test_command_processor_reports_a_missing_tesseract_binary(monkeypatch):
    import nova_agent.tools.screen_reader as screen_reader

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
