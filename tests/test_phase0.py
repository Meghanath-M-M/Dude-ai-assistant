from nova_agent.config.settings import INTENTS_PATH
from nova_agent.core.intent_router import IntentRouter
from nova_agent.core.safety import requires_confirmation
from nova_agent.main import CommandProcessor


def test_intents_load():
    router = IntentRouter(intents_path=INTENTS_PATH)
    assert len(router.intents) == 6
    assert "read_screen" in router.intents


def test_safe_intents_do_not_require_confirmation():
    router = IntentRouter(intents_path=INTENTS_PATH)
    assert all(not requires_confirmation(config) for config in router.intents.values())


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
