from pathlib import Path

from nova_agent.core.context_engine import ContextEngine
from nova_agent.core.safety import requires_confirmation, confirmation_prompt
from nova_agent.main import CommandProcessor


class FakeSTT:
    def __init__(self, text):
        self.text = text

    def transcribe(self, _audio_path):
        return self.text


class FakeRouter:
    def __init__(self, intent):
        self.intent = intent

    def match(self, _text):
        return self.intent, 0.99


class FakeTTS:
    def __init__(self):
        self.messages = []

    def speak(self, message):
        self.messages.append(message)


def test_requires_confirmation_flags_destructive_actions():
    config = {"action": "delete_file", "safe": False}
    assert requires_confirmation(config) is True
    assert confirmation_prompt(config) == "About to delete file. Say confirm to proceed."


def test_command_processor_returns_confirmation_for_unsafe_action():
    tts = FakeTTS()
    processor = CommandProcessor(
        FakeSTT("delete the file"),
        FakeRouter({"action": "delete_file", "safe": False}),
        tts,
    )

    response = processor.process("command.wav")

    assert response == "About to delete file. Say confirm to proceed."
    assert tts.messages == [response]


def test_command_processor_executes_unsafe_action_after_confirmation():
    tts = FakeTTS()
    processor = CommandProcessor(
        FakeSTT("delete the file"),
        FakeRouter({"action": "delete_file", "safe": False}),
        tts,
    )

    response = processor.process("command.wav", confirmed=True)

    assert response == "The delete file action is not enabled yet"
    assert tts.messages == [response]


def test_context_engine_logs_and_retrieves_last_command(tmp_path):
    db_path = Path(tmp_path) / "nova.db"
    engine = ContextEngine(db_path)

    engine.log_command("open_app", "open chrome")
    assert engine.get_last_command() == ("open_app", "open chrome")


def test_command_processor_logs_commands_to_context_engine(tmp_path):
    db_path = Path(tmp_path) / "nova.db"
    engine = ContextEngine(db_path)
    tts = FakeTTS()
    processor = CommandProcessor(
        FakeSTT("open chrome"),
        FakeRouter({"action": "open_app", "target": "chrome"}),
        tts,
        context=engine,
    )

    processor.process("command.wav")

    assert engine.get_last_command() == ("open_app", "open chrome")
