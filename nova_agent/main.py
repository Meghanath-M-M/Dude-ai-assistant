import argparse
import importlib.util
from pathlib import Path

from nova_agent.config.settings import Settings, ensure_directories
from nova_agent.core.intent_router import IntentRouter
from nova_agent.tools.app_controller import open_app


class CommandProcessor:
    def __init__(self, stt, router, tts, dry_run: bool = True):
        self.stt = stt
        self.router = router
        self.tts = tts
        self.dry_run = dry_run

    def process(self, audio_path: str | Path) -> str:
        text = self.stt.transcribe(str(audio_path))
        if not text:
            response = "I didn't catch that"
            self.tts.speak(response)
            return response

        intent, score = self.router.match(text)
        if intent is None:
            response = "I didn't catch that"
            self.tts.speak(response)
            return response

        action = intent["action"]
        if action == "open_app":
            response = open_app(intent["target"], dry_run=self.dry_run)
        else:
            response = f"The {action.replace('_', ' ')} action is not enabled yet"

        print(f"Heard: {text}")
        print(f"Intent: {action} ({score:.2f})")
        self.tts.speak(response)
        return response


def run_once() -> int:
    from nova_agent.core.recorder import record_command
    from nova_agent.core.stt import STTEngine
    from nova_agent.core.tts import TTSEngine

    settings = Settings()
    recording = Path("temp_command.wav")
    print("Press Enter, then speak your command...")
    input()
    record_command(recording, settings.command_seconds, settings.sample_rate)
    processor = CommandProcessor(
        STTEngine(settings.stt_model, settings.stt_device, settings.stt_compute_type),
        IntentRouter(settings.intent_threshold),
        TTSEngine(),
    )
    processor.process(recording)
    return 0


def check_environment() -> int:
    ensure_directories()
    settings = Settings()
    router = IntentRouter(settings.intent_threshold)
    print("Nova Phase 0 check")
    print(f"Project root: {router.project_root}")
    print(f"Intents loaded: {len(router.intents)}")
    print(f"Sample rate: {settings.sample_rate} Hz")
    for package in ("numpy", "scipy", "sounddevice"):
        status = "installed" if importlib.util.find_spec(package) else "missing"
        print(f"{package}: {status}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Nova local voice agent")
    parser.add_argument("--check", action="store_true", help="validate the Phase 0 environment")
    parser.add_argument("--once", action="store_true", help="record and process one command")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(check_environment())
    if args.once:
        raise SystemExit(run_once())
    print("Nova is scaffolded. Run `python -m nova_agent --once` to process one command.")


if __name__ == "__main__":
    main()
