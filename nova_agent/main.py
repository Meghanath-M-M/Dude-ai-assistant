import argparse
import importlib.util
import os
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from nova_agent.config.settings import (
    WAKE_WORD_MODEL_PATH,
    Settings,
    ensure_directories,
)
from nova_agent.core.context_engine import ContextEngine
from nova_agent.core.intent_router import IntentRouter
from nova_agent.core.query_extractor import extract_project_name, extract_search_query
from nova_agent.core.safety import confirmation_prompt, requires_confirmation
from nova_agent.core.vad import VADRecorder
from nova_agent.core.wake_word import WakeWordEngine
from nova_agent.tools.app_controller import open_app
from nova_agent.tools.browser import search_web
from nova_agent.ui.overlay import NovaHUD

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional at runtime for tests.
    sd = None


def time_of_day_greeting(now: datetime | None = None) -> str:
    """Pick a greeting that matches the cached Kokoro phrases."""
    hour = (now or datetime.now()).hour
    if 5 <= hour < 12:
        return "Good morning"
    if 12 <= hour < 17:
        return "Good afternoon"
    return "Good evening"


def build_router(settings: Settings) -> IntentRouter:
    """Create the Tier 1 router, optionally with the Tier 2 LLM fallback."""
    router = IntentRouter(settings.intent_threshold)
    if settings.llm_fallback:
        from nova_agent.core.llm_fallback import LLMFallback

        router.fallback = LLMFallback()
    return router


class CommandProcessor:
    def __init__(self, stt, router, tts, dry_run: bool = True, context=None, settings=None):
        self.stt = stt
        self.router = router
        self.tts = tts
        self.dry_run = dry_run
        self.context = context
        self.settings = settings or Settings()

    def process(self, audio_path: str | Path, confirmed: bool = False) -> str:
        text = self.stt.transcribe(str(audio_path))
        if not text:
            return self._respond("I didn't catch that")

        intent, score = self.router.match(text)
        if intent is None:
            return self._respond("I didn't catch that")

        action = intent["action"]
        print(f"Heard: {text}")
        print(f"Intent: {action} ({score:.2f})")

        if requires_confirmation(intent) and not confirmed:
            return self._respond(confirmation_prompt(intent))

        response = self.execute(action, intent, text)

        if self.context is not None:
            self.context.log_command(action, text)

        return self._respond(response)

    def execute(self, action: str, intent: dict, text: str) -> str:
        """Run one dispatched action and return the phrase to speak."""
        if action == "open_app":
            return open_app(intent["target"], dry_run=self.dry_run)

        if action == "browser_search":
            query = intent.get("query") or extract_search_query(text)
            if not query:
                return "What should I search for?"
            return search_web(query, dry_run=self.dry_run)

        if action == "screen_ocr":
            return self._read_screen()

        if action == "context_open":
            return self._open_project(text)

        if action == "system_volume":
            return self._control_volume(text)

        if action == "greet":
            return f"{time_of_day_greeting()}. How can I help?"

        if action == "time_check":
            return f"The current time is {datetime.now().strftime('%I:%M %p')}."

        return f"The {action.replace('_', ' ')} action is not enabled yet"

    def _respond(self, response: str) -> str:
        self.tts.speak(response)
        return response

    def _read_screen(self) -> str:
        from nova_agent.tools.screen_reader import read_screen

        try:
            return read_screen(
                tesseract_path=self.settings.tesseract_path,
                max_characters=self.settings.ocr_max_characters,
            )
        except RuntimeError as exc:
            return f"Screen reading is unavailable. {exc}"
        except Exception:  # pragma: no cover - depends on the local Tesseract install.
            return "Screen reading needs Tesseract OCR to be installed."

    def _open_project(self, text: str) -> str:
        if self.context is None:
            return "I don't have any projects saved yet."

        name = extract_project_name(text)
        path = self.context.find_project(name) if name else None
        if path is None and not name:
            path = self.context.get_preference("default_project")
        if path is None:
            if name:
                return f"I don't know where {name} is. Say set project {name} to a folder path."
            return "I don't know where that is. Say set project ml to a folder path."

        label = name or "default"
        if self.dry_run:
            return f"Would open project {label} at {path}"

        startfile = getattr(os, "startfile", None)
        if startfile is None:  # pragma: no cover - keeps non-Windows machines working.
            subprocess.Popen(["xdg-open", path])
        else:
            startfile(path)
        return f"Opening project {label}"

    def _control_volume(self, text: str) -> str:
        from nova_agent.tools.system import adjust_volume, detect_volume_action

        action = detect_volume_action(text)
        if action is None:
            return "Do you want the volume up, down, or muted?"
        return adjust_volume(action, dry_run=self.dry_run)


class NovaAgent:
    """Coordinate wake-word detection, command capture, and processing."""

    def __init__(
        self,
        wake_engine=None,
        vad_reader=None,
        processor=None,
        hud=None,
        debug: bool = False,
        wake_word: str = "hey nova",
        wake_threshold: float | None = None,
        context=None,
        dry_run: bool | None = None,
        queue_size: int = 64,
    ):
        self.context = context or ContextEngine()
        self.wake_engine = wake_engine or WakeWordEngine(
            model_path=str(WAKE_WORD_MODEL_PATH),
            threshold=wake_threshold,
            wake_word=wake_word,
        )
        self.vad_reader = vad_reader or VADRecorder()
        self.processor = processor
        self.hud = hud or NovaHUD()
        self.listening = False
        self.debug = debug
        self.dry_run = dry_run
        # Microphone frames are handed to a worker thread so the PortAudio
        # callback never blocks on Whisper or Kokoro playback.
        self.audio_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.is_speaking = False
        self._worker: threading.Thread | None = None
        self._stopping = False

    def process_chunk(self, chunk):
        """Return a status token or the recorded audio path for the next stage."""
        energy = float(
            np.sqrt(np.mean(np.square(np.asarray(chunk, dtype=np.float32)))) if chunk.size else 0.0
        )
        if self.debug:
            print(f"[debug] energy={energy:.4f} listening={self.listening}")

        if not self.listening:
            wake_hit = self.wake_engine.process_chunk(chunk)
            if wake_hit:
                self.listening = True
                self.hud.set_state("listening", "Listening")
                if self.debug:
                    print("[debug] wake detected")
                return "wake"
            return None

        self.hud.set_state("thinking", "Thinking")
        result = self.vad_reader.process_chunk(chunk)
        if result is None:
            return None

        self.listening = False
        if self.processor is not None:
            self.hud.set_state("speaking", "Processing")
            started = time.perf_counter()
            # Mute the pipeline while it works: Kokoro playback would otherwise
            # be captured by the open input stream and treated as a new wake word.
            self.is_speaking = True
            try:
                self._discard_pending_audio()
                self.processor.process(result)
            finally:
                self.is_speaking = False
                self._discard_pending_audio()
            if self.debug:
                print(f"[debug] command handled in {time.perf_counter() - started:.2f}s")
        self.hud.set_state("idle", "idle")
        if self.debug:
            print(f"[debug] captured command: {result}")
        return str(result)

    def _on_audio(self, indata, _frames, _time, _status) -> None:
        """PortAudio callback: queue the chunk and return immediately.

        Transcription, synthesis, and window automation must never run here;
        blocking this thread makes PortAudio drop microphone frames.
        """
        chunk = np.asarray(indata[:, 0], dtype=np.float32)
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            pass

    def process_pending(self, limit: int | None = None) -> int:
        """Drain queued microphone frames. Returns the number handled."""
        handled = 0
        while limit is None or handled < limit:
            try:
                chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            if self.is_speaking:
                continue
            self.process_chunk(chunk)
            handled += 1
        return handled

    def _discard_pending_audio(self) -> int:
        """Drop frames captured while Nova was talking to herself."""
        dropped = 0
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
            dropped += 1
        return dropped

    def _consume_loop(self) -> None:
        while not self._stopping:
            try:
                chunk = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.is_speaking:
                continue
            self.process_chunk(chunk)

    def run(self):
        if sd is None:
            raise RuntimeError("Microphone support requires sounddevice.")

        settings = Settings()
        dry_run = settings.dry_run if self.dry_run is None else self.dry_run

        if self.processor is None:
            from nova_agent.core.stt import STTEngine
            from nova_agent.core.tts import TTSEngine

            self.processor = CommandProcessor(
                STTEngine(settings.stt_model, settings.stt_device, settings.stt_compute_type),
                build_router(settings),
                TTSEngine(),
                dry_run=dry_run,
                context=self.context,
                settings=settings,
            )

        mode = "dry-run" if dry_run else "live"
        detector = getattr(self.wake_engine, "backend", "custom")
        print(
            f"NOVA starting (execution: {mode}, wake word: {settings.wake_word!r}, "
            f"detector: {detector})"
        )
        load_error = getattr(self.wake_engine, "load_error", None)
        if detector != "openwakeword" and load_error:
            print(f"Wake word note: {load_error}")

        self._stopping = False
        self._worker = threading.Thread(target=self._consume_loop, daemon=True)
        self._worker.start()

        try:
            with sd.InputStream(
                samplerate=settings.sample_rate,
                blocksize=1280,
                dtype="float32",
                channels=1,
                callback=self._on_audio,
            ):
                print("NOVA is listening. Say the wake word to begin.")
                while True:
                    sd.sleep(100)
        finally:
            self._stopping = True
            if self._worker is not None:
                self._worker.join(timeout=1.0)


def run_once(dry_run: bool = True) -> int:
    from nova_agent.core.recorder import record_command
    from nova_agent.core.stt import STTEngine
    from nova_agent.core.tts import TTSEngine

    settings = Settings()
    recording = Path("temp_command.wav")
    mode = "dry-run" if dry_run else "live"
    print(f"Press Enter, then speak your command... (execution: {mode})")
    input()
    record_command(recording, settings.command_seconds, settings.sample_rate)
    processor = CommandProcessor(
        STTEngine(settings.stt_model, settings.stt_device, settings.stt_compute_type),
        build_router(settings),
        TTSEngine(),
        dry_run=dry_run,
        context=ContextEngine(),
        settings=settings,
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
    print(f"STT: {settings.stt_model} on {settings.stt_device} ({settings.stt_compute_type})")
    print(f"Execution mode: {'dry-run' if settings.dry_run else 'live'}")
    tesseract = settings.tesseract_path or "default (PATH)"
    print(f"Tesseract: {tesseract}")
    detector = WakeWordEngine(
        model_path=str(WAKE_WORD_MODEL_PATH),
        threshold=settings.wake_threshold,
        wake_word=settings.wake_word,
    )
    print(f"Wake detector: {detector.backend}")
    if detector.load_error:
        print(f"Wake detector note: {detector.load_error}")
    for package in ("numpy", "scipy", "sounddevice", "faster_whisper", "kokoro", "openwakeword"):
        status = "installed" if importlib.util.find_spec(package) else "missing"
        print(f"{package}: {status}")
    return 0


def calibrate_microphone(seconds: float = 3.0) -> float:
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - depends on runtime environment.
        raise RuntimeError("Microphone calibration requires sounddevice.") from exc

    sample_rate = Settings().sample_rate
    frames = int(sample_rate * seconds)
    recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    audio = np.asarray(recording[:, 0], dtype=np.float32)
    recommended = WakeWordEngine.estimate_threshold(audio)
    print(f"Ambient energy sample: {float(np.abs(audio).mean()):.4f}")
    print(f"Recommended wake threshold: {recommended:.4f}")
    return recommended


def main() -> None:
    parser = argparse.ArgumentParser(description="Nova local voice agent")
    parser.add_argument("--check", action="store_true", help="validate the Phase 0 environment")
    parser.add_argument("--once", action="store_true", help="record and process one command")
    parser.add_argument("--listen", action="store_true", help="start the hands-free wake-word loop")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "sample ambient microphone energy and print a recommended threshold for "
            "the energy fallback detector"
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="actually open apps and folders instead of printing a dry run",
    )
    parser.add_argument(
        "--debug", action="store_true", help="print live microphone energy and wake detection state"
    )
    parser.add_argument(
        "--wake-word",
        default="hey nova",
        help="configured wake phrase label for the live assistant",
    )
    parser.add_argument(
        "--wake-threshold",
        type=float,
        default=None,
        help=(
            "override the wake trigger score (default: 0.5 with the openWakeWord "
            "model, 0.13 with the energy fallback)"
        ),
    )
    args = parser.parse_args()
    dry_run = False if args.live else Settings().dry_run
    if args.check:
        raise SystemExit(check_environment())
    if args.once:
        raise SystemExit(run_once(dry_run=dry_run))
    if args.calibrate:
        raise SystemExit(calibrate_microphone())
    if args.listen:
        raise SystemExit(
            NovaAgent(
                debug=args.debug,
                wake_word=args.wake_word,
                wake_threshold=args.wake_threshold,
                dry_run=dry_run,
            ).run()
        )
    print(
        "Nova is scaffolded. Run `python -m nova_agent --once`, `python -m nova_agent --calibrate`, or `python -m nova_agent --listen`."
    )


if __name__ == "__main__":
    main()
