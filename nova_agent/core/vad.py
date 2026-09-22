from __future__ import annotations

from pathlib import Path

import numpy as np

from nova_agent.config.settings import COMMAND_DIR, VAD_MODEL_PATH

SILERO_WINDOW = 512  # samples per Silero VAD frame at 16 kHz


class SileroVAD:
    """Stateful wrapper around the Silero VAD ONNX model.

    Silero expects fixed 512-sample windows plus its LSTM state, so callers hand
    over whatever block size the microphone produced and get one probability per
    complete window back.
    """

    def __init__(
        self, model_path: str | Path | None = VAD_MODEL_PATH, sample_rate: int = 16_000
    ):
        self.model_path = Path(model_path) if model_path else None
        self.sample_rate = sample_rate
        self.load_error: str | None = None
        self._session = None
        self._pending = np.zeros(0, dtype=np.float32)
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._load()

    @property
    def available(self) -> bool:
        return self._session is not None

    def _load(self) -> None:
        if self.model_path is None:
            self.load_error = "no VAD model configured"
            return
        if not self.model_path.exists():
            self.load_error = f"VAD model not found: {self.model_path}"
            return
        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:  # noqa: BLE001 -- onnxruntime missing/unloadable
            self._session = None
            self.load_error = f"{type(exc).__name__}: {exc}"

    def probabilities(self, chunk: np.ndarray) -> list[float]:
        """Return one speech probability for every complete 512-sample window."""
        if not self.available:
            raise RuntimeError("Silero VAD is not loaded.")

        samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if samples.size:
            self._pending = (
                np.concatenate((self._pending, samples)) if self._pending.size else samples
            )

        results: list[float] = []
        h, c = self._h, self._c
        while self._pending.size >= SILERO_WINDOW:
            window = self._pending[:SILERO_WINDOW].reshape(1, SILERO_WINDOW)
            self._pending = self._pending[SILERO_WINDOW:]
            output, h, c = self._session.run(
                None,
                {
                    "input": window,
                    "sr": np.array(self.sample_rate, dtype=np.int64),
                    "h": h,
                    "c": c,
                },
            )
            results.append(float(np.asarray(output).reshape(-1)[0]))
        self._h, self._c = h, c
        return results

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)


class VADRecorder:
    """Voice-command capture backed by Silero VAD, with an amplitude fallback.

    Speech starts the recording and a run of silent frames (the hangover) ends
    it, so a breath mid-sentence does not cut the command short. A minimum
    number of speech frames keeps a cough from becoming a command, and an
    optional ``max_seconds`` cap closes segments that never find a pause.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        max_silence: int = 9,
        min_speech: int = 1,
        speech_threshold: float = 0.5,
        voice_threshold: float = 0.025,
        vad_model_path: str | None = str(VAD_MODEL_PATH),
        command_path: Path | None = None,
        max_seconds: float | None = None,
    ):
        self.sample_rate = sample_rate
        self.max_silence = max_silence
        self.min_speech = min_speech
        self.speech_threshold = speech_threshold
        self.voice_threshold = voice_threshold
        self.max_seconds = max_seconds
        self.buffered_seconds = 0.0
        self.command_path = (
            Path(command_path) if command_path is not None else COMMAND_DIR / "command.wav"
        )
        self.buffer: list[np.ndarray] = []
        self.is_recording = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.vad = SileroVAD(vad_model_path, sample_rate) if vad_model_path else None
        self._last_probability = 0.0

    @property
    def backend(self) -> str:
        """Which detector is live: the trained model or the energy fallback."""
        return "silero" if (self.vad is not None and self.vad.available) else "energy"

    def reset(self) -> None:
        """Forget any partial capture (used before a confirmation prompt)."""
        self.buffer = []
        self.is_recording = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.buffered_seconds = 0.0
        self._last_probability = 0.0
        if self.vad is not None:
            self.vad.reset()

    def process_chunk(self, chunk: np.ndarray):
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.size == 0:
            return None

        if self._is_speech(chunk):
            if not self.is_recording:
                self.is_recording = True
                self.buffer = []
                self.speech_frames = 0
                self.buffered_seconds = 0.0
            self.buffer.append(chunk)
            self.speech_frames += 1
            self.silence_frames = 0
            self.buffered_seconds += chunk.size / self.sample_rate
            if self.max_seconds is not None and self.buffered_seconds >= self.max_seconds:
                return self._finalize()
            return None

        if not self.is_recording:
            return None

        self.silence_frames += 1
        self.buffer.append(chunk)
        self.buffered_seconds += chunk.size / self.sample_rate

        if self.silence_frames > self.max_silence:
            return self._finalize()
        if self.max_seconds is not None and self.buffered_seconds >= self.max_seconds:
            return self._finalize()
        return None

    def _finalize(self):
        """Close the recording, discarding captures that never held speech."""
        audio = np.concatenate(self.buffer)
        had_voice = self.speech_frames >= self.min_speech

        self.buffer = []
        self.is_recording = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.buffered_seconds = 0.0
        if self.vad is not None:
            self.vad.reset()

        if not had_voice:
            return None
        return self._write_audio(audio)

    def _is_speech(self, chunk: np.ndarray) -> bool:
        if self.vad is not None and self.vad.available:
            windows = self.vad.probabilities(chunk)
            if windows:
                self._last_probability = max(windows)
            # A chunk shorter than one window keeps the previous decision.
            return self._last_probability >= self.speech_threshold

        amplitude = float(np.mean(np.abs(chunk)))
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        return max(amplitude, rms) >= self.voice_threshold

    def _write_audio(self, audio: np.ndarray) -> Path:
        from uuid import uuid4

        from scipy.io.wavfile import write

        # A unique temp name per capture: concurrent or back-to-back commands
        # never clobber a recording still being transcribed.
        target = self.command_path.with_name(
            f"{self.command_path.stem}_{uuid4().hex[:8]}{self.command_path.suffix}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        write(target, self.sample_rate, (audio * 32767).astype(np.int16))
        return target
