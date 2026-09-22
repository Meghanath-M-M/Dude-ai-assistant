from __future__ import annotations

import numpy as np

from nova_agent.config.settings import ASSETS_DIR

# openWakeWord does not bundle its feature models in the wheel; the copies in
# this project keep the classifier runnable offline. Resolved absolutely so the
# detector survives being launched from another working directory.
DEFAULT_MELSPEC_PATH = str(ASSETS_DIR / "wake_word" / "melspectrogram.onnx")
DEFAULT_EMBEDDING_PATH = str(ASSETS_DIR / "wake_word" / "embedding_model.onnx")


class WakeWordEngine:
    """Wake-word detector using openWakeWord with an energy-based fallback.

    When an ONNX model is available, it uses the full openWakeWord pipeline
    (mel spectrogram + embedding + classifier). When the model is missing or
    incompatible, it falls back to an amplitude-threshold detector with
    smoothing and hysteresis to reduce false wake-ups from noise bursts.
    """

    # Classifier scores and loudness scores live on different scales, so the
    # trigger point depends on which detector actually loaded.
    MODEL_THRESHOLD = 0.5
    ENERGY_THRESHOLD = 0.13

    @staticmethod
    def resolve_threshold(threshold: float | None, using_model: bool) -> float:
        """Pick the trigger point for the detector that actually loaded."""
        if threshold is not None:
            return float(threshold)
        return WakeWordEngine.MODEL_THRESHOLD if using_model else WakeWordEngine.ENERGY_THRESHOLD

    @staticmethod
    def estimate_threshold(audio: np.ndarray, floor: float = 0.08) -> float:
        signal = np.asarray(audio, dtype=np.float32)
        if signal.size == 0:
            return floor
        mean_abs = float(np.abs(signal).mean())
        rms = float(np.sqrt(np.mean(np.square(signal))))
        energy = max(mean_abs, rms)
        recommended = min(max(energy * 3.0, floor), 0.5)
        return float(recommended)

    def __init__(
        self,
        model_path: str = "assets/wake_word/hey_nova.onnx",
        threshold: float | None = None,
        wake_word: str = "hey nova",
        strong_margin: float = 0.2,
        sustain_window: int = 3,
        sustain_margin: float = 0.08,
        melspec_path: str = DEFAULT_MELSPEC_PATH,
        embedding_path: str = DEFAULT_EMBEDDING_PATH,
    ):
        self.model_path = model_path
        self.wake_word = wake_word
        self.strong_margin = strong_margin
        self.sustain_window = sustain_window
        self.sustain_margin = sustain_margin
        self.load_error: str | None = None
        self._oww_model = None
        self._wake_history: list[float] = []
        self._load_model(model_path, melspec_path, embedding_path)
        self.threshold = self.resolve_threshold(threshold, self.using_model)

    def _load_model(self, model_path: str, melspec_path: str, embedding_path: str) -> None:
        """Load the openWakeWord pipeline and record why it failed if it did.

        A failure here silently degrades detection to a loudness heuristic, so
        the reason is kept on ``load_error`` instead of being swallowed.
        """
        if not model_path or model_path == "dummy.onnx":
            self.load_error = "no wake word model configured"
            return

        from pathlib import Path

        path = Path(model_path)
        if not path.exists():
            self.load_error = f"wake word model not found: {model_path}"
            return

        try:
            from openwakeword.model import Model

            extra_paths = {}
            melspec = Path(melspec_path)
            embedding = Path(embedding_path)
            if melspec.exists():
                extra_paths["melspec_model_path"] = str(melspec)
            if embedding.exists():
                extra_paths["embedding_model_path"] = str(embedding)

            self._oww_model = Model(
                wakeword_models=[str(path)],
                inference_framework="onnx",
                **extra_paths,
            )
        except Exception as exc:
            # Never take the agent down over a detector problem, but never hide it either.
            self._oww_model = None
            self.load_error = f"{type(exc).__name__}: {exc}"

    @property
    def backend(self) -> str:
        """Which detector is live: the trained model or the energy fallback."""
        return "openwakeword" if self._oww_model is not None else "energy"

    @property
    def using_model(self) -> bool:
        return self._oww_model is not None

    def _predict_score(self, chunk: np.ndarray) -> float:
        if self._oww_model is not None:
            try:
                # openWakeWord expects int16 PCM
                pcm = (np.asarray(chunk, dtype=np.float32) * 32767).astype(np.int16)
                self._oww_model.predict(pcm)
                scores = []
                for model_name in self._oww_model.prediction_buffer.keys():
                    buf = list(self._oww_model.prediction_buffer[model_name])
                    if buf:
                        scores.append(float(np.max(buf)))
                if scores:
                    return float(np.max(scores))
            except Exception:
                pass

        magnitude = float(np.abs(chunk).mean())
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        energy = max(magnitude, rms)
        return float(np.clip(energy * 6.0, 0.0, 1.0))

    def process_chunk(self, chunk: np.ndarray) -> bool:
        score = self._predict_score(chunk)

        if score >= self.threshold:
            self._wake_history.append(score)
        else:
            self._wake_history.clear()

        if not self._wake_history:
            return False

        recent_window = self._wake_history[-self.sustain_window :]
        recent_average = sum(recent_window) / len(recent_window)

        strong_signal = score >= self.threshold + self.strong_margin
        sustained_signal = len(recent_window) >= self.sustain_window and (
            recent_average >= self.threshold + self.sustain_margin
        )

        if strong_signal or sustained_signal:
            self._wake_history.clear()
            return True

        return False