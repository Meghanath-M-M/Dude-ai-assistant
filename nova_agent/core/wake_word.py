from __future__ import annotations

import logging

import numpy as np

from nova_agent.config.settings import ASSETS_DIR

logger = logging.getLogger(__name__)

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
    # trigger point depends on which detector actually loaded. The stand-in
    # model (openWakeWord's hey_mycroft) crossed 0.5 on background chatter
    # (field round 8 false wakes), so the default sits higher now; the spoken
    # "set wake threshold to <x>" or --wake-threshold still tunes it per room.
    MODEL_THRESHOLD = 0.65
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
        wake_word: str = "hey dude",
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
        self.last_score: float | None = None
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
        except Exception as exc:  # noqa: BLE001
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

    def status(self) -> dict:
        """Everything ``--wake-status`` reports about the live detector."""
        return {
            "backend": self.backend,
            "model_path": self.model_path,
            "threshold": self.threshold,
            "last_score": self.last_score,
            "load_error": self.load_error,
        }

    def _predict_score(self, chunk: np.ndarray) -> float:
        if self._oww_model is not None:
            try:
                # openWakeWord expects int16 PCM
                pcm = (np.asarray(chunk, dtype=np.float32) * 32767).astype(np.int16)
                self._oww_model.predict(pcm)
                latest = []
                for buf in self._oww_model.prediction_buffer.values():
                    frames = list(buf)
                    if frames:
                        # The last frame reflects the audio just heard; a max over
                        # the whole buffer would re-fire on stale peaks.
                        latest.append(float(frames[-1]))
                if latest:
                    score = float(max(latest))
                    self.last_score = score
                    return score
            except Exception as exc:  # noqa: BLE001
                # Never hide why the model path broke: silent fallback to the
                # loudness heuristic is how false wakes sneak in.
                logger.warning(
                    "wake-word model scoring failed (%s: %s); using energy fallback",
                    type(exc).__name__,
                    exc,
                )

        magnitude = float(np.abs(chunk).mean())
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        energy = max(magnitude, rms)
        score = float(np.clip(energy * 6.0, 0.0, 1.0))
        self.last_score = score
        return score

    def reset(self) -> None:
        """Forget detection history (used between commands)."""
        self._wake_history.clear()
        if self._oww_model is not None:
            try:
                self._oww_model.reset()
            except Exception as exc:  # noqa: BLE001 -- reset is best effort
                logger.debug("wake model reset failed: %s", exc)

    def observe(self, chunk: np.ndarray) -> float:
        """Return the raw score for a chunk without applying trigger logic.

        Used by ``--wake-probe`` so a real threshold can be picked from what the
        user's own voice actually scores.
        """
        return self._predict_score(chunk)

    @staticmethod
    def suggest_threshold(
        scores,
        ratio: float = 0.6,
        minimum: float = 0.2,
        maximum: float = 0.9,
    ) -> float | None:
        """Suggest a trigger score from observed wake-word scores.

        The trigger sits below the peaks so a slightly quieter delivery still
        fires, and well above the noise floor recorded alongside them.
        """
        values = [float(score) for score in scores if score is not None]
        if not values:
            return None
        peak = max(values)
        if peak <= 0:
            return None
        return round(min(max(peak * ratio, minimum), maximum), 3)

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