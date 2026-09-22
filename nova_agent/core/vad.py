from __future__ import annotations

from pathlib import Path

import numpy as np


class VADRecorder:
    """Very small VAD wrapper for voice-command capture.

    It is intentionally tolerant of missing optional dependencies: in the Phase 0
    environment, we still want the recorder to exist and behave predictably in
    tests without requiring a full torch or silero installation.
    """

    def __init__(self, sample_rate: int = 16_000, max_silence: int = 30):
        self.sample_rate = sample_rate
        self.max_silence = max_silence
        self.buffer: list[np.ndarray] = []
        self.is_recording = False
        self.silence_frames = 0
        self.model = None
        self.voice_threshold = 0.025

        # Avoid an eager network fetch during Phase 0 and tests. The recorder can
        # still function with a lightweight amplitude heuristic until Silero is
        # explicitly installed and trusted by the user.
        try:
            import torch  # noqa: F401
        except ImportError:
            self.model = None

    def process_chunk(self, chunk: np.ndarray):
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.size == 0:
            return None

        speech_prob = self._speech_probability(chunk)
        if speech_prob >= self.voice_threshold:
            if not self.is_recording:
                self.is_recording = True
                self.buffer = []
            self.buffer.append(chunk)
            self.silence_frames = 0
            return None

        if not self.is_recording:
            return None

        self.silence_frames += 1
        self.buffer.append(chunk)

        if self.silence_frames > self.max_silence:
            self.is_recording = False
            audio = np.concatenate(self.buffer)
            self.buffer = []
            return self._write_audio(audio)
        return None

    def _speech_probability(self, chunk: np.ndarray) -> float:
        if self.model is not None:
            try:
                import torch

                tensor = torch.from_numpy(chunk).float()
                return float(self.model(tensor, self.sample_rate).item())
            except Exception:
                pass
        amplitude = float(np.mean(np.abs(chunk)))
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        return max(amplitude, rms)

    def _write_audio(self, audio: np.ndarray) -> Path:
        from scipy.io.wavfile import write

        output_path = Path("temp_command.wav")
        write(output_path, self.sample_rate, (audio * 32767).astype(np.int16))
        return output_path
