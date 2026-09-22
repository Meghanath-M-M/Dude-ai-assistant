"""One-shot microphone recording for the ``--once`` command path."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def record_command(
    output_path: Path,
    duration: float = 3.0,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> Path:
    """Record ``duration`` seconds from the default microphone to ``output_path``.

    Returns the path to the written WAV file.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - depends on runtime environment.
        raise RuntimeError("Microphone recording requires sounddevice.") from exc

    from scipy.io.wavfile import write

    frames = int(duration * sample_rate)
    recording = sd.rec(frames, samplerate=sample_rate, channels=channels, dtype="float32")
    sd.wait()

    audio = np.asarray(recording[:, 0] if channels > 0 else recording, dtype=np.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, sample_rate, (audio * 32767).astype(np.int16))
    return output_path