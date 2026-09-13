from pathlib import Path


def record_command(
    output_path: str | Path,
    duration: int = 3,
    sample_rate: int = 16_000,
) -> Path:
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write
    except ImportError as exc:
        raise RuntimeError("Recording requires sounddevice and scipy.") from exc

    output = Path(output_path)
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    write(output, sample_rate, audio)
    return output
