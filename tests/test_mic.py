from pathlib import Path


def record_test(filename: str = "test.wav", duration: int = 3, sample_rate: int = 16_000) -> None:
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write
    except ImportError as exc:
        raise RuntimeError("Microphone test requires sounddevice and scipy.") from exc

    output = Path(filename)
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    write(output, sample_rate, audio)
    print(f"Saved recording to {output.resolve()}")


if __name__ == "__main__":
    record_test()
