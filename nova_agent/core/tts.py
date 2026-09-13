from pathlib import Path


class TTSEngine:
    def __init__(self, cache_dir: Path | None = None, voice: str = "af_heart"):
        self.cache_dir = cache_dir or Path("assets/tts_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.voice = voice
        self._pipeline = None

    def speak(self, text: str) -> None:
        """Play cached Kokoro audio, or generate it on demand when installed."""
        try:
            import sounddevice as sd
            from scipy.io import wavfile
        except ImportError as exc:
            raise RuntimeError("TTS playback requires sounddevice and scipy.") from exc

        cache_path = self.cache_dir / f"{text.strip().lower().replace(' ', '_')}.wav"
        if cache_path.exists():
            sample_rate, audio = wavfile.read(cache_path)
            sd.play(audio, sample_rate)
            sd.wait()
            return

        if self._pipeline is None:
            try:
                from kokoro import KPipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Live TTS requires kokoro. Install requirements.txt first."
                ) from exc
            self._pipeline = KPipeline(lang_code="a")

        from scipy.io import wavfile

        for _, _, audio in self._pipeline(text, voice=self.voice):
            wavfile.write(cache_path, 24_000, audio)
            sd.play(audio, 24_000)
            sd.wait()
            return
