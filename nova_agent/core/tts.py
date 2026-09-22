from pathlib import Path
import os
import re
import tempfile


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

        cache_path = self._cache_path(text)
        if cache_path.exists():
            try:
                sample_rate, audio = wavfile.read(cache_path)
            except (OSError, ValueError):
                cache_path.unlink(missing_ok=True)
            else:
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
            audio = self._normalize_audio(audio)
            self._write_cache(wavfile, cache_path, audio)
            sd.play(audio, 24_000)
            sd.wait()
            return

        raise RuntimeError("Kokoro returned no audio for the requested text.")

    def _cache_path(self, text: str) -> Path:
        filename = re.sub(r"[^a-z0-9._-]+", "_", text.strip().lower())
        filename = filename.strip(" ._") or "empty"
        return self.cache_dir / f"{filename}.wav"

    @staticmethod
    def _normalize_audio(audio):
        import numpy as np

        if isinstance(audio, tuple):
            audio = audio[0]
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32)
        return np.clip(audio, -1.0, 1.0)

    @staticmethod
    def _write_cache(wavfile, cache_path: Path, audio) -> None:
        """Write to a temporary WAV before replacing the cache entry."""
        file_descriptor, temporary_path = tempfile.mkstemp(
            suffix=".wav",
            dir=cache_path.parent,
        )
        os.close(file_descriptor)
        temporary_file = Path(temporary_path)
        try:
            wavfile.write(temporary_file, 24_000, audio)
            os.replace(temporary_file, cache_path)
        finally:
            temporary_file.unlink(missing_ok=True)
