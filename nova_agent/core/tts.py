import os
import re
import tempfile
import threading
import time
from collections.abc import Iterable
from pathlib import Path


class TTSEngine:
    """Kokoro playback with a phrase cache and cached-prefix composition.

    Synthesis is the slow part, so anything said often is cached. Dynamic
    replies ("The current time is 07:32 PM") are never cached -- they would grow
    without bound and never be reused -- but they can still reuse a cached
    prefix clip and synthesise only the tail.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        voice: str = "af_heart",
        max_cache_words: int = 6,
        max_cache_characters: int = 60,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path("assets/tts_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.voice = voice
        self.max_cache_words = max_cache_words
        self.max_cache_characters = max_cache_characters
        self._pipeline = None
        self._pipeline_lock = threading.Lock()
        # Self-reported time-to-speech for the stats stage: set per synthesis,
        # reset by every speak() so a cache hit honestly reports ~0 (audio
        # starts immediately) instead of a stale value.
        self.last_synthesis_seconds: float = 0.0
        # How many canonical replies the last preload had to synthesise.
        self.last_preload_count: int = 0

    def warm_up(self, phrases: Iterable[str] = ()) -> float:
        """Build the pipeline, pay the first-synthesis tax, cache ``phrases``.

        The very first synthesis imports kokoro, loads the model, and runs
        torch's JIT pass — measured live at 21.5s inside the tts stage of the
        first command (warm syntheses land near 2.1s, plus playback).

        ``phrases`` are the assistant's canonical replies (see ``preload``);
        caching them here is what keeps their synthesis out of a user command.
        """
        started = time.perf_counter()
        self._synthesize("Warm up.")
        self.preload(phrases)
        return time.perf_counter() - started

    def preload(self, phrases: Iterable[str]) -> int:
        """Cache canonical reply phrases; returns how many were synthesised.

        ``should_cache``'s length policy exists because *dynamic* replies (a
        clock reading, "open <whatever>") would grow the cache without bound and
        are never said twice. The assistant's fixed replies are the opposite
        case, and the longest of them is the worst: the 15-word "screen reading
        is unavailable ..." reply cost 2669ms of synthesis against a 2500ms tts
        budget — on *every* use, because the policy refused to cache it. Storing
        canonical replies whatever their length turns that into a file read, and
        ``speak`` consults the cache before the policy, so they stay served.

        Entries already on disk are skipped on purpose: a second boot must not
        re-synthesise the whole phrasebook.
        """
        from scipy.io import wavfile

        created = 0
        for phrase in dict.fromkeys(phrases):  # de-duplicate, keep the order
            text = (phrase or "").strip()
            if not text:
                continue
            path = self._cache_path(text)
            if path.exists():
                continue
            self._write_cache(wavfile, path, self._synthesize(text))
            created += 1
        self.last_preload_count = created
        return created

    def clear_cache(self) -> int:
        """Delete every cached clip; returns how many files went.

        Used when the voice changes: the cache is keyed by phrase, not by voice,
        so clips synthesised with the previous voice would otherwise keep
        playing — including the whole preloaded phrasebook.
        """
        removed = 0
        for path in self.cache_dir.glob("*.wav"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    def should_cache(self, text: str) -> bool:
        """Only short, stable phrases are worth keeping on disk."""
        stripped = text.strip()
        if not stripped or len(stripped) > self.max_cache_characters:
            return False
        if len(stripped.split()) > self.max_cache_words:
            return False
        return not any(character.isdigit() for character in stripped)

    def speak(self, text: str) -> None:
        """Play ``text`` using the cache, a cached prefix, or fresh synthesis."""
        self.last_synthesis_seconds = 0.0  # cache hits truly start instantly
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("TTS playback requires sounddevice.") from exc

        cached = self._read_wav(self._cache_path(text))
        if cached is not None:
            sd.play(cached[1], cached[0])
            sd.wait()
            return

        prefix_path, remainder = self._cached_prefix_split(text)
        if prefix_path is not None and remainder:
            prefix = self._read_wav(prefix_path)
            if prefix is not None:
                sd.play(prefix[1], prefix[0])
                sd.wait()
            # Only the tail is new, so the reply starts on the first syllable.
            self._synthesize_and_play(sd, remainder)
            return

        self._synthesize_and_play(sd, text, cache=self.should_cache(text))

    def _synthesize_and_play(self, sd, text: str, cache: bool = False) -> None:
        if not text.strip():
            return
        audio = self._synthesize(text)
        if cache:
            from scipy.io import wavfile

            self._write_cache(wavfile, self._cache_path(text), audio)
        sd.play(audio, 24_000)
        sd.wait()

    def _synthesize(self, text: str):
        # One lock around build *and* synthesis: the background warm thread
        # and the first live reply raced into two KPipeline builds (observed
        # as two repo_id warnings and double the load time).
        with self._pipeline_lock:
            started = time.perf_counter()
            if self._pipeline is None:
                try:
                    from kokoro import KPipeline
                except ImportError as exc:
                    raise RuntimeError(
                        "Live TTS requires kokoro. Install requirements.txt first."
                    ) from exc
                self._pipeline = KPipeline(lang_code="a")

            for _, _, audio in self._pipeline(text, voice=self.voice):
                normalized = self._normalize_audio(audio)
                if normalized.size:
                    self.last_synthesis_seconds = time.perf_counter() - started
                    return normalized
            raise RuntimeError("Kokoro returned no audio for the requested text.")

    def _read_wav(self, path: Path):
        """Read a cached WAV, discarding entries that are no longer readable."""
        if not path.exists():
            return None

        from scipy.io import wavfile

        try:
            return wavfile.read(path)
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            return None

    def _cached_prefix_split(self, text: str):
        """Split ``text`` into a cached leading phrase and the remaining tail."""
        words = text.split()
        longest = min(self.max_cache_words, max(len(words) - 1, 0))
        for count in range(longest, 0, -1):
            candidate = " ".join(words[:count])
            path = self._cache_path(candidate)
            if path.exists():
                return path, " ".join(words[count:])
        return None, text

    def prune_cache(self) -> list[str]:
        """Delete cache entries that can never be produced or reused.

        Two kinds of junk accumulate: names ``_cache_path`` could never generate
        (a stray apostrophe) and dynamic phrases carrying a timestamp or count,
        which the cache policy would not write again.
        """
        removed: list[str] = []
        for path in sorted(self.cache_dir.glob("*.wav")):
            stem = path.stem
            sanitized = re.sub(r"[^a-z0-9._-]+", "_", stem.lower()).strip(" ._")
            if sanitized != stem or any(character.isdigit() for character in stem):
                path.unlink(missing_ok=True)
                removed.append(path.name)
        return removed

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
