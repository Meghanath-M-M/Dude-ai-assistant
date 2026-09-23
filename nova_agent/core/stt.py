import time
import warnings


class STTEngine:
    def __init__(self, model_name: str = "small", device: str = "cpu", compute_type: str = "int8"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "STT requires faster-whisper. Install requirements.txt first."
            ) from exc
        self._whisper_model = WhisperModel
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.last_duration: float = 0.0

    def warm_up(self) -> float:
        """Pay the one-off model/allocator cost so the first command is fast.

        faster-whisper is lazy: the first transcription of a session is several
        times slower than the rest. Feeding it a second of silence at startup
        moves that cost out of the user's first command.
        """
        import numpy as np

        silence = np.zeros(16_000, dtype=np.float32)
        started = time.perf_counter()
        try:
            self.model.transcribe(
                silence,
                language="en",
                beam_size=1,
                best_of=1,
                vad_filter=False,
                condition_on_previous_text=False,
            )
        except Exception as exc:  # noqa: BLE001 -- local runtime may differ
            warnings.warn(f"STT warm-up failed: {exc}", RuntimeWarning, stacklevel=2)
            return 0.0
        return time.perf_counter() - started

    def transcribe(
        self,
        audio_path: str,
        prompt: str | None = None,
        hotwords: str | None = None,
    ) -> str:
        """Transcribe a wav; ``prompt`` seeds Whisper's decoding context.

        Wake detection primes this with the wake phrase: short two-word
        utterances otherwise fall into Whisper's generic priors on some
        voices ("what are you doing?" for "hey dude").

        ``hotwords`` carries the command slot's *name* vocabulary. Whisper has
        no reason to think "notepad" or "vscode" is a word, so the names the
        user actually says go in as a decode hint — faster-whisper's own
        parameter, and the same technique Home Assistant's Whisper server uses
        for entity names (its author measured that even 50 unrelated names cost
        nothing on general speech). Left None for wake and confirmation
        captures, where biasing toward app names would work against the phrase
        being listened for.

        Long input is safe: faster-whisper truncates the hint to half the
        model's text context.
        """
        started = time.perf_counter()
        try:
            segments = self._collect_segments(audio_path, prompt, hotwords)
        except RuntimeError as exc:
            if self.device != "cuda" or "cublas" not in str(exc).lower():
                raise
            warnings.warn(
                "CUDA libraries are unavailable; falling back to CPU transcription.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.device = "cpu"
            self.compute_type = "int8"
            self.model = self._whisper_model(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            segments = self._collect_segments(audio_path, prompt, hotwords)
        self.last_duration = time.perf_counter() - started
        return " ".join(segment.text for segment in segments).strip().lower()

    def _collect_segments(
        self,
        audio_path: str,
        prompt: str | None = None,
        hotwords: str | None = None,
    ):
        segments, _ = self._run_transcription(audio_path, prompt, hotwords)
        return list(segments)

    def _run_transcription(
        self,
        audio_path: str,
        prompt: str | None = None,
        hotwords: str | None = None,
    ):
        return self.model.transcribe(
            audio_path,
            language="en",
            beam_size=1,
            best_of=1,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            hotwords=hotwords,
        )
