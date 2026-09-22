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

    def transcribe(self, audio_path: str) -> str:
        try:
            segments = self._collect_segments(audio_path)
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
            segments = self._collect_segments(audio_path)
        return " ".join(segment.text for segment in segments).strip().lower()

    def _collect_segments(self, audio_path: str):
        segments, _ = self._run_transcription(audio_path)
        return list(segments)

    def _run_transcription(self, audio_path: str):
        return self.model.transcribe(
            audio_path,
            language="en",
            beam_size=1,
            best_of=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
