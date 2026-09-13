class STTEngine:
    def __init__(self, model_name: str = "small", device: str = "cuda", compute_type: str = "int8"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "STT requires faster-whisper. Install requirements.txt first."
            ) from exc
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(
            audio_path,
            language="en",
            beam_size=1,
            best_of=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text for segment in segments).strip().lower()
