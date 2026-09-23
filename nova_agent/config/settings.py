import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
MEMORY_DIR = PROJECT_ROOT / "memory"
CACHE_DIR = ASSETS_DIR / "tts_cache"
COMMAND_DIR = MEMORY_DIR / "commands"
INTENTS_PATH = Path(__file__).with_name("intents.json")
DATABASE_PATH = MEMORY_DIR / "nova.db"
WAKE_WORD_MODEL_PATH = ASSETS_DIR / "wake_word" / "hey_nova.onnx"
VAD_MODEL_PATH = ASSETS_DIR / "wake_word" / "silero_vad.onnx"


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment flag such as ``NOVA_DRY_RUN=0``."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    """Read a numeric environment variable, ignoring garbage values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


# Dry run stays the default: launching windows and closing applications are the
# only irreversible things this assistant does.
DEFAULT_DRY_RUN = _env_flag("NOVA_DRY_RUN", True)
DEFAULT_ALLOW_DESTRUCTIVE = _env_flag("NOVA_ALLOW_DESTRUCTIVE", False)
DEFAULT_STT_DEVICE = os.getenv("NOVA_STT_DEVICE", "cpu").strip() or "cpu"
DEFAULT_LLM_FALLBACK = _env_flag("NOVA_LLM_FALLBACK", False)
DEFAULT_LLM_MODEL = os.getenv("NOVA_LLM_MODEL", "phi4-mini").strip() or "phi4-mini"
DEFAULT_LLM_MIN_SCORE = _env_float("NOVA_LLM_MIN_SCORE", 0.65)
# Transcript wake (matching the spoken wake phrase in idle speech) is the
# primary trigger; NOVA_TRANSCRIPT_WAKE=0 keeps only the ONNX sound model.
DEFAULT_TRANSCRIPT_WAKE = _env_flag("NOVA_TRANSCRIPT_WAKE", True)


@dataclass(frozen=True)
class Settings:
    sample_rate: int = 16_000
    sample_block: int = 1_280
    command_seconds: int = 3
    stt_model: str = "small"
    stt_device: str = DEFAULT_STT_DEVICE
    stt_compute_type: str = "int8"
    stt_warmup: bool = True
    # sentence-transformers and Kokoro load lazily; at listen startup both are
    # warmed on a background thread so boot reaches "listening" immediately
    # while the ~21s-per-model load tax stays off command #1.
    intent_warmup: bool = True
    tts_warmup: bool = True
    intent_threshold: float = 0.82
    # The phrase the assistant answers to. The ONNX sound model is trained
    # for a fixed phrase of its own; this is what transcript wake matches on.
    wake_word: str = "hey dude"
    wake_threshold: float | None = None
    wake_sustain_window: int = 3
    wake_strong_margin: float = 0.2
    transcript_wake: bool = DEFAULT_TRANSCRIPT_WAKE
    # Idle segments quieter than this never reach STT: Whisper can echo the
    # wake prompt on near-silence, and ambient noise should not cost a decode.
    # Tuned against field data: ambient ~0.0011, quietest verified speech
    # attempt 0.0020 — the floor sits between them so soft wakes get through.
    wake_segment_min_rms: float = 0.0015
    # After a wake, a missed capture (empty transcript or nothing above the
    # confidence band — e.g. a video winning the first segment) keeps the
    # command turn open until this many seconds have passed since the wake,
    # so the real command can still arrive.
    command_window: float = 8.0
    # Absolute cap on one command turn: every miss re-arms command_window
    # (so the pipeline's own latency can't silently consume it), but the turn
    # as a whole dies by wake_time + this, so background audio can't hold it.
    command_max_turn: float = 30.0
    tesseract_path: str | None = os.getenv("NOVA_TESSERACT_PATH")
    dry_run: bool = DEFAULT_DRY_RUN
    allow_destructive: bool = DEFAULT_ALLOW_DESTRUCTIVE
    llm_fallback: bool = DEFAULT_LLM_FALLBACK
    llm_timeout: float = 3.0
    llm_model: str = DEFAULT_LLM_MODEL
    # Tier 2 (the LLM) may only answer inside this confidence band: below the
    # floor the honest reply is "I didn't catch that", never a guess.
    llm_min_score: float = DEFAULT_LLM_MIN_SCORE
    ocr_max_characters: int = 200
    vad_model_path: str | None = str(VAD_MODEL_PATH)
    vad_speech_threshold: float = 0.5
    vad_silence_frames: int = 9
    # Chunks are sample_block (1280 samples = 80 ms at 16 kHz), so 4 frames of
    # voice ≈ 320 ms — the >= 250 ms of sustained speech a real command needs.
    vad_min_speech_frames: int = 4
    # Segments (idle transcript probes and commands alike) are cut off here so
    # speech without a pause cannot buffer for minutes.
    vad_max_seconds: float = 30.0
    tts_voice: str = "af_heart"
    tts_max_cache_words: int = 6
    tts_max_cache_characters: int = 60
    confirmation_timeout: float = 5.0
    hud_enabled: bool = True


def ensure_directories() -> None:
    for directory in (
        ASSETS_DIR,
        CACHE_DIR,
        MEMORY_DIR,
        COMMAND_DIR,
        ASSETS_DIR / "wake_word",
    ):
        directory.mkdir(parents=True, exist_ok=True)
