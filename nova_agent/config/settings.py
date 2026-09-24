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
# Enrolled speaker vector for the wake voice gate (core/voice_gate.py).
VOICEPRINT_PATH = ASSETS_DIR / "voiceprint.npy"


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
# cuda by default: STTEngine falls back to CPU at load- and decode-time when
# the CUDA libs are missing, so machines without a GPU still work — but a
# launch without start.bat no longer silently runs whisper-small on CPU.
DEFAULT_STT_DEVICE = os.getenv("NOVA_STT_DEVICE", "cuda").strip() or "cuda"
DEFAULT_LLM_FALLBACK = _env_flag("NOVA_LLM_FALLBACK", False)
DEFAULT_LLM_MODEL = os.getenv("NOVA_LLM_MODEL", "phi4-mini").strip() or "phi4-mini"
DEFAULT_LLM_MIN_SCORE = _env_float("NOVA_LLM_MIN_SCORE", 0.65)
# Wave 2: the tool-calling brain. Off by default (Tier 1's measured accuracy is
# the baseline, and the brain needs a running Ollama with a tool-capable model).
DEFAULT_LLM_BRAIN = _env_flag("NOVA_LLM_BRAIN", False)
# Transcript wake (matching the spoken wake phrase in idle speech) is the
# primary trigger; NOVA_TRANSCRIPT_WAKE=0 keeps only the ONNX sound model.
DEFAULT_TRANSCRIPT_WAKE = _env_flag("NOVA_TRANSCRIPT_WAKE", True)
# Field round 8 (false wakes): after every reply the mic is deaf this long so
# the room's reverb tail — our own playback still decaying — cannot reach the
# wake detectors; and idle transcript probes close this early so background
# chatter cannot buffer a 30s segment into a multi-second transcription.
DEFAULT_WAKE_COOLDOWN = _env_float("NOVA_WAKE_COOLDOWN", 0.5)
DEFAULT_IDLE_SEGMENT_MAX = _env_float("NOVA_IDLE_SEGMENT_MAX", 10.0)
# Voice gate (core/voice_gate.py): restrict wake to the enrolled speaker.
# NOVA_VOICE_GATE=0 keeps phrase-only wake; the threshold is the cosine
# similarity below which a wake candidate is confidently "someone else" —
# ECAPA different-speaker scores concentrate below ~0.3, and 0.4 leaves
# headroom for short wake probes ("hey dude" is ~0.7s of voice).
DEFAULT_VOICE_GATE = _env_flag("NOVA_VOICE_GATE", True)
DEFAULT_VOICE_THRESHOLD = _env_float("NOVA_VOICE_THRESHOLD", 0.4)


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
    # Reverb-tail guard: _on_audio drops frames for this long after every
    # reply (set via _finish_speaking) — the room is still ringing with us.
    wake_cooldown: float = DEFAULT_WAKE_COOLDOWN
    # Idle transcript probes close here so background chatter cannot buffer
    # toward vad_max_seconds and cost seconds of STT (field round 8: an 11.5s
    # transcription from one long noise segment). Command captures keep the
    # full vad_max_seconds budget; _begin_listening restores it.
    idle_segment_max: float = DEFAULT_IDLE_SEGMENT_MAX
    # Voice gate (core/voice_gate.py): False keeps phrase-only wake; the
    # threshold is the cosine similarity below which a wake candidate is
    # confidently someone else (noise has no speaker; other voices aren't you).
    voice_gate: bool = DEFAULT_VOICE_GATE
    voice_threshold: float = DEFAULT_VOICE_THRESHOLD
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
    # Wave 2: the tool-calling brain (core/llm_brain.py). When on it supersedes
    # the Tier 2 string-parsing fallback, which build_router skips.
    llm_brain: bool = DEFAULT_LLM_BRAIN
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
