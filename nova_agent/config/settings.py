from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
MEMORY_DIR = PROJECT_ROOT / "memory"
CACHE_DIR = ASSETS_DIR / "tts_cache"
INTENTS_PATH = Path(__file__).with_name("intents.json")
DATABASE_PATH = MEMORY_DIR / "nova.db"
WAKE_WORD_MODEL_PATH = ASSETS_DIR / "wake_word" / "hey_nova.onnx"


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment flag such as ``NOVA_DRY_RUN=0``."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Dry run stays the default because launching windows and closing applications
# are the only irreversible things this assistant does.
DEFAULT_DRY_RUN = _env_flag("NOVA_DRY_RUN", True)
DEFAULT_STT_DEVICE = os.getenv("NOVA_STT_DEVICE", "cpu").strip() or "cpu"
DEFAULT_LLM_FALLBACK = _env_flag("NOVA_LLM_FALLBACK", False)


@dataclass(frozen=True)
class Settings:
    sample_rate: int = 16_000
    command_seconds: int = 3
    stt_model: str = "small"
    stt_device: str = DEFAULT_STT_DEVICE
    stt_compute_type: str = "int8"
    intent_threshold: float = 0.82
    wake_word: str = "hey nova"
    wake_threshold: float = 0.13
    tesseract_path: str | None = os.getenv("NOVA_TESSERACT_PATH")
    dry_run: bool = DEFAULT_DRY_RUN
    ocr_max_characters: int = 200
    llm_fallback: bool = DEFAULT_LLM_FALLBACK


def ensure_directories() -> None:
    for directory in (ASSETS_DIR, CACHE_DIR, MEMORY_DIR, ASSETS_DIR / "wake_word"):
        directory.mkdir(parents=True, exist_ok=True)
