from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
MEMORY_DIR = PROJECT_ROOT / "memory"
CACHE_DIR = ASSETS_DIR / "tts_cache"
INTENTS_PATH = Path(__file__).with_name("intents.json")
DATABASE_PATH = MEMORY_DIR / "nova.db"


@dataclass(frozen=True)
class Settings:
    sample_rate: int = 16_000
    command_seconds: int = 3
    stt_model: str = "small"
    stt_device: str = "cuda"
    stt_compute_type: str = "int8"
    intent_threshold: float = 0.82
    tesseract_path: str | None = os.getenv("NOVA_TESSERACT_PATH")


def ensure_directories() -> None:
    for directory in (ASSETS_DIR, CACHE_DIR, MEMORY_DIR, ASSETS_DIR / "wake_word"):
        directory.mkdir(parents=True, exist_ok=True)
