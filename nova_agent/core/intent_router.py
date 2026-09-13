import json
from pathlib import Path
from typing import Any


class IntentRouter:
    def __init__(self, threshold: float = 0.82, intents_path: Path | None = None):
        self.threshold = threshold
        self.project_root = Path(__file__).resolve().parents[2]
        self.intents_path = (
            intents_path or self.project_root / "nova_agent" / "config" / "intents.json"
        )
        self.intents: dict[str, dict[str, Any]] = self._load_intents()
        self._encoder = None
        self._embedding_matrix = None
        self._labels: list[str] = []

    def _load_intents(self) -> dict[str, dict[str, Any]]:
        with self.intents_path.open(encoding="utf-8") as file:
            return json.load(file)

    def _prepare_embeddings(self) -> None:
        if self._embedding_matrix is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Intent embeddings require sentence-transformers. Install requirements.txt first."
            ) from exc
        examples = []
        for label, config in self.intents.items():
            for example in config["examples"]:
                self._labels.append(label)
                examples.append(example)
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._embedding_matrix = self._encoder.encode(examples, normalize_embeddings=True)

    def match(self, text: str) -> tuple[dict[str, Any] | None, float]:
        if not text.strip():
            return None, 0.0
        self._prepare_embeddings()
        vector = self._encoder.encode([text], normalize_embeddings=True)[0]
        similarities = self._embedding_matrix @ vector
        best_index = int(similarities.argmax())
        score = float(similarities[best_index])
        intent = self.intents[self._labels[best_index]]
        return (intent, score) if score >= self.threshold else (None, score)
