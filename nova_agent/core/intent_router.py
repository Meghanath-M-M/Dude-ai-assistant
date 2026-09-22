import json
import re
from pathlib import Path
from typing import Any


class IntentRouter:
    PROJECT_WORDS = {"project", "projects", "workspace", "folder", "repo", "repository"}
    VOLUME_WORDS = {"volume", "sound", "mute", "unmute", "louder", "quieter"}

    FILLER_WORDS = {
        "hello",
        "hi",
        "hey",
        "there",
        "please",
        "can",
        "you",
        "could",
        "would",
        "kindly",
        "the",
        "a",
        "an",
        "my",
        "me",
        "now",
        "please",
    }

    def __init__(
        self,
        threshold: float = 0.82,
        intents_path: Path | None = None,
        fallback: Any | None = None,
    ):
        self.threshold = threshold
        self.project_root = Path(__file__).resolve().parents[2]
        self.intents_path = (
            intents_path or self.project_root / "nova_agent" / "config" / "intents.json"
        )
        self.intents: dict[str, dict[str, Any]] = self._load_intents()
        self._encoder = None
        self._embedding_matrix = None
        self._labels: list[str] = []
        self.fallback = fallback

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
        self._labels = []
        for label, config in self.intents.items():
            for example in config["examples"]:
                self._labels.append(label)
                examples.append(example)
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._embedding_matrix = self._encoder.encode(examples, normalize_embeddings=True)

    def _canonicalize(self, text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        tokens = [token for token in cleaned.split() if token and token not in self.FILLER_WORDS]
        return " ".join(tokens)

    def _lexical_fallback(self, text: str) -> tuple[dict[str, Any] | None, float]:
        normalized = self._canonicalize(text)
        if not normalized:
            return None, 0.0

        if normalized in {"open chrome", "chrome", "browser", "launch browser", "start browser"}:
            return self.intents["open_chrome"], 0.75

        if (
            normalized.startswith("search ")
            or " search " in normalized
            or normalized.endswith("search")
        ):
            return self.intents["search_web"], 0.75

        if normalized in {"read screen", "screen", "what this say", "read page"}:
            return self.intents["read_screen"], 0.75

        # "open my ml project" barely differs from the examples yet can land
        # under the embedding threshold, so claim the obvious shapes outright.
        tokens = normalized.split()
        if tokens and tokens[0] in {"open", "launch", "show", "start"}:
            if any(word in tokens for word in self.PROJECT_WORDS):
                return self.intents["open_project"], 0.75

        if any(word in tokens for word in self.VOLUME_WORDS):
            return self.intents["system_volume"], 0.75

        for _, config in self.intents.items():
            for example in config.get("examples", []):
                example_text = self._canonicalize(example)
                if not example_text:
                    continue
                if (
                    normalized == example_text
                    or example_text in normalized
                    or normalized in example_text
                ):
                    return config, 0.75
        return None, 0.0

    def match(self, text: str) -> tuple[dict[str, Any] | None, float]:
        if not text.strip():
            return None, 0.0
        self._prepare_embeddings()
        vector = self._encoder.encode([text], normalize_embeddings=True)[0]
        similarities = self._embedding_matrix @ vector
        best_index = int(similarities.argmax())
        score = float(similarities[best_index])
        intent = self.intents[self._labels[best_index]]
        if score >= self.threshold:
            return intent, score

        lexical_intent, lexical_score = self._lexical_fallback(text)
        if lexical_intent is not None:
            return lexical_intent, lexical_score

        if self.fallback is not None:
            fallback_intent = self.fallback.classify(text)
            # "unknown" must stay a miss, otherwise the dispatcher reports
            # "The unknown action is not enabled yet" instead of asking again.
            if fallback_intent and fallback_intent.get("action") not in (None, "unknown"):
                return fallback_intent, float(fallback_intent.get("confidence", 0.75))
        return None, score
