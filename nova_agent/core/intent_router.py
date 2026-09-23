import json
import re
import threading
import time
from pathlib import Path
from typing import Any, ClassVar


class IntentRouter:
    PROJECT_WORDS: ClassVar[set[str]] = {
        "project", "projects", "workspace", "folder", "repo", "repository"
    }
    VOLUME_WORDS: ClassVar[set[str]] = {"volume", "sound", "mute", "unmute", "louder", "quieter"}

    # Verbs that introduce an app ("open notepad"). Spoken on their own they
    # name no target at all, which is the dispatcher's cue to ask "Open what?"
    # instead of launching a default app.
    LAUNCH_WORDS: ClassVar[set[str]] = {"open", "launch", "start", "run", "show"}

    # A time word is enough to mean a time question: field transcripts strip
    # the question around it ("what is the time" -> "so run the time").
    TIME_WORDS: ClassVar[set[str]] = {"time", "clock"}

    FILLER_WORDS: ClassVar[set[str]] = {
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
    }

    def __init__(
        self,
        threshold: float = 0.82,
        intents_path: Path | None = None,
        fallback: Any | None = None,
        fallback_floor: float = 0.65,
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
        # The background warm-up thread and the first live match can race into
        # the lazy load; without this, both would build a second encoder.
        self._embed_lock = threading.Lock()
        self.fallback = fallback
        # Below this score Tier 1 wasn't even close: answering with the LLM
        # would be guessing, so the dispatcher re-asks instead.
        self.fallback_floor = fallback_floor
        # Runner-up of the last match: a miss prints the label it *almost*
        # matched ("Intent: none (best score 0.51 for time_check, ...)"), which
        # is the difference between a diagnosable miss and "I didn't catch that".
        self.last_best_label: str | None = None
        self.last_best_score: float = 0.0

    def _load_intents(self) -> dict[str, dict[str, Any]]:
        with self.intents_path.open(encoding="utf-8") as file:
            return json.load(file)

    def _prepare_embeddings(self) -> None:
        if self._embedding_matrix is not None:
            return
        with self._embed_lock:
            if self._embedding_matrix is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Intent embeddings require sentence-transformers. "
                    "Install requirements.txt first."
                ) from exc
            examples = []
            self._labels = []
            for label, config in self.intents.items():
                for example in config["examples"]:
                    self._labels.append(label)
                    examples.append(example)
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            self._embedding_matrix = self._encoder.encode(
                examples, normalize_embeddings=True
            )

    def warm_up(self) -> float:
        """Load the embedding model eagerly and return the seconds spent.

        ``_prepare_embeddings`` is lazy, so without this call the first
        *command* pays for ``import sentence_transformers`` (torch included)
        plus the MiniLM load — measured live at 9.7s inside the intent stage,
        long enough to outlive the post-wake retry window.
        """
        started = time.perf_counter()
        self._prepare_embeddings()
        self._encoder.encode(["warm up the intent encoder"], normalize_embeddings=True)
        return time.perf_counter() - started

    def _canonicalize(self, text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        tokens = [token for token in cleaned.split() if token and token not in self.FILLER_WORDS]
        return " ".join(tokens)

    def _lexical_fallback(self, text: str) -> tuple[dict[str, Any] | None, float]:
        normalized = self._canonicalize(text)
        # A one-character fragment ("i" off a noisy capture) is a substring of
        # almost every example — a field run claimed "launch visual studio code"
        # from a bare "i" and opened VS Code from noise. Refuse to claim on it.
        if len(normalized) < 2:
            return None, 0.0

        tokens = normalized.split()

        # A bare launch verb names nothing. The example loop below claimed it
        # anyway (its whole-word check matched "\bopen\b" *inside* the example
        # "open chrome"), so every mangle that collapsed to one verb launched
        # Chrome — field log: `Heard: open.` -> "Would open chrome". Return an
        # open_app with no target instead, and the dispatcher asks which app.
        if all(token in self.LAUNCH_WORDS for token in tokens):
            return {"action": "open_app", "safe": True}, 0.75

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
        if (
            tokens
            and tokens[0] in {"open", "launch", "show", "start"}
            and any(word in tokens for word in self.PROJECT_WORDS)
        ):
            return self.intents["open_project"], 0.75

        if any(word in tokens for word in self.VOLUME_WORDS):
            return self.intents["system_volume"], 0.75

        # Match whole words only: raw substring containment let short text
        # ("i") claim an example by matching a letter inside a longer word.
        for config in self.intents.values():
            for example in config.get("examples", []):
                example_text = self._canonicalize(example)
                if not example_text:
                    continue
                if (
                    normalized == example_text
                    or re.search(rf"\b{re.escape(example_text)}\b", normalized)
                    or re.search(rf"\b{re.escape(normalized)}\b", example_text)
                ):
                    return config, 0.75

        # A time word with no example-shaped phrase around it is a time
        # question ("so run the time"). This sits *after* the example loop on
        # purpose: a phrase that already is an example keeps its own intent,
        # so "it's time to open chrome" still opens Chrome.
        if any(word in tokens for word in self.TIME_WORDS):
            return self.intents["time_check"], 0.75

        # Any launch verb plus a named app: "open gallery". Project/volume/
        # chrome shapes were claimed above, so the remainder is an app phrase
        # the dispatcher must resolve (or honestly refuse), not a miss.
        if tokens and tokens[0] in {"open", "launch", "start", "run", "show"}:
            return {"action": "open_app", "safe": True}, 0.75
        return None, 0.0

    def match(self, text: str) -> tuple[dict[str, Any] | None, float]:
        if not text.strip():
            return None, 0.0
        self._prepare_embeddings()
        vector = self._encoder.encode([text], normalize_embeddings=True)[0]
        similarities = self._embedding_matrix @ vector
        best_index = int(similarities.argmax())
        score = float(similarities[best_index])
        self.last_best_label = self._labels[best_index]
        self.last_best_score = score
        intent = self.intents[self._labels[best_index]]
        if score >= self.threshold:
            return intent, score

        lexical_intent, lexical_score = self._lexical_fallback(text)
        if lexical_intent is not None:
            return lexical_intent, lexical_score

        if self.fallback is not None and score >= self.fallback_floor:
            fallback_intent = self.fallback.classify(text)
            # "unknown" must stay a miss, otherwise the dispatcher reports
            # "The unknown action is not enabled yet" instead of asking again.
            if fallback_intent and fallback_intent.get("action") not in (None, "unknown"):
                return fallback_intent, float(fallback_intent.get("confidence", 0.75))
        return None, score
