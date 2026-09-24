"""Speaker verification for the wake path — only your voice wakes Dude.

Field round 8 follow-up: with the threshold and reverb-tail guards in place,
every remaining false wake was the *transcript* path matching a Whisper
prompt echo on noise — the wake phrase is primed as the decode hint, so
non-speech audio "transcribes" as it. Text cannot separate that from a real
utterance; a speaker embedding can: noise has no speaker, and other people
or media are not the enrolled voice.

Contract — degradation is never a dead loop:

- No voiceprint, ``NOVA_VOICE_GATE=0``, no backend, or unusable audio →
  the check returns ``None`` (open) and the wake proceeds on the phrase
  alone, exactly as before.
- Only a *confident mismatch* returns ``False`` and suppresses the wake.

Enrollment is explicit: ``python -m nova_agent --enroll-voice`` records three
short clips, carves them into probe-length (1.5s) windows, and stores every
window's L2-normalized ECAPA embedding as a row of ``VOICEPRINT_PATH``
(git-ignored, like the models) — verification scores the probe against the
**nearest** reference and only vetoes a *confident* mismatch (threshold
0.4: different-speaker ECAPA scores concentrate below ~0.3). The speechbrain
model downloads once into ``assets/speaker_model`` and runs offline after.
Embeddings run on CPU: the segment is short, and the GPU belongs to Whisper.
Field round 10: silence-padded probes measured against one long-utterance
centroid were vetoing the enrolled voice itself — hence the trimming, the
window references, and the score printed in every wake line.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from nova_agent.config.settings import ASSETS_DIR, VOICEPRINT_PATH, Settings

# speechbrain ECAPA trained on VoxCeleb: <100 MB, downloaded once to
# assets/speaker_model on first use (the Kokoro precedent).
SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

# Below this there is no voice to compare — and no real wake phrase either
# (a full "hey dude" needs ~0.6s of audio): on a *phrase-matched* segment
# this is a mismatch verdict, because prompt echoes love short noise.
MIN_EMBEDDING_SECONDS = 0.25


def trim_silence(samples: np.ndarray, margin_samples: int = 1600) -> np.ndarray:
    """Drop leading/trailing quiet so an enrollment clip is mostly voice."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return samples
    peak = float(np.max(np.abs(samples)))
    if peak <= 0.0:
        return samples
    voiced = np.where(np.abs(samples) >= peak * 0.1)[0]
    if voiced.size == 0:
        return samples
    start = max(0, int(voiced[0]) - margin_samples)
    stop = min(int(samples.size), int(voiced[-1]) + 1 + margin_samples)
    return samples[start:stop]


def carve_windows(samples: np.ndarray, sample_rate: int, window_seconds: float = 1.5):
    """Split a clip into probe-length windows (the length a wake really is).

    Enrollment clips are seconds of continuous speech; a real wake probe is
    ~0.7–1.5s of voice. Comparing like lengths keeps a short probe from being
    measured against a long-utterance centroid (field round 10: the enrolled
    voice itself was vetoed at threshold 0.5).
    """
    window = int(window_seconds * sample_rate)
    if samples.size <= window:
        return [samples] if samples.size else []
    return [
        samples[start : start + window]
        for start in range(0, samples.size - window + 1, window)
    ]


def _read_wav_mono(path: Path) -> np.ndarray | None:
    """Read a recorded segment as float32 mono, or None when unreadable."""
    try:
        _rate, data = wavfile.read(path)
    except Exception:  # noqa: BLE001 - a bad file must not kill the wake loop
        return None
    samples = np.asarray(data)
    if samples.ndim > 1:
        samples = samples[:, 0]
    if samples.size == 0:
        return None
    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float32) / float(np.iinfo(samples.dtype).max)
    else:
        samples = samples.astype(np.float32)
    return samples


class VoiceGate:
    """Verifies *who* is speaking at a wake candidate."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: Callable[[np.ndarray], np.ndarray] | None = None,
        voiceprint_path: Path | None = None,
    ):
        settings = settings or Settings()
        self.enabled = settings.voice_gate
        self.threshold = settings.voice_threshold
        self.sample_rate = settings.sample_rate
        self.voiceprint_path = Path(voiceprint_path or VOICEPRINT_PATH)
        # Tests inject a plain embedder; production loads speechbrain lazily.
        self._embedder = embedder
        self._model = None
        self._torch = None
        self._backend_failed = False
        self._references: list[np.ndarray] = []
        self.last_score: float | None = None
        if self.enabled and self.voiceprint_path.exists():
            try:
                stored = np.asarray(np.load(self.voiceprint_path), dtype=np.float32)
                rows = stored.reshape(1, -1) if stored.ndim == 1 else stored
                # Each row is an L2-normalized reference embedding (probe-length
                # windows); a single row is an old centroid voiceprint and
                # still works.
                self._references = []
                for row in rows:
                    norm = float(np.linalg.norm(row))
                    if norm:
                        self._references.append(row / norm)
            except Exception as exc:  # noqa: BLE001 - corrupt file = gate open
                warnings.warn(
                    f"voice gate: voiceprint unreadable ({exc}); waking on the phrase alone",
                    RuntimeWarning,
                    stacklevel=2,
                )

    @property
    def engaged(self) -> bool:
        """True only when a check will actually run on wake candidates."""
        return self.enabled and bool(self._references)

    def status(self) -> str:
        if not self.enabled:
            return "disabled (NOVA_VOICE_GATE=0)"
        if not self._references:
            return "off (phrase-only; run --enroll-voice to restrict to your voice)"
        return f"on (only your enrolled voice wakes Dude, threshold {self.threshold:g})"

    def verify_file(self, wav_path) -> bool | None:
        """Check the speaker in a phrase-matched wake segment.

        ``False`` for audio too short to hold a real wake phrase — a Whisper
        prompt echo on a noise click is exactly what fits there.
        """
        self.last_score = None
        if not self.engaged:
            return None
        samples = _read_wav_mono(Path(wav_path))
        if samples is None:
            return None
        # Judge the voice, not the VAD's silence padding: a probe that is half
        # trailing quiet drags the embedding toward nothing (field round 10:
        # enrolled self-scores of 0.81 were vetoing the enrolled speaker).
        samples = trim_silence(samples)
        if samples.size < int(MIN_EMBEDDING_SECONDS * self.sample_rate):
            # A full "hey dude" needs ~0.6s of *voice* — less than that is a
            # prompt echo, not a person.
            self.last_score = 0.0
            return False
        return self.verify_samples(samples)

    def verify_samples(self, samples) -> bool | None:
        """``False`` = confidently someone else; ``None`` = open (cannot judge)."""
        self.last_score = None
        if not self.engaged:
            return None
        samples = trim_silence(np.asarray(samples, dtype=np.float32))
        if samples.size < int(MIN_EMBEDDING_SECONDS * self.sample_rate):
            return None
        embedding = self.embed(samples)
        if embedding is None:
            return None
        # Nearest reference, not the session centroid: probe-length windows
        # match their own recording instead of averaging the session away.
        self.last_score = max(
            float(np.dot(embedding, reference)) for reference in self._references
        )
        return self.last_score >= self.threshold

    def embed(self, samples) -> np.ndarray | None:
        """L2-normalized speaker embedding; None when it cannot be read."""
        samples = np.asarray(samples, dtype=np.float32)
        try:
            if self._embedder is not None:
                raw = np.asarray(self._embedder(samples), dtype=np.float32)
            else:
                model = self.load_model()
                if model is None or self._torch is None:
                    return None
                wave = self._torch.from_numpy(samples).unsqueeze(0)
                with self._torch.no_grad():
                    raw = model.encode_batch(wave).squeeze().cpu().numpy().astype(np.float32)
        except Exception as exc:  # noqa: BLE001 - degrade open, never kill the loop
            warnings.warn(f"voice gate: embedding failed ({exc})", RuntimeWarning, stacklevel=2)
            return None
        norm = float(np.linalg.norm(raw))
        if norm == 0.0:
            return None
        return raw / norm

    def load_model(self):
        """Lazily load the ECAPA model; None after any failure (gate opens)."""
        if self._model is not None:
            return self._model
        if self._backend_failed:
            return None
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            self._backend_failed = True
            warnings.warn(
                f"voice gate: speechbrain not installed ({exc}); waking on the phrase alone",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        try:
            self._model = EncoderClassifier.from_hparams(
                source=SPEAKER_MODEL_SOURCE,
                savedir=str(ASSETS_DIR / "speaker_model"),
                run_opts={"device": "cpu"},
            )
        except Exception as exc:  # noqa: BLE001 - download/load failure = open
            self._backend_failed = True
            warnings.warn(
                f"voice gate: speaker model unavailable ({exc}); waking on the phrase alone",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        self._torch = torch
        return self._model


def enroll_voice(settings: Settings | None = None, record_seconds: float = 5.0) -> bool:
    """Record a short session and store the mean voice embedding (interactive)."""
    settings = settings or Settings()
    gate = VoiceGate(settings)
    if gate.load_model() is None:
        print("Voice enrollment needs the speaker model:")
        print("  pip install speechbrain   (the ECAPA model downloads on first use)")
        return False
    import sounddevice as sd

    prompts = (
        "Keep talking for 5 seconds - tell me what you did today.",
        "Now say, at a natural pace: hey dude, hey dude, hey dude.",
        "One more, as you would really talk to me: hey dude, what's the time?",
    )
    print("Voice enrollment: three short clips, about 15 seconds total.")
    print("Record where you actually use Dude - the real background helps.")
    clips = []
    for number, prompt in enumerate(prompts, 1):
        print(f"[{number}/{len(prompts)}] {prompt}")
        audio = sd.rec(
            int(record_seconds * settings.sample_rate),
            samplerate=settings.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        clip = trim_silence(audio[:, 0])
        if clip.size < int(0.5 * settings.sample_rate):
            print("      mostly quiet - skipping that clip.")
            continue
        clips.append(clip)
        print("      recorded.")
    if not clips:
        print("Enrollment failed: no usable speech recorded. Try again.")
        return False
    # Probe-length references: carve each clip into ~1.5s windows (what a wake
    # segment actually looks like) and store every window as its own row —
    # verification then takes the *max* similarity over references instead of
    # measuring a short probe against one long-utterance centroid.
    windows = []
    for clip in clips:
        windows.extend(carve_windows(clip, settings.sample_rate))
    references = []
    for window in windows:
        embedding = gate.embed(window)
        if embedding is not None:
            references.append(embedding)
    if len(references) < 2:
        print("Enrollment failed: could not read your voice from those clips. Try again.")
        return False
    gate.voiceprint_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(gate.voiceprint_path, np.stack(references).astype(np.float32))
    # Leave-one-out: each window scored against the *other* windows — exactly
    # what a wake probe will be scored against at the threshold.
    similarities = [
        max(
            float(np.dot(embedding, references[j]))
            for j in range(len(references))
            if j != index
        )
        for index, embedding in enumerate(references)
    ]
    print(
        f"Voiceprint saved: {gate.voiceprint_path} "
        f"({len(references)} reference windows)"
    )
    print(
        "Self-check (how a short wake clip would score): "
        f"min {min(similarities):.2f}, avg {sum(similarities) / len(similarities):.2f}"
    )
    if min(similarities) < settings.voice_threshold + 0.05:
        print(
            f"Note: lowest self-score is near the {settings.voice_threshold:g} "
            "threshold - if wakes get rejected, re-record here or lower "
            "NOVA_VOICE_THRESHOLD."
        )
    print(f"Voice gate ON - wakes only for you (threshold {settings.voice_threshold:g}).")
    print("Disable any time with NOVA_VOICE_GATE=0.")
    return True
