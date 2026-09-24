"""Decide when a person is talking over the assistant (barge-in).

While Kokoro speaks, the microphone hears the reply too. A fixed energy
threshold cannot tell the two apart: playback bleed depends on volume and
speaker placement, and it is *steady* while the assistant talks, whereas a
voice laid over it is a step change above that steady floor.

So the gate adapts. An exponential moving average tracks the echo/ambient
floor, updated only on frames that are not themselves loud — a person speaking
therefore cannot ratchet the baseline up against themselves. A frame counts as
barge-in once it exceeds ``max(floor, baseline * multiplier)`` for at least
``hold_seconds``.

This is deliberately an energy gate rather than the Silero VAD: Silero
classifies the assistant's own echo as speech (it *is* speech), so a VAD
probability threshold would cut us off on our own voice, and sharing VAD state
with command capture would corrupt segment assembly. The energy floor plus the
adaptive baseline is what separates "our reply bleeding back in" from "a person
raising their voice over it".
"""

import numpy as np


class BargeGate:
    """An echo-adaptive energy gate: feed frames, it reports when to stop."""

    def __init__(
        self,
        *,
        floor: float = 0.05,
        multiplier: float = 3.0,
        hold_seconds: float = 0.3,
        baseline_alpha: float = 0.1,
        sample_rate: int = 16_000,
    ):
        self.floor = floor
        self.multiplier = multiplier
        self.hold_seconds = hold_seconds
        self.baseline_alpha = baseline_alpha
        self.sample_rate = sample_rate
        self.reset()

    def reset(self) -> None:
        """Forget the last turn's floor and loud-run (start of playback)."""
        self.baseline = 0.0
        self._loud_seconds = 0.0
        self._primed = False
        self.tripped = False

    @staticmethod
    def rms(frame) -> float:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(arr))))

    @property
    def threshold(self) -> float:
        return max(self.floor, self.baseline * self.multiplier)

    def feed(self, frame, dt: float | None = None) -> bool:
        """Offer one mic frame; True once barge-in is confirmed (latched).

        ``dt`` defaults to the frame's duration at ``sample_rate``: how long
        the voice stayed above the threshold, which is what ``hold_seconds``
        is measured against.
        """
        if self.tripped:
            return True
        arr = np.asarray(frame, dtype=np.float32)
        energy = self.rms(arr)
        if dt is None:
            dt = arr.size / self.sample_rate if arr.size else 0.0

        if not self._primed:
            # First frame after reset(): adopt it as the floor. barge_begin()
            # flushed the pre-playback backlog, so this frame is playback
            # bleed — calibrating to our own echo, not to silence (which would
            # make the very first echo syllable look like a shout and trip us).
            self.baseline = energy
            self._primed = True

        if energy > self.threshold:
            self._loud_seconds += dt
            if self._loud_seconds >= self.hold_seconds:
                self.tripped = True
                return True
        else:
            # At/below the threshold this is the floor (echo or ambient):
            # track it, and treat any run of "loud" as ended.
            self.baseline += self.baseline_alpha * (energy - self.baseline)
            self._loud_seconds = 0.0
        return False
