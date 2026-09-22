from __future__ import annotations


class AudioSession:
    """Tracks microphone lifecycle states for the local assistant."""

    def __init__(self):
        self.state = "idle"
        self.events: list[str] = []

    def start_listening(self) -> None:
        self.state = "listening"
        self.events.append(self.state)

    def start_capture(self) -> None:
        self.state = "capturing"
        self.events.append(self.state)

    def finish_capture(self) -> None:
        self.state = "processing"
        self.events.append(self.state)

    def reset(self) -> None:
        self.state = "idle"
        self.events.append(self.state)
