from __future__ import annotations

from collections import deque
from typing import Any, Callable


class CommandQueue:
    """Simple in-memory FIFO queue for sequential command execution."""

    def __init__(self):
        self._queue: deque[tuple[Callable[..., Any], tuple[Any, ...]]] = deque()

    def enqueue(self, callback: Callable[..., Any], *args: Any) -> None:
        self._queue.append((callback, args))

    def flush(self) -> list[Any]:
        results: list[Any] = []
        while self._queue:
            callback, args = self._queue.popleft()
            results.append(callback(*args))
        return results
