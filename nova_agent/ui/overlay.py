"""On-screen status overlay for the assistant.

PyQt widgets must be touched from the thread that created them, and the
assistant calls ``set_state`` from the microphone worker thread. So the HUD
keeps plain data behind a lock and the window polls that snapshot on a timer:
no cross-thread widget access, and no risk of a UI call blocking audio.
"""

from __future__ import annotations

import threading
from typing import ClassVar

OVERLAY_SIZE = (380, 132)
POLL_INTERVAL_MS = 100


def qt_available() -> bool:
    """Whether a real overlay can be drawn here."""
    try:
        import PyQt5.QtWidgets  # noqa: F401
    except ImportError:
        return False
    return True


def _build_window_class():
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QBrush, QColor, QFont, QPainter
    from PyQt5.QtWidgets import QWidget

    class NovaOverlay(QWidget):
        """Frameless translucent panel showing state, status, and transcript."""

        def __init__(self, snapshot):
            super().__init__()
            self._snapshot = snapshot
            self._last = None
            self.setWindowFlags(
                Qt.FramelessWindowHint
                | Qt.WindowStaysOnTopHint
                | Qt.Tool
                | Qt.WindowDoesNotAcceptFocus
            )
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setAttribute(Qt.WA_ShowWithoutActivating)
            self.setFixedSize(*OVERLAY_SIZE)
            self._place_bottom_right()

            self._timer = QTimer(self)
            self._timer.timeout.connect(self._sync)
            self._timer.start(POLL_INTERVAL_MS)

        def _place_bottom_right(self) -> None:
            from PyQt5.QtWidgets import QApplication

            screen = QApplication.primaryScreen()
            if screen is None:
                return
            area = screen.availableGeometry()
            self.move(
                area.right() - self.width() - 24,
                area.bottom() - self.height() - 24,
            )

        def _sync(self) -> None:
            state = self._snapshot()
            key = (state["state"], state["status"], state["transcript"], state["color"])
            if key != self._last:
                self._last = key
                self.update()

        def paintEvent(self, _event):
            state = self._snapshot()
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            painter.setBrush(QBrush(QColor(18, 20, 24, 210)))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(self.rect(), 16, 16)

            accent = QColor(state["color"])
            painter.setBrush(QBrush(accent))
            painter.drawEllipse(20, 24, 26, 26)

            painter.setPen(accent)
            painter.setFont(QFont("Segoe UI", 11, QFont.DemiBold))
            painter.drawText(58, 44, f"DUDE  {state['state'].upper()}")

            painter.setPen(QColor(236, 238, 242))
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(20, 76, 340, 20, Qt.AlignLeft, state["status"][:48])

            painter.setPen(QColor(150, 156, 168))
            painter.setFont(QFont("Segoe UI", 9, QFont.StyleItalic))
            painter.drawText(20, 98, 340, 20, Qt.AlignLeft, state["transcript"][:52])

    return NovaOverlay
class NovaHUD:
    """Assistant status display with a console fallback.

    Attributes mirror the previous console-only contract (``current_state``,
    ``status_text``, ``color``) so callers and tests stay unchanged.
    """

    COLORS: ClassVar[dict[str, str]] = {
        "idle": "#333333",
        "listening": "#00d4ff",
        "thinking": "#ffaa00",
        "speaking": "#00ff88",
        "confirming": "#ff8800",
        "error": "#ff4444",
    }

    def __init__(self, enabled: bool = True, verbose: bool = True):
        self.current_state = "idle"
        self.status_text = "idle"
        self.color = self.COLORS["idle"]
        self.transcript = ""
        self.enabled = bool(enabled)
        self.verbose = verbose
        self.error: str | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._app = None
        self._window = None
        self.overlay_active = False

    def set_state(self, state: str, message: str = "") -> None:
        """Record a state change. Safe to call from any thread."""
        with self._lock:
            self.current_state = state
            self.status_text = message or state
            self.color = self.COLORS.get(state, "#ffffff")
        if self.verbose and not self.overlay_active:
            print(f"HUD [{state}] {self.status_text}".rstrip())

    def show_transcript(self, text: str) -> None:
        """Show what was heard; empty text clears the ribbon."""
        with self._lock:
            self.transcript = (text or "").strip()

    def snapshot(self) -> dict[str, str]:
        """Read the current display values (used by the overlay's timer)."""
        with self._lock:
            return {
                "state": self.current_state,
                "status": self.status_text,
                "transcript": self.transcript,
                "color": self.color,
            }

    def start(self) -> bool:
        """Open the overlay in a background thread. Returns whether it started."""
        if not self.enabled or not qt_available():
            if self.verbose:
                reason = "disabled" if not self.enabled else "PyQt5 unavailable"
                print(f"HUD overlay not shown ({reason}); using console output.")
            return False

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Wait briefly so a failed launch is reported at startup rather than
        # silently falling back in the middle of the first command.
        self._thread.join(timeout=2.0)
        return self.overlay_active

    def _run(self) -> None:
        try:
            from PyQt5.QtWidgets import QApplication

            window_class = _build_window_class()
            self._app = QApplication.instance() or QApplication([])
            self._window = window_class(self.snapshot)
            self._window.show()
            self.overlay_active = True
            self._app.exec_()
        except Exception as exc:  # noqa: BLE001 -- pragma: no cover, display-dependent
            self.error = f"{type(exc).__name__}: {exc}"
            self.enabled = False
            if self.verbose:
                print(f"HUD overlay failed ({self.error}); using console output.")
        finally:
            self.overlay_active = False

    def stop(self) -> None:
        """Close the overlay, if it is open."""
        if self._app is not None:
            try:
                self._app.quit()
            except Exception:  # noqa: BLE001, S110 -- pragma: no cover, shutdown best effort
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
