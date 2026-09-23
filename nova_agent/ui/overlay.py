"""On-screen status overlay for the assistant.

PyQt widgets must be touched from the thread that created them, and the
assistant calls ``set_state`` from the microphone worker thread. So the HUD
keeps plain data behind a lock and the window polls that snapshot on a timer:
no cross-thread widget access, and no risk of a UI call blocking audio.
"""

from __future__ import annotations

import sys
import threading
from typing import ClassVar

OVERLAY_SIZE = (380, 132)
POLL_INTERVAL_MS = 100

# Qt 5 says this once when a QApplication is built off the main thread. The
# overlay *has* to live on its own thread here — the main thread is parked in
# the sounddevice callback loop — so the note is not actionable, and it reads
# like a fault next to a command log. Everything else Qt says still gets
# through; only this exact message is dropped.
BENIGN_QT_MESSAGES = ("QApplication was not created in the main() thread",)


def _silence_benign_qt_messages() -> None:
    """Install a Qt message handler that drops only the known-benign note."""
    try:
        from PyQt5.QtCore import qInstallMessageHandler
    except ImportError:  # pragma: no cover - qt_available() already checked
        return

    def handler(_mode, _context, message):
        if any(benign in message for benign in BENIGN_QT_MESSAGES):
            return
        print(message, file=sys.stderr)

    qInstallMessageHandler(handler)


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
        # Quit is *requested* by another thread and performed by the Qt thread
        # (see _quit_if_requested): calling quit() straight from the caller made
        # Qt tear down its timers cross-thread on every shutdown
        # ("QObject::killTimer: Timers cannot be stopped from another thread").
        self._stop_requested = threading.Event()
        self._quit_timer = None
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

        self._stop_requested.clear()  # a HUD may be started again after a stop
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Wait briefly so a failed launch is reported at startup rather than
        # silently falling back in the middle of the first command.
        self._thread.join(timeout=2.0)
        return self.overlay_active

    def _run(self) -> None:
        try:
            from PyQt5.QtCore import QTimer
            from PyQt5.QtWidgets import QApplication

            _silence_benign_qt_messages()
            window_class = _build_window_class()
            self._app = QApplication.instance() or QApplication([])
            self._window = window_class(self.snapshot)
            self._window.show()
            # Poll for a stop request *on this thread*, which owns the event
            # loop and therefore its timers.
            self._quit_timer = QTimer()
            self._quit_timer.timeout.connect(self._quit_if_requested)
            self._quit_timer.start(POLL_INTERVAL_MS)
            self.overlay_active = True
            self._app.exec_()
        except Exception as exc:  # noqa: BLE001 -- pragma: no cover, display-dependent
            self.error = f"{type(exc).__name__}: {exc}"
            self.enabled = False
            if self.verbose:
                print(f"HUD overlay failed ({self.error}); using console output.")
        finally:
            # Tear the Qt objects down *on this thread*. The event loop and its
            # timers belong to it; letting Python's GC drop them from the main
            # thread is what printed "QObject::killTimer: Timers cannot be
            # stopped from another thread" / "QObject::~QObject: ..." on exit.
            if self._quit_timer is not None:
                self._quit_timer.stop()
                self._quit_timer = None
            window, self._window = self._window, None
            if window is not None:
                window.hide()
                window.close()
                del window
            self._app = None  # the QApplication dies where it was created
            self.overlay_active = False

    def _quit_if_requested(self) -> None:
        """Quit the event loop when another thread asked for it (Qt thread only)."""
        if self._stop_requested.is_set() and self._app is not None:
            self._app.quit()

    def stop(self) -> None:
        """Close the overlay, if it is open.

        The request is handed to the overlay's own thread and this call waits
        briefly for it; that is what keeps Qt from tearing down its timers from
        a foreign thread on exit.
        """
        self._stop_requested.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.overlay_active and self._app is not None:  # pragma: no cover
            try:  # last-resort quit if the poll timer never got a chance to run
                self._app.quit()
            except Exception:  # noqa: BLE001, S110 -- shutdown best effort
                pass
