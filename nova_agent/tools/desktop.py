"""Destructive desktop actions.

Every function here is gated behind a spoken confirmation and remains
unregistered in the capability registry unless ``NOVA_ALLOW_DESTRUCTIVE=1``.
Files are never deleted outright: they go to the Recycle Bin so a misheard
command is recoverable.
"""

from __future__ import annotations

from pathlib import Path

from nova_agent.tools.guards import GuardViolation, ensure_inside

DOWNLOAD_DIR = Path.home() / "Downloads"
TRASHABLE_SUFFIXES = (
    ".zip",
    ".msi",
    ".exe",
    ".dmg",
    ".iso",
    ".pdf",
    ".7z",
    ".tar",
    ".gz",
)


def close_active_window(dry_run: bool = True) -> str:
    """Close whichever window currently has focus."""
    try:
        import pygetwindow as gw
    except ImportError as exc:  # pragma: no cover - depends on the local install.
        raise RuntimeError("Closing a window requires pygetwindow.") from exc

    window = gw.getActiveWindow()
    if window is None:
        return "No active window to close."

    title = (window.title or "the active window").strip()
    if dry_run:
        return f"Would close {title}"

    try:
        window.close()
    except Exception:  # noqa: BLE001 -- pragma: no cover, some windows refuse to close
        return f"Could not close {title}"
    return f"Closed {title}"


def find_latest_download(root: Path | None = None) -> Path | None:
    """Return the most recent unfinished-looking download, if there is one."""
    directory = Path(root) if root is not None else DOWNLOAD_DIR
    if not directory.is_dir():
        return None

    candidates = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in TRASHABLE_SUFFIXES
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def move_latest_download_to_recycle_bin(
    dry_run: bool = True, root: Path | None = None
) -> str:
    """Send the newest download to the Recycle Bin instead of deleting it."""
    candidate = find_latest_download(root)
    if candidate is None:
        return "There is no recent download to move."

    directory = Path(root) if root is not None else DOWNLOAD_DIR
    try:
        # Resolve first: a symlink or junction pointing out of the
        # allow-listed folder must not be followed to whatever it targets.
        candidate = ensure_inside(directory, candidate)
    except GuardViolation:
        return "That download points outside the folder; I left it alone."

    if dry_run:
        return f"Would move {candidate.name} to the Recycle Bin"

    try:
        from send2trash import send2trash
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("The Recycle Bin requires send2trash.") from exc

    send2trash(str(candidate))
    return f"Moved {candidate.name} to the Recycle Bin"


def lock_workstation(dry_run: bool = True) -> str:
    """Lock the session; harmless to confirm but still user-initiated."""
    if dry_run:
        return "Would lock the workstation"

    try:
        import ctypes

        ctypes.windll.user32.LockWorkStation()
    except Exception as exc:  # pragma: no cover - Windows-only call.
        raise RuntimeError(f"Could not lock the workstation: {exc}") from exc
    return "Locked the workstation"