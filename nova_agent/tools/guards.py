"""Guard rails for side-effecting tools.

Two invariants, both enforced by ``tests/test_tool_guards.py``:

1. The tool layer may never hard-delete files (``os.remove``/``os.unlink``/
   ``Path.unlink``/``shutil.rmtree``/``os.system``) and may never shell out
   with ``shell=True`` or a bare-string command -- Recycle Bin or a
   list-formed ``subprocess`` call only.
2. Destructive helpers must run every path through :func:`ensure_inside`
   before acting on it, so symlinks, junctions, and ``..`` segments cannot
   smuggle a target in from outside the allow-listed root.
"""

from __future__ import annotations

from pathlib import Path


class GuardViolation(RuntimeError):
    """Raised when an action would touch something outside its allow-list."""


def ensure_inside(root: Path | str, candidate: Path | str) -> Path:
    """Return ``candidate`` fully resolved if it stays inside ``root``.

    Both sides are resolved first so ``..`` segments, symlinks, and NTFS
    junctions cannot smuggle a path past the check. Callers must act only on
    the returned path; anything else raises :class:`GuardViolation`.
    """
    resolved_root = Path(root).resolve()
    resolved = Path(candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise GuardViolation(f"{candidate} is outside {resolved_root}")
    return resolved
