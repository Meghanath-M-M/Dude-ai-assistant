import difflib
import os
import shutil
import subprocess
from pathlib import Path

APP_PATHS = {
    "chrome": Path(
        os.getenv("NOVA_CHROME_PATH", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    ),
    "code": Path(
        os.getenv(
            "NOVA_CODE_PATH",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"),
        )
    ),
}


def _start_menu_roots() -> list[Path]:
    roots = []
    for env in ("APPDATA", "PROGRAMDATA"):
        base = os.getenv(env)
        if base:
            roots.append(Path(base) / "Microsoft/Windows/Start Menu/Programs")
    return roots


# Common spoken names that match neither the executable nor any shortcut
# ("open calculator" -> calc.exe; Win11 ships no Calculator.lnk).
RESOLVE_ALIASES = {"calculator": "calc"}

# Tolerance for a misheard shortcut name ("notepadd" -> Notepad.lnk). Kept
# conservative on purpose: a 4-character floor plus this ratio keeps "notes"
# away from "Notepad" while a swallowed final consonant still lands. Treating
# the edit distance as a *cost* that must stay inside a budget (rather than as
# a score to maximize) follows Home Assistant's Hassil matcher.
FUZZY_MIN_LENGTH = 4
FUZZY_CUTOFF = 0.82


def resolve_app(name: str) -> Path | None:
    """Find an installed app the hard way: PATH first, then Start Menu links.

    ``APP_PATHS`` only knows chrome and code; "open notepad" or "open excel"
    should still work out of the box. Executables on PATH cover System32
    tools (notepad, calc); .lnk shortcuts cover Start Menu apps, matched by
    exact name, then prefix, then any whole word ("edge" -> "Microsoft Edge").
    ``RESOLVE_ALIASES`` covers spoken names that match neither. Returns
    ``None`` so the caller can answer honestly instead of guessing.
    """
    wanted = " ".join(name.lower().split())
    if not wanted:
        return None
    candidates = [wanted]
    alias = RESOLVE_ALIASES.get(wanted)
    if alias:
        candidates.append(alias)

    shortcuts: list[Path] = []
    for root in _start_menu_roots():
        if not root.is_dir():
            continue
        found = sorted(root.rglob("*.lnk"))
        if found:
            shortcuts.extend(found)  # user Start Menu is scanned before system
    stems = {link.stem.lower(): link for link in shortcuts}

    for candidate in candidates:
        exe = shutil.which(candidate) or shutil.which(f"{candidate}.exe")
        if exe:
            return Path(exe)
        if candidate in stems:
            return stems[candidate]
    if len(wanted) >= 3:
        long_enough = [c for c in candidates if len(c) >= 3]
        for stem in sorted(stems):  # deterministic: prefix, then whole word
            if any(stem.startswith(c) for c in long_enough):
                return stems[stem]
        for stem in sorted(stems):
            words = stem.replace("-", " ").split()
            if any(c in words for c in candidates):
                return stems[stem]

    if len(wanted) >= FUZZY_MIN_LENGTH:
        # Last resort: the closest Start Menu *name*, matched against both the
        # shortcut titles and their individual words ("Microsoft Edge" ->
        # "edge"), so "spotifi" still finds Spotify. Shortcut stems win on
        # collision, and the cutoff keeps unrelated names out.
        names: dict[str, Path] = dict(stems)
        for stem, link in sorted(stems.items()):
            for word in stem.replace("-", " ").split():
                names.setdefault(word, link)
        close = difflib.get_close_matches(
            wanted, sorted(names), n=1, cutoff=FUZZY_CUTOFF
        )
        if close:
            return names[close[0]]
    return None


def open_path(name: str, path: Path, dry_run: bool = True) -> str:
    """Launch a resolved executable/shortcut with the usual dry-run guard."""
    if dry_run:
        return f"Would open {name} ({path})"
    if not path.exists():
        return f"App not found: {path}"
    subprocess.Popen([str(path)])
    return f"Opening {name}"


def open_app(name: str, dry_run: bool = True) -> str:
    path = APP_PATHS.get(name) or resolve_app(name)
    if path is None:
        return f"Unknown app: {name}"
    return open_path(name, path, dry_run=dry_run)
