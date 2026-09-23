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


def multi_word_shortcut_names() -> set[str]:
    """Every multi-word Start Menu name, lowercase — the *long* names.

    Field feedback ("you need to add a long name"): Whisper's priors carry
    single words like "edge" fine, but long names ("microsoft edge", "file
    explorer") decode poorly without a hint, so the command slot biases the
    decoder with these. Lowercased because that is how they will be spoken;
    single-word stems stay out on purpose — they dilute the prompt budget
    faster-whisper caps at half the text context.
    """
    names: set[str] = set()
    for root in _start_menu_roots():
        if not root.is_dir():
            continue
        for link in root.rglob("*.lnk"):
            if " " in link.stem:
                names.add(link.stem.lower())
    return names


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
    """Launch a resolved executable or shortcut with the usual dry-run guard.

    CreateProcess (``subprocess.Popen``) cannot execute a ``.lnk`` shortcut or
    a directory — live "open microsoft edge" died with WinError 193 "%1 is not
    a valid Win32 application", because the resolver hands back Start Menu
    shortcuts (this machine's "chrome" resolves to one too). Shortcuts and
    folders therefore go through the shell (``os.startfile``, which also keeps
    a shortcut's own arguments); executables keep the direct Popen, with the
    shell as one fallback when CreateProcess refuses an unusual file. If
    neither can launch it, say so instead of raising into the listen loop.
    """
    if dry_run:
        return f"Would open {name} ({path})"
    if not path.exists():
        return f"App not found: {path}"
    shell_first = path.suffix.lower() == ".lnk" or path.is_dir()
    try:
        if shell_first:
            os.startfile(path)  # the shell opens shortcuts and folders
        else:
            subprocess.Popen([str(path)])
    except OSError as exc:
        if shell_first:
            print(f"Shell launch failed for {path}: {exc}")
            return f"I couldn't open {name}."
        print(f"CreateProcess refused {path}: {exc}; trying the shell")
        try:
            os.startfile(path)
        except OSError as shell_exc:
            print(f"Shell launch failed for {path}: {shell_exc}")
            return f"I couldn't open {name}."
    return f"Opening {name}"


def open_app(name: str, dry_run: bool = True) -> str:
    path = APP_PATHS.get(name) or resolve_app(name)
    if path is None:
        return f"Unknown app: {name}"
    return open_path(name, path, dry_run=dry_run)
