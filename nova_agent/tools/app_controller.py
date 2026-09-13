import os
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


def open_app(name: str, dry_run: bool = True) -> str:
    path = APP_PATHS.get(name)
    if path is None:
        return f"Unknown app: {name}"
    if dry_run:
        return f"Would open {name} ({path})"
    if not path.exists():
        return f"App not found: {path}"
    subprocess.Popen([str(path)])
    return f"Opening {name}"
