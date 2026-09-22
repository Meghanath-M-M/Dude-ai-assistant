"""Static and runtime guard rails for the side-effecting tool layer.

The AST scan keeps ``nova_agent/tools/`` honest: no hard-deleting calls, no
``shell=True``, no bare-string ``subprocess`` commands. The containment tests
prove ``ensure_inside`` blocks ``..``/symlink escapes and that
``tidy_downloads`` refuses a link that escapes the Downloads folder.
"""

import ast
from pathlib import Path

import pytest

from nova_agent.tools import desktop
from nova_agent.tools.guards import GuardViolation, ensure_inside

TOOLS_DIR = Path(__file__).resolve().parents[1] / "nova_agent" / "tools"

HARD_DELETE_KINDS = {
    "os.remove",
    "os.unlink",
    "path.unlink",
    "shutil.rmtree",
    "os.system",
}
SUBPROCESS_KINDS = {"subprocess.shell", "subprocess.string-command"}
SUBPROCESS_FUNCS = {"Popen", "run", "call", "check_call", "check_output"}

ALL_KINDS = HARD_DELETE_KINDS | SUBPROCESS_KINDS


def _scan_source(name: str, source: str) -> list[str]:
    """Return ``file:line: kind`` findings for one parsed source string."""
    findings = []
    tree = ast.parse(source, filename=name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        func = node.func
        receiver = func.value.id if isinstance(func.value, ast.Name) else None
        where = f"{name}:{node.lineno}"

        if func.attr == "unlink":
            kind = "os.unlink" if receiver == "os" else "path.unlink"
            findings.append(f"{where}: {kind}")
        elif func.attr == "remove" and receiver == "os":
            findings.append(f"{where}: os.remove")
        elif func.attr == "rmtree":
            findings.append(f"{where}: shutil.rmtree")
        elif func.attr == "system" and receiver == "os":
            findings.append(f"{where}: os.system")
        elif receiver == "subprocess" and func.attr in SUBPROCESS_FUNCS:
            for keyword in node.keywords:
                if (
                    keyword.arg == "shell"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    findings.append(f"{where}: subprocess.shell")
            if node.args and not isinstance(node.args[0], (ast.List, ast.Tuple)):
                findings.append(f"{where}: subprocess.string-command")
    return findings


def _violations(kinds: set[str]) -> list[str]:
    findings = []
    for path in sorted(TOOLS_DIR.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        findings.extend(
            line
            for line in _scan_source(path.name, source)
            if line.split(": ", 1)[1] in kinds
        )
    return findings


def test_tools_never_hard_delete_files():
    assert _violations(HARD_DELETE_KINDS) == []


def test_subprocess_calls_are_list_formed_and_shell_free():
    assert _violations(SUBPROCESS_KINDS) == []


def test_every_tool_file_parses():
    for path in sorted(TOOLS_DIR.glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_the_scan_actually_catches_forbidden_patterns():
    """Guard the guard: a planted violation must be reported, with location."""
    bad = _scan_source(
        "bad.py",
        "import os\nimport shutil\nos.remove('a')\nshutil.rmtree('b')\n",
    )
    assert bad == ["bad.py:3: os.remove", "bad.py:4: shutil.rmtree"]

    bad = _scan_source("bad.py", "import subprocess\nsubprocess.run('dir', shell=True)\n")
    assert bad == [
        "bad.py:2: subprocess.shell",
        "bad.py:2: subprocess.string-command",
    ]

    assert _scan_source("good.py", "import subprocess\nsubprocess.run(['dir'])\n") == []
    assert _scan_source("good.py", "from send2trash import send2trash\nsend2trash('f')\n") == []


def test_ensure_inside_accepts_children_of_the_root(tmp_path):
    downloads = tmp_path / "downloads"
    inside = downloads / "file.zip"
    inside.parent.mkdir()
    inside.write_bytes(b"x")

    assert ensure_inside(downloads, inside) == inside.resolve()


def test_ensure_inside_rejects_dot_dot_escapes(tmp_path):
    with pytest.raises(GuardViolation):
        ensure_inside(tmp_path / "downloads", tmp_path / "downloads" / ".." / "secret.zip")


def test_ensure_inside_rejects_symlinks_that_point_outside(tmp_path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    secret = tmp_path / "secret.zip"
    secret.write_bytes(b"x")
    link = downloads / "trap.zip"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this machine")

    with pytest.raises(GuardViolation):
        ensure_inside(downloads, link)


def test_tidy_downloads_refuses_a_link_that_escapes_the_folder(tmp_path, monkeypatch):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    secret = tmp_path / "secret.zip"
    secret.write_bytes(b"x")
    link = downloads / "trap.zip"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this machine")
    monkeypatch.setattr(desktop, "DOWNLOAD_DIR", downloads)

    message = desktop.move_latest_download_to_recycle_bin(dry_run=False)

    assert "left it alone" in message
    assert secret.exists()


def test_tidy_downloads_still_reports_a_normal_candidate(tmp_path, monkeypatch):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    (downloads / "setup.zip").write_bytes(b"x")
    monkeypatch.setattr(desktop, "DOWNLOAD_DIR", downloads)

    message = desktop.move_latest_download_to_recycle_bin(dry_run=True)

    assert message == "Would move setup.zip to the Recycle Bin"
