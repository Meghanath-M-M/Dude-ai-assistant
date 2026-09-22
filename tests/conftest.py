"""Shared pytest hooks for the Nova test suite.

Wake/VAD tests *skip* when the ONNX models are absent, which lets a fresh
clone stay green while proving nothing about the real backends. This banner
makes that state impossible to miss in the run output (``python -m nova_agent
--check`` reports the same thing per backend).
"""

from nova_agent.config.settings import VAD_MODEL_PATH, WAKE_WORD_MODEL_PATH


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    missing = [path for path in (WAKE_WORD_MODEL_PATH, VAD_MODEL_PATH) if not path.exists()]
    if not missing:
        return
    skipped = len(terminalreporter.stats.get("skipped", []))
    terminalreporter.write_sep(
        "!",
        "RUNTIME ASSETS MISSING - a green suite does not prove the wake/VAD backends",
    )
    for path in missing:
        terminalreporter.write_line(f"  missing: {path}")
    if skipped:
        terminalreporter.write_line(f"  ({skipped} test(s) skipped in this run)")
    terminalreporter.write_line("  run `python -m nova_agent --check` for per-backend details")
