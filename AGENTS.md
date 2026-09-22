# AGENTS.md

## What this is

Dude (package `nova_agent`, classes still `Nova*`): a local-first Windows voice
assistant (Python 3.11, single package, no monorepo). Whisper STT → MiniLM intent
routing → SQLite context → Kokoro TTS. Windows-only in practice
(`os.startfile`, pyautogui/pygetwindow), though a few paths have xdg-open fallbacks.

Wake is two-tier: **transcript wake** (primary) segments idle speech with Silero
VAD, transcribes it, and matches the wake phrase (default `"hey dude"`, see
`core/wake_phrase.py`); the ONNX audio model (`WakeWordEngine`) is the fast
secondary trigger.

## Commands

All commands assume the project venv. `start.bat` bootstraps it (creates `.venv`,
installs `requirements.txt`, runs the listener).

```powershell
.venv\Scripts\Activate.ps1
python -m pytest                          # full suite (testpaths=tests, pythonpath=.)
python -m pytest tests/test_phase0.py::test_intents_load   # single test
python -m ruff check .                    # ruff 0.16.x, line-length 100 (pyproject.toml)
python -m nova_agent --check              # env/detector/capability report — run after setup changes
python -m nova_agent --listen             # hands-free loop (add --live to really act)
python -m tests.test_mic                  # records a real 3s wav; NOT collected by pytest
```

There is no typechecker, no CI workflow, no pre-commit config. Verification is
`ruff check` + `pytest` locally.

## Invariants (do not break)

- **Dry-run is the default.** Actions only execute with `--live` or `NOVA_DRY_RUN=0`.
  New action code must honor `dry_run` and print what it *would* do.
- **Destructive actions** (`close_window`, `tidy_downloads`, `lock_workstation`) stay
  behind `NOVA_ALLOW_DESTRUCTIVE=1` *and* a spoken confirmation. Recycle bin only —
  never `os.remove`/`shutil.rmtree` for user files.
- **`CapabilityRegistry` gates the dispatcher** (`CommandProcessor.run_observation`).
  New actions must be added to `IMPLEMENTED_ACTIONS`/`DESTRUCTIVE_ACTIONS` in
  `nova_agent/main.py`, or they answer "not enabled yet".
- The PortAudio callback (`_on_audio`) must never block — queue frames only; heavy
  work happens on the worker thread. Frames captured while `is_speaking` are dropped
  so the assistant cannot wake itself on its own TTS output. It must also **copy**
  before queueing: `indata` is PortAudio's reusable ring-buffer memory, and a view
  of it is overwritten by the next callback (the VAD holds frames until segment
  finalization, which silently wrote silence — the "too quiet"/`''` transcript bug).

## Architecture notes (not obvious from filenames)

- `nova_agent/main.py` (~1,250 lines) is the real hub: CLI flags, `NovaAgent` listen
  loop (idle branch: audio model → transcript wake via `vad_reader` + STT),
  `CommandProcessor` (transcribe → observe → run_observation, split so a
  spoken confirmation can sit between choosing and running an action), and
  `check_environment`/`calibrate_microphone`/`wake_probe` helpers.
  `handle_command(audio_path=None, ..., text=...)` skips transcription when the
  text is already known (transcript-wake one-shot commands).
- `nova_agent/core/wake_phrase.py` = pure transcript matcher (normalize +
  word-boundary hit + remainder extraction); `core/` otherwise = pipeline stages;
  `tools/` = side-effecting Windows actions;
  `ui/overlay.py` = HUD state machine (Qt signals, never touch widgets off the GUI
  thread); `config/settings.py` = frozen `Settings` dataclass + path constants;
  `config/intents.json` = intent definitions.
- Env vars (`NOVA_*`) are read **once at import** of `settings.py` (module-level
  defaults), not per-call. Restart the process to change them.
- `ContextEngine()` defaults to CWD-relative `memory/nova.db`, while `settings.DATABASE_PATH`
  is repo-root anchored. Running from another CWD silently creates a second database.

## Testing quirks

- Tests inject fakes: `CommandProcessor(FakeSTT, FakeRouter, FakeTTS)` — follow that
  pattern; no test needs a microphone or GPU.
- Tests hard-code counts/names: `len(router.intents) == 15`, risky intents ==
  `{close_window, tidy_downloads, lock_workstation}` (both in `tests/test_phase0.py`).
  Adding or renaming an intent in `intents.json` requires updating those tests and
  `IMPLEMENTED_ACTIONS`.
- Wake/VAD tests **skip** (not fail) when the ONNX models are missing — a green
  suite does not prove the models exist. `tests/conftest.py` prints a loud
  "RUNTIME ASSETS MISSING" banner when that happens; check `--check` output
  for the per-backend reason.

## Runtime assets

`.onnx` wake/VAD models (`assets/wake_word/`), TTS cache (`assets/tts_cache/`), and
`memory/*.db` are git-ignored and regenerate/download locally. `silero_vad.onnx`
must exist for the real VAD backend; `hey_nova.onnx` gates the audio-model
backend (note: it is byte-for-byte openWakeWord's `hey_mycroft_v0.1.onnx` — it
fires on "hey mycroft", not "hey dude"; transcript wake is the primary path).
Otherwise the code falls back to energy heuristics (different score scale,
different thresholds).

## Docs

- `README.md` — accurate for commands and env vars.
- `docs/ROADMAP.md` — phase status, **partially stale**: it still describes features
  as pending that are already implemented (Silero VAD, CapabilityRegistry wiring,
  destructive intents + confirmation loop, `--device`, `app_aliases`, fuzzy project
  matching, ruff installed). Verify against code before trusting its status claims.
