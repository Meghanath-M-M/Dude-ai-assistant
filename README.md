# Nova Local Voice Agent

A local-first desktop voice assistant for Windows. Nothing leaves the machine:
Whisper transcribes, MiniLM routes the intent, SQLite remembers, and Kokoro speaks.

## Quick start

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nova_agent --check
```

`--check` reports the interpreter, the loaded intents, the STT device, the
execution mode, and which optional packages are installed.

## Running

```powershell
python -m nova_agent --once                  # record one command and process it
python -m nova_agent --listen                # hands-free wake-word loop
python -m nova_agent --listen --live         # hands-free, and really open things
python -m nova_agent --calibrate             # measure ambient mic energy
python -m nova_agent --listen --debug        # live energy + timing output
python -m pytest                             # run the test suite
```

If you want the installed entry point:

```powershell
pip install -e .
nova-agent --listen --live
```

### Dry run is the default

Every action runs in dry-run mode unless you pass `--live` (or set
`NOVA_DRY_RUN=0`). In dry-run, Nova says what it *would* do and touches nothing.
This is deliberate: launching applications is the only irreversible thing the
assistant does today, so it is opt-in.

## What Nova can do

| Utterance | Action |
|---|---|
| "open chrome", "launch the browser" | Starts Chrome |
| "open vscode", "launch my code editor" | Starts VS Code |
| "search the web for rust ownership" | Opens the search with the extracted query |
| "mute the volume", "turn up the sound" | Multimedia volume keys |
| "read the screen", "what does this say" | OCR of the current screen |
| "open my ml project" | Opens a folder remembered in SQLite |
| "hello", "what time is it" | Small talk and the clock |

Wake word, hands-free capture, and everything above are wired through a worker
thread, so microphone frames are never dropped while Whisper or Kokoro runs, and
Nova cannot trigger herself on her own speech.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `NOVA_DRY_RUN` | `1` | `0` executes actions instead of describing them |
| `NOVA_STT_DEVICE` | `cpu` | `cuda` uses the GPU when cuBLAS is available |
| `NOVA_TESSERACT_PATH` | unset | Path to `tesseract.exe` for screen reading |
| `NOVA_LLM_FALLBACK` | `0` | `1` enables the optional Ollama Tier 2 router |
| `NOVA_CHROME_PATH` | Program Files | Chrome executable |
| `NOVA_CODE_PATH` | `%LOCALAPPDATA%` | VS Code executable |

Tunables live in `nova_agent/config/settings.py`; the six (+2) task definitions
live in `nova_agent/config/intents.json`.

## Optional Windows tools

- **Tesseract OCR** is required for screen reading. Install it and set
  `NOVA_TESSERACT_PATH` if it is not on `PATH`; without it, "read the screen"
  says so instead of failing.
- **Playwright Chromium** (`playwright install chromium`) for scripted browsing.
- **Ollama** with a small model for the Tier 2 fallback (`NOVA_LLM_FALLBACK=1`).

Nova prefers CUDA for faster-whisper. If the CUDA cuBLAS DLLs are unavailable,
it automatically retries transcription on the CPU. This is slower but does not
require a CUDA toolkit installation.

For the microphone test:

```powershell
python -m tests.test_mic
```

## Notes

- Wake word models (`assets/wake_word/*.onnx`) and cached Kokoro audio
  (`assets/tts_cache/*.wav`) are git-ignored and regenerate locally.
- Phase status, known gaps, and the next steps live in `docs/ROADMAP.md`.

