# Dude — Local Voice Agent

A local-first desktop voice assistant for Windows (package `nova_agent`, class
names still `Nova*`). Nothing leaves the machine: Whisper transcribes, MiniLM
routes the intent, SQLite remembers, and Kokoro speaks.

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
python -m nova_agent --calibrate             # two-step wizard: ambient energy, then 5x "hey dude"
python -m nova_agent --wake-status           # wake phrase, backend, model path, threshold, last score
python -m nova_agent --wake-probe            # record wake attempts: transcript hits + model threshold
python -m nova_agent --listen --debug        # live energy + timing output
python -m nova_agent --listen --stats        # latency summary (min/median/mean/max vs budget) on exit
python -m pytest                             # run the test suite
```

If you want the installed entry point:

```powershell
pip install -e .
nova-agent --listen --live
```

### Dry run is the default

Every action runs in dry-run mode unless you pass `--live` (or set
`NOVA_DRY_RUN=0`). In dry-run, Dude says what it *would* do and touches nothing.
This is deliberate: launching applications is the only irreversible thing the
assistant does today, so it is opt-in.

## Waking Dude ("hey dude")

Two detectors run while `--listen` is idle, and they work differently:

- **Transcript wake (primary)** — idle speech is segmented by Silero VAD,
  transcribed by Whisper, and matched against the wake phrase (word-boundary
  aware, so "hey dudes" doesn't count). It needs no training and no threshold:
  say **"hey dude"** and it answers. Idle transcription is **primed with the
  wake phrase** (`initial_prompt`) — unprompted Whisper systematically mishears
  short phrases on some voices ("what are you doing?" for "hey dude") — and
  segments below an RMS floor (`wake_segment_min_rms`) are skipped so
  near-silence never reaches the decoder. Every exchange is visible on the
  console as it happens: `Wake detected - listening for your command.` →
  `Heard: ...` / `Intent: ...` (misses show *why*: `Heard: (nothing
  intelligible)` or `Intent: none (best score ... below the confidence band)`)
  → `Response: ...` (the same line that is spoken). A missed capture (empty
  transcript or a score below the confidence band — e.g. a TV or video winning
  the first segment after a wake) **keeps the turn open** — `command_window`
  (8 s) is re-armed after *every* miss (so the pipeline's own latency can't
  consume it; `command_max_turn` caps the whole turn at 30 s) — and prints
  `Still listening for your command.`, so the real command can still arrive;
  speaking the phrase and the command in one breath
  ("hey dude, open chrome") bypasses that race entirely. Startup prints
  `STT warm-up` and `Intent warm-up` so the first command doesn't pay the
  model-load tax. Because it transcribes
  idle speech, expect Whisper CPU use whenever someone nearby is talking; set
  `NOVA_TRANSCRIPT_WAKE=0` to turn it off.
- **Audio model (secondary)** — `assets/wake_word/hey_nova.onnx` scores every
  chunk. Note: that file is byte-for-byte openWakeWord's official
  `hey_mycroft_v0.1.onnx`, so it fires on **"hey mycroft"**, not "hey dude" —
  a bonus trigger, not the main one. A real `hey dude` model can replace it
  later via openWakeWord training.

Wake latency for the transcript path is roughly the end of the phrase plus one
short transcription (~0.5–1 s). Use `--calibrate` to see per-attempt HIT/miss
results through the real pipeline.

## What Dude can do

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
Dude cannot trigger itself on its own speech.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `NOVA_DRY_RUN` | `1` | `0` executes actions instead of describing them |
| `NOVA_STT_DEVICE` | `cpu` | `cuda` uses the GPU when cuBLAS is available |
| `NOVA_TESSERACT_PATH` | unset | Path to `tesseract.exe` for screen reading |
| `NOVA_LLM_FALLBACK` | `0` | `1` enables the optional Ollama Tier 2 router |
| `NOVA_LLM_MODEL` | `phi4-mini` | Ollama model the Tier 2 router asks for |
| `NOVA_LLM_MIN_SCORE` | `0.65` | Tier 1 scores below this never reach the LLM |
| `NOVA_TRANSCRIPT_WAKE` | `1` | `0` disables transcript wake (audio model only) |
| `NOVA_CHROME_PATH` | Program Files | Chrome executable |
| `NOVA_CODE_PATH` | `%LOCALAPPDATA%` | VS Code executable |

Tunables live in `nova_agent/config/settings.py` (wake phrase: `wake_word`,
default `"hey dude"`); the 15 task definitions live in
`nova_agent/config/intents.json`.

## Optional Windows tools

- **Tesseract OCR** is required for screen reading. Install it and set
  `NOVA_TESSERACT_PATH` if it is not on `PATH`; without it, "read the screen"
  says so instead of failing.
- **Playwright Chromium** (`playwright install chromium`) for scripted browsing.
- **Ollama** for the Tier 2 fallback. `start.bat` sets `NOVA_LLM_FALLBACK=1`
  before launching; pull the default model once with `ollama pull phi4-mini`
  (~2.5 GB). When the daemon or model is missing the fallback just answers
  `unknown`, so the voice loop never stalls.

Dude prefers CUDA for faster-whisper. If the CUDA cuBLAS DLLs are unavailable,
it automatically retries transcription on the CPU. This is slower but does not
require a CUDA toolkit installation.

For the microphone test:

```powershell
python -m tests.test_mic
```

## Notes

- Verify changes with `python -m pytest` and `python -m ruff check .`
  (ruff line-length 100); run a single test with
  `python -m pytest tests/test_wake_loop.py::test_name`.
- Wake word models (`assets/wake_word/*.onnx`) and cached Kokoro audio
  (`assets/tts_cache/*.wav`) are git-ignored and regenerate locally.
- Phase status, known gaps, and the next steps live in `docs/ROADMAP.md`.

