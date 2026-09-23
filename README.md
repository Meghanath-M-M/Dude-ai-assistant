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
execution mode, the volume and brightness backends (Core Audio vs media
keys; WMI vs unsupported), and which optional packages are installed.

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
  intelligible)` or `Intent: none (best score 0.51 for time_check, below the
  confidence band)` — the runner-up intent is named, so a near miss is
  diagnosable instead of just silent) → `Response: ...` (the same line that is
  spoken). A missed capture (empty
  transcript or a score below the confidence band — e.g. a TV or video winning
  the first segment after a wake) **keeps the turn open** — `command_window`
  (8 s) is re-armed after *every* miss (so the pipeline's own latency can't
  consume it; `command_max_turn` caps the whole turn at 30 s) — and prints
  `Still listening for your command.`, so the real command can still arrive;
  speaking the phrase and the command in one breath
  ("hey dude, open chrome") bypasses that race entirely. Boot prints
  `STT warm-up` synchronously and then warms the intent encoder and Kokoro on
  a **background thread** (`Intent warm-up: 21.3s (background)`,
  `TTS warm-up: 6.2s (background, 9 replies preloaded)`), so the
  listener is up immediately while command #1 doesn't pay the ~21s-per-model
  load tax. Command captures — and confirmation replies — transcribe with a
  command-vocabulary prompt (`COMMAND_PROMPT`), the same priming that made the
  wake phrase reliable; unprompted Whisper-small was mangling clear commands in
  the field (`mute the volume` → `we hope the volume`, `open python projects`
  → `open by 10 projects`). `--stats` latency budgets are field-calibrated
  over three rounds (stt 3s, intent 0.2s, tts 2.5s for synthesis-to-audio
  only, action 0.5s, total 15s — total is wall clock and includes *speaking*
  the reply, but never a filesystem path: `speakable()` keeps `C:\...` on the
  console only). Commands that *start* while the background warm-up still
  holds the model loads are excluded from the stats (they block by design —
  a boot-window command pinned intent at 19088ms; the verdict is taken at
  start, since such a command can outlive warm blocked on the TTS synthesis
  lock), and each command's stage timings reset at capture so a wake-segment
  transcription can't leak into its stats row.
  Because it transcribes
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
| "mute the volume", "turn up the sound", "set volume to thirty percent" | Real mute/level via Core Audio (`pycaw`); media keys as fallback |
| "increase the brightness", "dim the screen", "set brightness to fifty percent" | Real brightness via the display's WMI interface (`wmi`); honest refusal on displays without it |
| "read the screen", "what does this say" | OCR of the current screen |
| "open my ml project" | Opens a folder remembered in SQLite |
| "open notepad", "open excel", "open notepadd" | Any installed app (PATH, then Start Menu shortcuts, then a close-name match) |
| "hello", "bye", "what time is it" | Small talk, farewells, and the clock |
| "what's your name", "who are you" | Says who it is |
| "what can you do", "help" | Lists what is actually enabled |
| "increase brightness and open microsoft edge" | Compound commands run every matching clause in turn (risky clauses still ask first) |
| "open" (no app named) | Asks **"Open what?"** — never opens a default app |

Two rules keep the guessing out of the pipeline:

- **A launch verb with no app asks instead of acting.** "open" alone used to
  match the example *inside* "open chrome" and launch the browser; now the turn
  stays open and the answer ("notepad") is understood as the app it named.
- **A misheard name must still be a real app.** Resolution is exact first, then
  PATH, then Start Menu shortcuts, then a *conservative* closest-name match
  (≥4 characters, 0.82 similarity), and anything else gets the honest
  "I don't know how to open X yet." — never a guess.

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
default `"hey dude"`); the 18 task definitions live in
`nova_agent/config/intents.json`.

Command captures are primed twice for Whisper: an `initial_prompt` built from
the intent vocabulary (`COMMAND_PROMPT`) and a `hotwords` hint holding the app
names this install understands (the intent examples' app words, any alias
taught with "set browser to ...", and the long multi-word Start Menu names
like "microsoft edge" — the ones Whisper mangles). Whisper has no way to
guess a name it has
never seen — without it, "open notepad" can come back as "open note pad" or
worse. Wake and confirmation captures deliberately get **no** hotwords: biasing
a "yes" toward an app name would cancel confirmations.

Kokoro replies are cached (short, digit-free phrases only, so dynamic text
cannot bloat the cache) and the cache is *preloaded* during the background TTS
warm-up with the assistant's fixed replies — `CommandProcessor.canned_replies()`,
which includes the long ones the cache policy would otherwise refuse to store
(the screen-reading failure used to cost 2.7 s of synthesis on every use).

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

