# Nova Roadmap

Status of the local voice agent, phase by phase. Phases are ordered by risk:
prove the loop, then make it reliable, then make it pretty.

Legend: **done** | **next** | **later**

## Phase 0 — Environment and baseline: done

- Python 3.11.9 venv at `.venv`, all runtime dependencies installed and verified.
- `pytest` was **not** actually installed (a stale `.pytest_cache` suggested
  otherwise); it is installed now and the suite is green.
- Deleted stray root recordings (`temp_command.wav`, `test.wav`).
- `.gitignore` now covers `.kilo/` and `assets/wake_word/*.tflite`.
- Hugging Face cache is warm: `all-MiniLM-L6-v2`, `faster-whisper-small`,
  `Kokoro-82M`, so the first run after a reboot is not a download.

## Phase 1 — The live loop: done

- **Audio thread split.** Microphone frames are queued by the PortAudio callback
  and consumed on a worker thread. Transcription, synthesis, and window
  automation no longer run in the audio callback, so frames are not dropped.
- **No self-triggering.** Frames captured while Nova is speaking are discarded,
  so she cannot wake herself on her own Kokoro playback.
- **Real execution.** `NOVA_DRY_RUN` / `--live` decides whether actions execute.
  Dry run remains the default; `--listen` reports the mode it is running in.
- **Query extraction.** "search the web for rust ownership" searches for
  "rust ownership" rather than the whole transcript.
- **Screen OCR wired.** The configured Tesseract path and length are used, and a
  missing binary produces an actionable message instead of a crash.
- **Orphan intents implemented.** `context_open` and `system_control` used to
  answer "not enabled yet".
- **Fuzzy project lookup.** "open my ml project" resolves "ML Projects".
- **LLM fallback contract.** `unknown` is a miss again; the prompt matches the
  dispatcher; `NOVA_LLM_FALLBACK=1` opts in.
- **Wake word actually runs.** openWakeWord ships without its feature models, so
  the classifier silently failed to load and detection fell back to loudness,
  where noise scored 0.29 against a 0.13 threshold. The pipeline now points at
  the in-repo feature models: noise scores 0.0, and the trigger threshold
  follows the active backend (0.5 model / 0.13 fallback). Cost: 3.3 ms per
  80 ms frame.

## Phase 2 — Wake word and VAD quality: next

- Tune the model threshold and hysteresis against real speech (record "hey nova"
  ~10 times and look at the score distribution; `--calibrate` currently only
  covers the energy fallback).
- Load `assets/wake_word/silero_vad.onnx` into `VADRecorder` through
  `onnxruntime` (already installed). Today the recorder uses an amplitude
  heuristic, so quiet speech can be truncated and background noise can extend a
  recording. Add a hangover of ~0.7 s before finalising.
- Move the `temp_command.wav` write out of the working directory.
- Verify the ONNX wake model loads from a clean checkout (the `.onnx` files are
  git-ignored, so document how to regenerate them).

## Phase 3 — Latency: next

- `NOVA_STT_DEVICE=cuda` and a warm-up transcription at startup (the first
  Whisper call pays the model load).
- Per-stage timings behind `--debug` (already partially wired) plus a `--stats`
  summary. Budget: wake to response under 2 s.
- Compose cached TTS prefixes with short dynamic tails instead of synthesising
  whole sentences (there are already `the_current_time_is` / `searching_for`
  clips in the cache from an earlier attempt).

## Phase 4 — Context depth: later

- `app_aliases` table so "browser" maps to a configured app.
- Multi-turn: "do that again" on top of `ContextEngine.get_last_command`, and a
  pending-intent slot so a spoken confirmation can complete an action.
- Preferences that change behaviour: browser, voice, wake threshold, dry run.

## Phase 5 — HUD overlay: later

- `ui/overlay.py` is still a print stub even though PyQt5 5.15.11 is installed.
  Build a frameless, translucent, always-on-top widget with the four states and
  a transcript ribbon, driven by Qt signals from the worker thread, plus a
  `--no-hud` flag so headless runs and the existing HUD tests keep working.

## Phase 6 — Safety, exercised: later

- The confirmation path exists and is tested but no intent is marked
  `safe: false`, so it is dead code in practice. Add a small destructive set
  behind `NOVA_ALLOW_DESTRUCTIVE=1` (close active window, move the last download
  to the Recycle Bin with `send2trash`, lock the workstation).
- Voice confirmation loop: capture the next utterance, accept only
  yes/confirm/proceed, speak "Cancelled" on anything else.
- Wire `CapabilityRegistry` in as the dispatcher gate so nothing can bypass
  confirmation, and keep hard blocks on `os.remove` / `shutil.rmtree`.

## Phase 7 — Tier 2 fallback: later

- Ollama is not installed, so `LLMFallback` currently degrades to `unknown`.
  Install `qwen2.5:3b` (~2 GB VRAM) or leave it off.
- Validate any LLM-proposed action against the capability registry, add a
  timeout, and only use it in the 0.65-0.82 confidence band.

## Phase 8 — Polish: later

- Delete or wire the remaining unused scaffolding (`CommandQueue`,
  `AudioSession`, `RuntimeMonitor`, `CapabilityRegistry`).
- Install `ruff` (declared in `pyproject.toml`) or drop the section; add
  `psutil` if `RuntimeMonitor` is kept.
- One-hour soak test: real microphone, repeated wake cycles, watch RSS, VRAM,
  cache size, and the SQLite file.
