# AGENTS.md

## What this is

Dude (package `nova_agent`, classes still `Nova*`): a local-first Windows voice
assistant (Python 3.11, single package, no monorepo). Whisper STT → MiniLM intent
routing → SQLite context → Kokoro TTS. Windows-only in practice
(`os.startfile`, pyautogui/pygetwindow), though a few paths have xdg-open fallbacks.

Wake is two-tier: **transcript wake** (primary) segments idle speech with Silero
VAD, transcribes it, and matches the wake phrase (default `"hey dude"`, see
`core/wake_phrase.py`); the ONNX audio model (`WakeWordEngine`) is the fast
secondary trigger. Field round 8 (false wakes from background/echo) added four
guards: the sound model's default threshold is **0.65** (`MODEL_THRESHOLD` — the
stand-in `hey_mycroft` model crossed 0.5 on chatter), `_on_audio` drops the
**reverb tail** (`wake_cooldown`, 0.5s after every reply, stamped by
`NovaAgent._finish_speaking`), idle probes close at **`idle_segment_max`** (10s;
commands keep `vad_max_seconds`), and the wake line names its source —
`Wake detected (audio model|transcript) - listening for your command.`
Field round 9 added the **voice gate** (`core/voice_gate.py`): after
`python -m nova_agent --enroll-voice` stores a voiceprint (three clips carved
into probe-length 1.5s windows — round 10, after silence-padded probes scored
against one long-utterance centroid at 0.5 vetoed the enrolled voice itself),
every wake candidate — the transcript segment and a rolling ~1.5s window for
the audio model — is trimmed to its voiced part and scored (speechbrain ECAPA,
**max** cosine over the reference windows ≥ `voice_threshold`, default 0.4 =
the confident-mismatch line, since different-speaker ECAPA scores concentrate
below ~0.3). A mismatch prints
`[wake] ignored (voice mismatch 0.32 < 0.4): '<text>'` and stays silent; an
accepted wake prints `[wake] matched: '<text>' (voice 0.61)` — scores visible
so thresholds tune from data. No voiceprint, a missing
backend, or `NOVA_VOICE_GATE=0` leaves the gate open (phrase-only wake as
before) — degradation is never a dead loop.

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
  so the assistant cannot wake itself on its own TTS output — the one exception is
  actual playback, where `TTSEngine.speak`'s barge gate is deliberately the queue's
  only consumer (see the barge-in note below); `_on_audio` itself still does nothing
  but queue. It must also **copy**
  before queueing: `indata` is PortAudio's reusable ring-buffer memory, and a view
  of it is overwritten by the next callback (the VAD holds frames until segment
  finalization, which silently wrote silence — the "too quiet"/`''` transcript bug).

## Architecture notes (not obvious from filenames)

- `nova_agent/main.py` (~2,000 lines) is the real hub: CLI flags, `NovaAgent` listen
  loop (idle branch: audio model → transcript wake via `vad_reader` + STT),
  `CommandProcessor` (transcribe → observe → run_observation, split so a
  spoken confirmation can sit between choosing and running an action), and
  `check_environment`/`calibrate_microphone`/`wake_probe` helpers.
  `handle_command(audio_path=None, ..., text=...)` skips transcription when the
  text is already known (transcript-wake one-shot commands).
  `observe` splits a compound utterance ("increase brightness and open
  microsoft edge") on conjunctions (`split_compound`, capped at 3 clauses) and
  queues follow-up clauses in `pending_compound` — but only when *two* clauses
  match an intent, so "please and open chrome" still routes whole;
  `run_observation` runs the queue in turn (one `_respond` per call via
  `join_replies`), pausing whenever a clause needs its own spoken confirmation,
  and sums clause action times for `--stats`.
- Barge-in (Wave 1a): `core/barge.py`'s `BargeGate` decides when a person is
  talking over the reply. It is an *energy* gate with an echo-adaptive
  baseline (EMA updated only below threshold, so a raised voice cannot ratchet
  the bar against itself), primed from the first frame after `barge_begin`
  flushes the pre-playback backlog — deliberately not the Silero VAD, which
  scores the assistant's own echo as speech and would cut us off on ourselves.
  `TTSEngine.speak` streams every segment through one `sd.OutputStream` in
  ~50ms chunks (`_play_interruptible`), checking `self._interrupt` and
  `barge_check` between writes and never calling the global
  `sounddevice.stop()` (that would take the mic stream down with it);
  `TTSEngine.stop()` only raises the flag — honored at the next chunk
  boundary — and each `speak()` clears it first so a stale stop cannot swallow
  a fresh reply. The hooks (`barge_begin`/`barge_check`) are installed on
  `processor.tts` in `NovaAgent.run()`; between chunks the worker thread,
  blocked in playback, is the only reader of `audio_queue`, which is why
  `_barge_check` can drain it there while the echo guard drops frames
  everywhere else.
- `nova_agent/core/wake_phrase.py` = pure transcript matcher (normalize +
  word-boundary hit + remainder extraction); `core/` otherwise = pipeline stages;
  `tools/` = side-effecting Windows actions;
  `ui/overlay.py` = HUD state machine (Qt signals, never touch widgets off the GUI
  thread); `config/settings.py` = frozen `Settings` dataclass + path constants;
  `config/intents.json` = intent definitions.
- `nova_agent/core/voice_gate.py` = speaker verification for wake candidates
  (`VoiceGate` + `enroll_voice`): embeddings load lazily on CPU, every failure
  degrades *open* (warn once, phrase-only wake), and tests inject their own
  gate — conftest's autouse fixture keeps a real local voiceprint from gating
  the sine-wave fixtures. STT now defaults to **cuda** (`DEFAULT_STT_DEVICE`):
  `STTEngine.__init__` falls back to CPU at load (RuntimeError → warn + rebuild)
  on top of the existing cuBLAS decode retry, so a bare `python -m nova_agent`
  uses the GPU while non-CUDA machines still boot. The decode retry was firing
  on this machine (`Library cublas64_12.dll is not found`) because ctranslate2
  loads cuBLAS lazily and the pip wheels park it in `site-packages/nvidia/*/bin`
  — `stt.py::_add_cuda_dll_dirs` puts those dirs on the DLL search path (note:
  `find_spec("nvidia")` is a PEP 420 namespace package, `origin is None`,
  which silently defeated the first attempt; keep the `os.add_dll_directory`
  handles in `_DLL_DIRS` or GC removes them). With `nvidia-cublas-cu12` +
  `nvidia-cudnn-cu12` on requirements: steady-state STT 0.07s per 2s clip.
- Env vars (`NOVA_*`) are read **once at import** of `settings.py` (module-level
  defaults), not per-call. Restart the process to change them.
- `ContextEngine()` defaults to CWD-relative `memory/nova.db`, while `settings.DATABASE_PATH`
  is repo-root anchored. Running from another CWD silently creates a second database.
- App opening is a resolution chain: `extract_app_phrase` (known alias first,
  then the trailing phrase of a launch verb) → `context.get_alias` →
  `APP_PATHS` (chrome/code) → `tools/app_controller.resolve_app` (PATH, then
  Start Menu `.lnk`, then a deliberately conservative closest-name match:
  `FUZZY_MIN_LENGTH`/`FUZZY_CUTOFF`, ≥4 chars and 0.82 ratio, so "notepadd"
  finds Notepad while "notes" does not), else an honest "I don't know how to
  open X yet."
  `set browser` accepts any name the same resolver can find (it used to
  whitelist only chrome/code), so "set browser to edge" sticks.
  A launch verb with **no** app at all (bare "open.", where `extract_app_phrase`
  returns None and the intent has no `target`) asks `OPEN_WHAT_RESPONSE`
  ("Open what?") and sets `CommandProcessor.pending_clarification`; the next
  capture is then dispatched as `open <answer>` instead of being read as a new
  command. `RETRY_RESPONSES` (miss **and** the question) keeps that turn open in
  `NovaAgent._finish_turn`. Defaulting to chrome there was the field bug where
  `Heard: open.` opened the browser. `PRONOUN_TARGETS` ("it", "that", …) name
  nothing, so "open it" first resolves against `CommandProcessor.last_open` —
  the last app or project actually launched (a dedicated field, *not*
  `last_text`, which the pronoun command itself would overwrite with "open it"
  and so lose the target after one use) — asking `OPEN_WHAT_RESPONSE` only
  when history holds no referent. Launching: `.lnk`
  shortcuts and folders go through `os.startfile` (ShellExecute —
  CreateProcess refuses them with WinError 193 "%1 is not a valid Win32
  application", the live `open microsoft edge` crash; the shell also keeps a
  shortcut's own arguments), executables keep `Popen` with the shell as an
  OSError fallback, and a double failure answers "I couldn't open X." instead
  of raising into the listen loop.
- `CommandProcessor.transcribe(..., command=True)` is the *only* slot that gets
  STT `hotwords` (`stt_hotwords()`: app words + resolver aliases + remembered
  aliases + the multi-word Start Menu names — "microsoft edge", "file
  explorer" — from a once-per-session `multi_word_shortcut_names()` scan;
  single-word stems stay out of that budget). Wake segments and confirmation
  replies must not get them — a "yes"
  decoded as an app name cancels a confirmation. `hotwords` is faster-whisper's
  own parameter (prompt-token biasing, auto-truncated to half the text context).
- TTS caching has two tiers, and the difference matters for the `tts` latency
  stage: `should_cache` governs *dynamic* replies (≤6 words, ≤60 chars, no
  digits) so they cannot bloat the cache, while `TTSEngine.preload(phrases)`
  stores the fixed replies from `CommandProcessor.canned_replies()` whatever
  their length, during the background warm-up (`warm_up(phrases=...)`). `speak`
  reads the cache *before* consulting the policy, so a preloaded reply reports
  `last_synthesis_seconds == 0` — that is the whole reason the 15-word
  screen-reading failure stopped blowing the 2.5 s `tts` budget. Preload skips
  entries already on disk (no re-synthesis on later boots). The screen-reading
  string is single-sourced in `tools/screen_reader.py`
  (`TESSERACT_MISSING_MESSAGE`) because the cached key must match exactly.
- The TTS cache is keyed by **phrase only**, so a voice change drops and
  rebuilds it: `_set_preference` → `_recache_replies()` clears
  (`TTSEngine.clear_cache()`) and re-preloads on a daemon thread — otherwise
  every cached reply, including the preloaded phrasebook, keeps playing in the
  old voice after "set voice to am_michael".
  `_respond` prints the full response but speaks `speakable(response)` —
  filesystem paths are console detail, never read aloud. `--stats` skips
  commands that **start** while `NovaAgent.warm_done` is clear (background
  warm-up in flight) so boot-window artifacts don't pin the worst-case
  verdict — the verdict is taken at command *start* because a mid-warm
  command can outlive warm blocked on the TTS synthesis lock. `_run_captured`
  also resets `processor.timings` per command so the wake-segment
  transcription can't leak into a text-path command's snapshot.
- `_lexical_fallback` claims on **whole words only** and refuses
  normalizations shorter than 2 chars — raw substring matching let a bare
  "i" claim "launch visual studio code" and open VS Code from noise (field
  round 3). A launch verb plus an unknown app claims `open_app` at 0.75
  after all specific shapes (project/volume/examples) have had their turn.
  Two field-round-5 guards sit around that loop: a normalization made *only* of
  launch verbs (`LAUNCH_WORDS`) returns `open_app` with no target instead of
  matching `\bopen\b` inside the example "open chrome", and a `TIME_WORDS` token
  claims `time_check` *after* the example loop (so "it's time to open chrome"
  still opens Chrome). `IntentRouter.match` also records `last_best_label`, which
  the miss line prints so a near miss names the intent it nearly matched.
- Volume control is Core Audio first (`pycaw`, `tools/system.py`):
  `parse_volume_command` matches **unmute before mute** ("unmute" contains
  "mute") and understands spoken levels ("fifty percent"); `apply_volume`
  reads the level *back* from the endpoint ("Volume set to 30 percent.").
  `_endpoint()` calls `comtypes.CoInitialize()` because commands run on the
  worker thread (pycaw fails there with "CoInitialize has not been called" —
  found live; `--check` works only because it runs on the main thread).
  Without pycaw the media-key fallback can only nudge and toggle — it says
  "Toggled mute" and refuses exact levels rather than lying. `--check` prints
  `Volume backend: core audio | media keys`. Tests use the fake endpoint in
  `tests/test_volume.py` — never move the machine's real volume; dry-run
  returns "Would ..." *before* touching the endpoint.
- Brightness shares `tools/system.py` and the `system_control` action:
  `intents.json`'s `system_brightness` (target `brightness`) routes through
  `CommandProcessor._control_system`, which picks the device from
  `intent.get("target")`. Levels go through the `wmi` package to `root\WMI`,
  and `WmiSetBrightness` is called **by keyword** (`Brightness=`, `Timeout=1`)
  because this machine's MOF declares `(Brightness, Timeout)` — the *reverse*
  of the `(Timeout, Brightness)` order community recipes pass positionally,
  and the wrong guess blanks the screen to 1%.
  `pythoncom.CoInitialize()`/`CoUninitialize()` bracket the live call (the
  worker-thread lesson pycaw taught, on a second bus). The parsers guard each
  other: `parse_volume_command` bails on "bright"/"dim" tokens (the field
  phrase "increase the brightness" parsed as volume-up — "increase" alone was
  enough) and `parse_brightness_command` requires one. Routing gives a
  brightness *word* the same outright 0.75 claim `VOLUME_WORDS` has
  (`IntentRouter.BRIGHTNESS_WORDS`): the field phrases "turn brightness down
  to 50"/"decrease the brightness to 50" scored 0.77/0.78 — the right intent,
  but under the 0.82 embedding band with no verbatim example to claim. Live
  level replies settle-poll ≤1 s for the applied value; a machine with no controllable
  display answers "I can't change the brightness on this display." `--check`
  prints `Brightness backend: wmi | unsupported`. Tests use the fake monitor
  in `tests/test_brightness.py` — never move the machine's real brightness.
- The HUD is the one component that owns a Qt thread. `NovaHUD.stop()` only
  *requests* a stop (`_stop_requested`, polled by a QTimer on the Qt thread);
  calling `app.quit()` from the caller's thread produced
  "QObject::killTimer: Timers cannot be stopped from another thread" on every
  exit. `_silence_benign_qt_messages()` drops exactly Qt's "QApplication was not
  created in the main() thread" advisory (the overlay cannot live on the main
  thread — that thread is in the sounddevice loop) and forwards every other Qt
  message. Ctrl+C is swallowed inside `run()` so the `finally` (stats + HUD
  shutdown) still runs, with a `KeyboardInterrupt` backstop in `main()`.
- Reminders (Wave 1c) live in `ContextEngine` (`reminders` table: `text`,
  `fire_at` unix seconds, `fired` flag) and are parsed by `parse_reminder`
  (`core/query_extractor.py`, which splits on the **last** " in " so a reminder
  may name a place, and reads spoken numbers / "an hour" / "half an hour"). The
  `set_reminder`/`manage_reminders` actions are deliberately **not** dry-run
  gated, exactly like `set_project`/`set_browser`: a SQLite memory row plus a
  later spoken reply are both dry-run-safe, so the default soak exercises them.
  Firing runs in `NovaAgent._consume_loop`, throttled to ~1s — on the worker
  thread that also speaks commands, so `is_speaking` is never toggled from two
  places and a reminder can never overlap a command — via `_fire_due_reminders`,
  which marks each row fired *before* `tts.speak` (barge-in applies; a synthesis
  failure cannot loop it) and is wrapped in try/except so a dead reminder can't
  kill the loop.

- Wave 2 LLM brain (`NOVA_LLM_BRAIN=1`, `core/llm_brain.py`) sits *behind* the
  untouched Tier 1 fast path: `observe` runs `router.match` first and only hands
  the phrase to the brain when Tier 1 resolved nothing (no confident match, no
  lexical claim). Tools are built from `intents.json` × `IMPLEMENTED_ACTIONS` —
  one tool **per intent**, named by *label* because actions aren't unique (both
  volume and brightness use `system_control`) — and the model selects through
  Ollama's native tool-calling; a call maps back to `intents[label]` and is
  re-gated by the `CapabilityRegistry`. Tools take **no arguments on purpose**:
  every action re-reads its details (app/query/level/reminder) from the
  transcript through the field-hardened parsers. No tool call = a conversational
  reply, streamed sentence-by-sentence (`take_sentences`) through `tts.speak`,
  so the Wave 1a chunked/bargeable playback applies; `run_observation` then
  skips the second `_respond` (a streamed reply is marked `llm_reply` and would
  otherwise play twice). Brain time is its own `llm` budget stage
  (`LATENCY_BUDGETS`; `format_stats` skips stages with no samples, so the
  default brain-off soak is unaffected). Every failure — no Ollama, a timeout,
  a hallucinated tool, a disabled action — degrades to an honest miss, never a
  guess. When the brain is on, `build_router` skips the Tier 2 string-parsing
  fallback (both compete for the same band; the brain wins).

## Testing quirks

- Tests inject fakes: `CommandProcessor(FakeSTT, FakeRouter, FakeTTS)` — follow that
  pattern; no test needs a microphone or GPU.
- Tests hard-code counts/names: `len(router.intents) == 20`, risky intents ==
  `{close_window, tidy_downloads, lock_workstation}` (both in `tests/test_phase0.py`).
  Adding or renaming an intent in `intents.json` requires updating those tests and
  `IMPLEMENTED_ACTIONS`.
- Wake/VAD tests **skip** (not fail) when the ONNX models are missing — a green
  suite does not prove the models exist. `tests/conftest.py` prints a loud
  "RUNTIME ASSETS MISSING" banner when that happens; check `--check` output
  for the per-backend reason.

## Runtime assets

`.onnx` wake/VAD models (`assets/wake_word/`), TTS cache (`assets/tts_cache/`),
`assets/voiceprint.npy` + `assets/speaker_model/` (the voice gate's enrolled
vector and its once-downloaded ECAPA model), and `memory/*.db` are git-ignored
and regenerate/download locally. `silero_vad.onnx`
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

## Waves (post-Wave 1)

### Wave 2 — LLM tool-calling brain
- **Trigger**: `NOVA_LLM_BRAIN=1`
- **Model**: Ollama tool-capable (phi4-mini / qwen2.5:7b)
- **Architecture**: Tool schemas from `intents.json` + `IMPLEMENTED_ACTIONS`; MiniLM fast path untouched; streaming sentence TTS reusing LA; new `llm` budget stage in `--stats`
- **Gate**: Only after Tier 1 measured accuracy is recorded
- **Status**: ✅ built, default-off — `nova_agent/core/llm_brain.py`,
  `tests/test_llm_brain.py`. Flip `NOVA_LLM_BRAIN=1` only after the Tier 1
  baseline is recorded, and only with a running Ollama that has a tool-capable
  model pulled (e.g. `ollama pull phi4-mini`). ("reusing LA" = the Wave 1a
  chunked/bargeable TTS.)

### Wave 3 — Screen vision Q&A
- **Model**: Ollama vision model
- **Builds on**: `screen_ocr` (existing)
- **Capability**: Ask about what's on screen

### Wave 4 — Smart home (Home Assistant)
- **New file**: `nova_agent/tools/home.py`
- **New intents**: HA domain
- **Config**: `NOVA_HA_URL` + `NOVA_HA_TOKEN` (long-lived token) via env only
- **Independent**: No model dependency; gated on HA credentials

### Wave 5 — Custom "hey dude" wake model
- **Deliverable**: `hey_dude.onnx` replacing the openWakeWord `hey_mycroft_v0.1.onnx` stand-in
- **Requirement**: Training-sample validation (your recordings + threshold check)

---

### Smaller items folded in opportunistically

- **Routines** — recurring / sequenced actions (the "routines" half of reminders & routines); deferred until reminders has field time.
- **CI workflow + typechecker** — no CI/pre-commit exists today; add when convenient.

---

### User-only validations

1. ~~One-hour soak with `--stats` → "Budget met on every stage"~~ **Tier 1
   baseline recorded** (53 commands, brain off): intent/tts/action/total all
   met budget; **stt over** (median 4.0s, max 11.5s — long noise segments +
   whisper-small on CPU). Field round 8 hardened the wake path (guards above);
   round 9 made **cuda the default** (`NOVA_STT_DEVICE`), but the 4-command
   re-soak still ran on cpu (launched directly, before the default flipped)
   and was over on stt/intent/total. **Re-soak pending** to confirm every
   stage meets budget.
2. ~~Barge-in echo field test~~ — worked live in the soak ("talked over the
   reply — playback stopped"), no self-trip on own echo seen.
3. ~~Reminder live check~~ — fired in the soak ("Reminder: stretch").
4. **False-wake + voice-gate field check** — `--enroll-voice` once more (round
   10 carves probe-length reference windows and reports a leave-one-out
   self-check at wake length), confirm the boot line reads
   `Voice gate: on (...)`, then listen for a day: expect
   `[wake] matched: '<text>' (voice 0.6x)` for real wakes and
   `[wake] ignored (voice mismatch 0.3x < 0.4)` for anything that isn't you —
   if *your* wakes get rejected, note the printed score (`NOVA_VOICE_THRESHOLD`
   tunes the line); re-run the soak (all stages under budget) to lift the
   Wave 2 gate.

---

### Sequence note

Waves 2 and 3 both need Ollama models (you may install both together); Wave 4 is independent of models but gated on HA credentials; Wave 5 needs audio samples. Waves 3–5 could run in any order once Wave 2's brain lands — suggested order: 2 → 3 → 4 → 5 unless you'd rather reorder.
