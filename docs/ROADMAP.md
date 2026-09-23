# Dude Roadmap (package `nova_agent`)

Status of the local voice agent, phase by phase. Phases are ordered by risk:
prove the loop, then make it reliable, then make it pretty.

Legend: **done** | **next** | **later**

Statuses reconciled against the code (Sep 2026). When this file and the code
disagree, trust the code.

## Phase 0 — Environment and baseline: done

- Python 3.11 venv at `.venv`, all runtime dependencies installed and verified.
- `pytest` and `ruff` (0.16.x, line-length 100) installed; suite green (65 tests).
- `.gitignore` covers `.kilo/`, `*.wav`, `memory/*.db`, wake models, TTS cache.

## Phase 1 — The live loop: done

- Audio callback only queues frames; a worker thread consumes them, so nothing
  heavy runs in the PortAudio callback and frames are not dropped.
- No self-triggering: frames captured while the assistant is speaking are discarded.
- Dry run is the default; `--live` / `NOVA_DRY_RUN=0` opts into real execution.
- Query extraction, screen OCR, orphan intents, fuzzy project lookup, LLM
  fallback contract, and in-repo openWakeWord feature models all wired.

## Phase 2 — Wake word and VAD quality: next

Already landed: Silero VAD loads into `VADRecorder` via `onnxruntime` (energy
heuristic is the fallback), ~0.7 s hangover before finalising, recordings written
under `memory/commands/` with a **unique name per capture** (never the working
directory), `--wake-probe` measures real scores and suggests a threshold.
**B9 observability:** the `_predict_score` failure path now logs
`logging.WARNING` + reason instead of swallowing, scoring uses the **last**
prediction-buffer frame (stale peaks no longer re-fire), `--wake-status` prints
the wake phrase + transcript-wake state, backend (`onnx`/`energy`), model path,
threshold, and last score, and `--calibrate` is a two-step wizard: (a) ambient
energy → energy-fallback threshold, (b) 5× "hey dude" → per-attempt transcript
HIT/miss through the real pipeline plus the sound model's score distribution.
Silence/echo guard: `process_chunk` refuses capture while `is_speaking`, and
`vad_min_speech_frames = 4` (4 × 80 ms = 320 ms ≥ 250 ms of
voice) before a command is accepted.

**Transcript wake (this pass) — the assistant now answers to "hey dude":**
the `hey_nova.onnx` file is byte-for-byte openWakeWord's `hey_mycroft_v0.1.onnx`
(it fires on "hey mycroft", never "hey dude"), so the primary trigger is text:
idle speech is segmented by `VADRecorder`, transcribed, and matched by
`core/wake_phrase.py` (normalized, word-boundary aware, remainder extraction so
"hey dude, open chrome" runs as one command via `handle_command(text=...)`).
Gated by `settings.transcript_wake` (env `NOVA_TRANSCRIPT_WAKE`, default on) and
only active when a real `processor.stt` exists. The audio model stays as the
fast secondary trigger; `wake_word` default is now `"hey dude"` (flag
`--wake-word` overrides). A `vad_max_seconds = 30` cap closes segments that
never find a pause so lecture-length speech cannot buffer unbounded. Visible
branding renamed Nova → Dude (package/classes still `nova_agent`/`Nova*`).
**Field finding + fix:** unprompted Whisper (small *and* base) systematically
misheard the user's mic saying "hey dude" as `'what are you doing?'`/`'i do it.'`
(rate, volume, and `vad_filter` ruled out by replaying the saved captures).
Wake transcription now primes `initial_prompt` with the phrase itself — it
transcribes those captures correctly and echoes the phrase on **0/22** non-wake
clips, silence, tone, and noise — plus an RMS gate
(`wake_segment_min_rms`) so near-silence never reaches the primed
decoder; `--wake-probe` reports gated attempts as "too quiet" instead of
hallucinated words. Floor field-tuned to `wake_segment_min_rms = 0.0015`
(ambient measured ~0.0011, quietest verified speech attempt 0.0020 — an
initial 0.004 floor rejected valid soft attempts). Wake now also prints
"Wake detected - listening for your command." even without `--debug`, since
a silent successful wake was indistinguishable from a dead loop.
**Live-loop root cause (found with the new diagnostics):** `_on_audio` queued
a *view* of PortAudio's reusable ring-buffer memory — the energy lines showed
real speech (0.04–0.08) while the frames the VAD held until finalization had
been overwritten by later callbacks, so live segments were written from
silence ("too quiet", or `''` transcripts in earlier runs). The probe worked
because blocking `stream.read()` returns fresh arrays. Fixed by copying in
the callback (`np.array(...)`); debug now prints segment duration/RMS and
dropped-frame counts and keeps unmatched segments as `idle_*.wav` for replay.
Two command-path bugs surfaced in the same successful wake run: `ContextEngine`'s
connection was main-thread-affine while commands run on the consumer thread
(`sqlite3.ProgrammingError` in `get_alias`) → connect with
`check_same_thread=False` (usage is sequential: the main thread touches context
only before the worker starts); and any uncaught command error killed
`_consume_loop` outright — the assistant looked alive but never responded — so
the consumer now catches per chunk, prints `Command failed (...)` plus the
traceback, and keeps looping.
**Silent-command visibility:** `run_observation`'s miss paths (empty text, or
no intent above the confidence band) returned `_respond("I didn't catch that")`
with no console output, and responses were TTS-only — a *cached* reply (`assets/tts_cache/`)
emits no synthesis warnings either, so a missed command looked exactly like a
dead loop while the assistant was actually replying. Every outcome now prints:
`Heard: (nothing intelligible)` / `Intent: none (best score ... below the
confidence band)`, and `_respond` echoes every reply as a `Response: ...` line.
**Retry-on-miss:** a miss inside `command_window` (re-armed from the *miss*,
not the wake) keeps the command turn open (`Still listening for your command.`)
instead of dropping to idle — field logs showed a background *video* winning
the first capture while the user's real command was still being spoken (its
transcript was pure video chatter); the real command then landed in the idle
path and was discarded for lacking the wake phrase. `command_max_turn` (30 s)
bounds the whole turn so background audio can't hold it. `_run_captured` now
returns the response so `_finish_turn` can decide between closing the turn and
retrying.
**Cold-start latency (found live):** `[debug] stages` reported
`intent=9716ms` / `command handled in 13.33s` on the session's first command —
`IntentRouter._prepare_embeddings` is lazy, so command #1 paid for
`import sentence_transformers` (torch included) + the MiniLM load; worse, that
latency outlived the original wake-time window, so the retry never fired on
exactly the command that needed it. Fixes: `IntentRouter.warm_up()` at listen
startup (gated by `intent_warmup`, prints `Intent warm-up: X.XXs`), the
re-armed window above, and `LLMFallback`'s previously dead `timeout` now
actually passed via `ollama.Client(timeout=...)` so a cold phi4-mini can't
stall the stage either. The `dropped frames: N` counter on idle segments is the
worker-busy blind spot: mic frames arriving while a command is processing are
queue-dropped by design (echo guard), so shortening the pipeline is what
shortens the deaf window.
**Field results (both validations observed live):** intent fell 9716ms →
9-46ms after warm-up, and the retry chain ran end to end in `--stats`
(`Heard: open from.` → miss → `Still listening for your command.` →
`Heard: open chrome.` → `Intent: open_app (0.75)`). Synchronous warm-up then
proved the wrong trade — boot paid 21-45s for it — so intent *and* Kokoro now
warm on a daemon thread at listen startup (`(background)` suffixed prints,
listener up immediately), with `_prepare_embeddings` under a double-checked
lock so the warm thread and the first live match can't build two encoders.
`TTSEngine.warm_up()` prepays the 21.5s first-synthesis tax (import + model
load + torch JIT). `LATENCY_BUDGETS` recalibrated from the field data:
stt 2.5s (observed ~1.7s), intent 0.1s (~30ms), tts 4.0s (~2.1s warm — the
50ms figure measured playback-inclusive time against a cache-only guess),
action 0.5s (~1ms), total 7.0s.
**Field round 2 (13 commands):** background warm confirmed — boot immediate,
`Intent warm-up: 12.66s (background)`, `TTS warm-up: 5.97s (background)` —
and the retry chain fired 6× (`open chrome`, `current time`, `i read the
screen`, `open my project` all landed with intent 0.75–0.89). The remaining
miss class is **command mishears on healthy audio**: every saved capture was
measured, real commands sit at rms 0.02–0.056 / peak ~0.23 while junk
fragments sit at 0.003–0.010 — but historically-valid soft wakes measured
0.002, so a command quiet-gate would overlap real speech and falsely reject
it; skipped deliberately (the retry absorbs junk, and the fix targets the
healthy-audio mishears). That fix is `COMMAND_PROMPT`: the wake phrase's
proven priming pattern, pointed at the intent vocabulary, applied at all
three command-slot transcription sites (`handle_command`,
`CommandProcessor.process`, confirmation replies). A warm-vs-speak race was
also found — two `repo_id` warnings meant two KPipeline builds — so the whole
of `_synthesize` now runs under `_pipeline_lock`, and it reports
`last_synthesis_seconds` (time-to-audio) which `_respond` prefers for the tts
stage: playback duration is reply length, not performance. Budgets
recalibrated a second time: stt 3.0 / intent 0.2 / tts 2.5 / action 0.5 /
total 15.0 (wall clock, includes speaking the reply — max observed 13.8s).
**Field round 3 (11 commands) — the prompt worked, three new bugs surfaced:**
`COMMAND_PROMPT` verified in the field (`mute the volume` → `system_control`
0.91, `open python project folder` correctly transcribed, `read the screen`
0.91; the retry recovered `current ime` → `current time`). New findings, all
fixed: **(a)** `Heard: i.` → `open_app (0.75)` → "Would open code" — a
one-character fragment substring-matched the example "launch visual studio
code" (the letter *i*) and claimed VS Code from pure noise; `_lexical_fallback`
now refuses normalizations shorter than 2 chars and matches examples on word
boundaries only. **(b)** the "out-of-the-box" gap: `open gallery` scored 0.46
→ miss, because only chrome/code were ever launchable. Now a launch verb plus
an unknown app claims `open_app` lexically (project/volume/example shapes
still win first), `extract_app_phrase` keeps the trailing phrase for unknown
apps, and `tools/app_controller.resolve_app` resolves it through PATH
(System32: notepad, calc) then Start Menu `.lnk` shortcuts (exact → prefix →
whole word), or the dispatcher answers honestly ("I don't know how to open X
yet."). **(c)** `bye` scored 0.38 → miss; greeting examples now include
bye/goodbye/see you later, and greet replies with a farewell for them.
**(d)** the budget table's outliers were all explainable: intent max 1986ms /
tts max 4757ms were command #1 blocking on the warm thread's lazy loads (by
design), and total max 24843ms was Kokoro reading a full Windows path aloud
("Would open code (C:\Users\...)") — `--stats` now skips commands handled
during warm-up (`warm_done`, printed when it happens), and `_respond` speaks
`speakable(response)` (paths stay console-only), so the verdict reflects
steady state.
**Field round 4 (13 commands) — out-of-box opening works; two stats bugs
found:** `open notepad` resolved through PATH, `bye` → farewell, the retry
recovered `enter time` → `time`, `mute the volume` → 0.91, and `speakable()`
kept paths out of speech (tts max 1767ms). New findings, all fixed:
**(e)** `open calculator` resolved to nothing (no `Calculator.lnk`, and
`calc.exe` ≠ "calculator") — `RESOLVE_ALIASES` in `app_controller` maps the
spoken name to the executable; **(f)** `--stats` *still* counted the
warm-blocked command #1 (intent 19088ms, total 33010ms): its TTS blocked on
the warm-up's synthesis lock, so it finished after `warm_done.set()` and an
end-only check passed — the skip verdict is now taken at command **start**;
**(g)** `processor.timings` was never reset per command, so the wake-segment
transcription's `stt` leaked into text-path command rows (the 10888ms
outlier) — `_run_captured` clears timings at entry. Rerun pending for the
clean "Budget met on every stage" verdict.

Remaining:

- Field-tune via `--calibrate` (expect transcript HIT counts, not model scores);
  document wake-model regeneration (`*.onnx` is git-ignored). Optionally train a
  real "hey dude" model later (openwakeword training) to restore a fast neural
  secondary trigger on the intended phrase.

__Done when:__ 20/20 wake attempts trigger on first try, 0 false wakes over
10 minutes of music/talking, "open chrome" said quietly still registers.

## Phase 3 — Latency and snappiness: next

Already landed: `--device {cpu,cuda}` + `NOVA_STT_DEVICE`, startup warm-up with
the cuBLAS→CPU fallback tested, per-stage timers (`stt`/`intent`/`tts`/`action`)
printed under `--debug`, TTS cached-prefix composition (`_cached_prefix_split`).
**`--stats`** records a per-command latency snapshot (`stt`, `intent`, `tts`,
`action`, `total` — the `tts` stage is newly timed in `_respond`) and on session
exit prints min/median/mean/max per stage against `LATENCY_BUDGETS`, with a
worst-case verdict (`ok` / `over by Nms` / `Budget EXCEEDED on: ...`). The
verdict deliberately uses the max, not the mean.

Remaining:

- One clean real-mic validation session (`python -m nova_agent --listen --stats`)
  of ~10 consecutive commands against `LATENCY_BUDGETS` (stt ≤ 3 s, intent ≤
  200 ms, tts ≤ 2.5 s, action ≤ 500 ms, total ≤ 15 s). Rounds 3 and 4 ran, but
  both worst-case verdicts were pinned by bugs fixed right after (round 3:
  warm-window commands counted, path-reciting replies; round 4: the warm skip
  checked only at command end and stage timings leaked across commands) —
  rerun for the clean "Budget met on every stage" proof the tooling exists
  for.

## Phase 4 — Context engine depth: done

Already landed: `app_aliases` table + `resolve_app`, `find_project` fuzzy/contains
lookup, "do that again" (`repeat_last` on `get_last_command`), time-of-day
greetings from cached phrases, `set browser to …` persisted across restarts.

Also landed this pass:

- **Multi-turn confirmation window** (`handle_command`): a reply that is not
  yes/no is tried as a *new* command — "close the window" → "no, open chrome"
  runs the replacement instead of discarding the reply. Bounded by
  `MAX_COMMAND_SWAPS = 2` so destructive prompts cannot ping-pong; silence,
  rejection, and unroutable speech still cancel. `_await_confirmation` now
  returns the raw reply text and the caller classifies it.
- **`set_preference` intent** (intents.json is now 15 intents): wake threshold
  (0..1, validated), dry-run on/off ("enable dry run" / "go live"), TTS voice
  ("set voice to af heart" → `af_heart`, shape-validated). Applied live —
  threshold through the injected wake engine, `tts.voice`, `processor.dry_run`
  — and persisted in the `preferences` table. Startup precedence is CLI flag >
  spoken preference > env default (`resolve_dry_run`, `stored_preference_float`
  in `main.py`; NovaAgent reads `wake_threshold`/`tts_voice` at construction).

## Phase 5 — HUD overlay: done

- Real PyQt5 overlay in `ui/overlay.py`: frameless, translucent, always-on-top,
  never takes focus (`WindowDoesNotAcceptFocus` + `WA_ShowWithoutActivating`),
  four states + confirming/error, pulse-free but live transcript ribbon.
- Thread-safe by design: worker writes plain data behind a lock, the window polls
  a snapshot on a Qt timer — no cross-thread widget access.
- `--no-hud` keeps headless runs and `tests/test_hud.py` working.

## Phase 6 — Safety, exercised: done

Already landed: spoken confirmation loop (5 s timeout, rejection words checked
first, unclear → cancel), three destructive intents behind
`NOVA_ALLOW_DESTRUCTIVE=1` + confirmation, `CapabilityRegistry` wired as the
dispatcher gate, Recycle Bin via `send2trash` (never `os.remove`).

Also landed this pass:

- **Guard layer** — `nova_agent/tools/guards.py`: `ensure_inside(root, candidate)`
  fully resolves both sides (defeats `..`, symlinks, NTFS junctions) and raises
  `GuardViolation` otherwise. `tests/test_tool_guards.py` AST-scans every file
  in `tools/` and fails the build on `os.remove`/`os.unlink`/`Path.unlink`/
  `shutil.rmtree`/`os.system`, `subprocess(..., shell=True)`, or a bare-string
  subprocess command (list-formed `Popen` stays legal); a planted-violation
  test proves the scanner itself reports with file:line.
- **Path allow-list in practice** — `move_latest_download_to_recycle_bin` runs
  the candidate through `ensure_inside` *before* the dry-run branch and before
  `send2trash`, so a link escaping Downloads is refused ("I left it alone")
  rather than followed or trashed.
- **Confirmation cannot be bypassed** — `tests/test_runtime_safety.py`: a spy
  on `CommandProcessor.execute` asserts an unsafe action never executes
  unconfirmed, and a second test asserts a spoken confirmation still gets
  "not enabled yet" without `NOVA_ALLOW_DESTRUCTIVE=1`.

Note on "double-confirm": nothing in the toolset is truly irreversible (Recycle
Bin only; close/lock are reversible), so the second barrier is the env-flag
capability gate, and both barriers are asserted by the tests above.

__Done when:__ ✅ "close the active window" asks, "confirm" executes, "no"
cancels, and an unsafe action cannot bypass confirmation (asserted by test).

## Phase 7 — Tier 2 LLM fallback: done

Already landed: strict function-call parsing, allow-listed actions/targets,
3 s timeout, graceful `unknown` when Ollama is down.

Also landed this pass:

- **Confidence band** — Tier 2 now only answers when
  `settings.llm_min_score` (0.65, env `NOVA_LLM_MIN_SCORE`) ≤ score <
  `intent_threshold` (0.82). Below the floor the fallback is never consulted
  and the dispatcher re-asks with "I didn't catch that"; at/above the
  threshold Tier 1 has already claimed the utterance; lexical claims still
  preempt the LLM.
- **Registry validation** — `LLMFallback` now receives the dispatcher's own
  `CapabilityRegistry` (`build_router(settings, capabilities)`) and refuses to
  propose any action that gate would reject. The model is configurable via
  `NOVA_LLM_MODEL` (default `phi4-mini`).

Environment, decided and installed: the `ollama` Python client (0.6.2) is in
the venv and pinned in `requirements.txt`; the fallback model is
**phi4-mini** (2.5 GB, MIT, ~2.8 GB VRAM at Q4_K_M — leaves ~3 GB of the
RTX 3050's 6 GB for Whisper/Kokoro, and `keep_alive=5m` unloads it when
idle). The compiled default stays `NOVA_LLM_FALLBACK=0`; `start.bat` sets it
to `1` before launching.

__Done when:__ ✅ live check against the running daemon: "search the web for
python decorators" → `browser_search("python decorators")`, "open chrome" →
`open_app("chrome")`, "what should I have for breakfast" → `unknown`;
`python -m nova_agent --check` reports `Tier 2 fallback: on`.

## Phase 8 — Polish, tests, docs: next

Already landed: README current (`--listen/--calibrate/--debug/--wake-probe`,
`NOVA_TESSERACT_PATH`, `NOVA_CHROME_PATH`, wake-model notes), `docs/ROADMAP.md`
exists, ruff installed and the whole tree passes `ruff check .` (broad-except
and local-time patterns carry explicit `noqa: ... -- reason`), `RuntimeMonitor` fed by `_record_metrics`.

Also landed this pass:

- **Dead scaffolding deleted** — `CommandQueue` and `AudioSession` (plus their
  tests) removed: both were Phase 0/1 experiments never imported by the app.
  The production loop already has the real thing — `NovaAgent.audio_queue`
  (bounded `queue.Queue`, drop-oldest, worker thread) and the HUD state machine
  in `ui/overlay.py`. Tracked in git history if ever wanted back.
- **Model-missing runs are now loud** — `tests/conftest.py` prints a
  "RUNTIME ASSETS MISSING" banner listing absent `hey_nova.onnx`/
  `silero_vad.onnx` and the skip count, so a green suite can no longer
  silently imply the wake/VAD backends were exercised. `--check` remains the
  per-backend detail report.
- **Transcript-wake test coverage** — `tests/test_wake_phrase.py`: matcher
  units (normalization, word boundaries, remainder extraction), agent
  integration (idle wake, no-phrase stays idle, one-shot tail runs as the
  command, `NOVA_TRANSCRIPT_WAKE=0` disables, processors without STT never
  segment idle audio, a failing idle transcription doesn't kill the loop),
  prompt plumbing (`wake_prompt_for`, quiet-segment gate, `STTEngine`
  → `initial_prompt` forwarding), and the `vad_max_seconds` cap.

Remaining:

- One-hour soak: real mic, wake→command cycles; watch RSS, VRAM, cache size, the
  SQLite file; confirm zero dropped audio and no unbounded growth.

## Phase 9 — Command understanding (field round 5): landed

A live `--listen --stats` session showed three misses where the pipeline *heard*
the right words and the wiring lost them:

    Heard: open.               -> Intent: open_app (0.75) -> "Would open chrome"
    Heard: so, i run the time. -> Intent: none (best score 0.51)
    Heard: name.               -> Intent: none (best score 0.44)

Also observed: Ctrl+C printed a `KeyboardInterrupt` traceback after the stats
table, and every shutdown printed `QApplication was not created in the main()
thread` plus `QObject::killTimer: Timers cannot be stopped from another thread`.

- **A bare launch verb asks instead of guessing** — `_lexical_fallback`'s
  example loop matched whole words *both ways*, so `\bopen\b` matched inside the
  example "open chrome" and any transcript that collapsed to one verb launched
  the browser. `LAUNCH_WORDS`-only normalizations now return `open_app` with no
  target, and `_open_app` answers `OPEN_WHAT_RESPONSE` ("Open what?"). The turn
  stays open (`RETRY_RESPONSES`) and `pending_clarification` makes the answer
  ("notepad") dispatch as `open notepad` — the reference behaviour from
  isair/jarvis: never resolve to a sub-item without its parent noun.
- **A time word is enough** — `TIME_WORDS` (`time`/`clock`) claims `time_check`
  at 0.75, placed *after* the example loop so "it's time to open chrome" is
  still an open command.
- **Identity and help intents** — "what's your name"/"who are you" and
  "help"/"what can you do" (18 intents now; `length == 18` asserted in
  `tests/test_phase0.py`). The help reply is generated from the capability
  registry, so a disabled destructive action is never advertised.
- **Misses name the runner-up** — `IntentRouter.match` stores
  `last_best_label`, so the console reads
  `Intent: none (best score 0.51 for time_check, below the confidence band)`.
- **STT name vocabulary** — the command slot now passes faster-whisper
  `hotwords` built from the app words plus remembered aliases (`stt_hotwords()`).
  Technique taken from Home Assistant's Whisper server (bias toward the names
  the user says; its author measured that irrelevant names do not hurt general
  transcription). Deliberately **not** applied to wake or confirmation
  captures — a "yes" biased toward app names would cancel a confirmation.
- **Bounded fuzzy app resolution** — after PATH and Start Menu (exact → prefix →
  whole word), `resolve_app` tries a closest-name match with a 4-character floor
  and 0.82 similarity ("notepadd" → Notepad, "notes" → still refused). The
  edit-distance-as-a-budget idea comes from Home Assistant's Hassil matcher.
- **Clean shutdown** — Ctrl+C is swallowed in `NovaAgent.run()` so the `finally`
  still prints stats and closes the HUD, with a `main()` backstop; `NovaHUD`
  requests a stop from the caller and performs it on the Qt thread, and the
  known-benign Qt advisory is filtered while every other Qt message passes
  through.
- **The phrasebook (follow-up to the same session's `--stats` verdict)** — the
  first live run after these fixes ended with `Budget EXCEEDED on: tts`
  (2669ms against 2500ms). The cause was content, not speed: the 15-word
  "Screen reading is unavailable …" reply was re-synthesised on *every* use,
  because `should_cache` (≤6 words, no digits) refuses long phrases. Fixed by
  splitting the two cases the policy was conflating: `TTSEngine.preload()`
  stores the *canonical* replies (`CommandProcessor.canned_replies()`) whatever
  their length during the background warm-up, which now prints
  `TTS warm-up: 6.2s (background, N replies preloaded)`. `speak` reads the cache
  before consulting the policy, so preloaded replies report
  `last_synthesis_seconds == 0` and the tts stage drops to ~0ms. Preload skips
  entries already on disk, and `TESSERACT_MISSING_MESSAGE` is now single-sourced
  so the cached key cannot drift from the spoken text.

- **Round-5 follow-up: volume, pronouns, the voice cache** —
  **Volume** left media-key nudging for Core Audio (`pycaw`):
  `tools/system.py` parses the full command shape (unmute matched before
  mute, spoken levels like "fifty percent", directions) and applies it with a
  read-back ("Volume set to 30 percent."), real mute/unmute state, and a 10%
  step for up/down — the media keys remain an honest fallback ("Toggled mute";
  exact levels refused without pycaw) and `--check` reports the live backend.
  **Pronouns**: "open it"/"open that" asks "Open what?" through
  `is_pronoun_target` instead of answering "I don't know how to open it
  yet.", joining the same clarification turn a bare launch verb gets.
  **Voice cache**: the TTS cache is keyed by phrase, not voice, so
  "set voice to …" now purges it and re-preloads the phrasebook on a
  background thread (`_recache_replies`) — otherwise every canned reply would
  keep playing in the old voice forever.

- **Round-5 follow-up, part 2: brightness** — the field log
  `Heard: increase the brightness` scored 0.42 for system_volume — correctly
  refused, but the parser underneath was a trap: `parse_volume_command`
  returned volume-up ("increase" alone was enough). There is now a
  `system_brightness` intent (target `brightness`, 18 intents total) sharing
  the `system_control` action; `_control_system` picks the device from the
  intent target. `apply_brightness` drives the display's `root\WMI`
  interface through `wmi`/`pywin32` (~50 ms — a PowerShell subprocess would
  have burned the whole 500 ms action budget), calls `WmiSetBrightness`
  **by keyword** after the live MOF proved this machine declares
  `(Brightness, Timeout)` — the reverse of what community recipes pass
  positionally — brackets COM with `pythoncom.CoInitialize()`/
  `CoUninitialize()` (the worker-thread lesson pycaw taught), and reads the
  applied level back through a ≤1 s settle poll. Both parsers now refuse
  each other's phrases, a display without WMI control answers "I can't
  change the brightness on this display.", `--check` gained a
  `Brightness backend:` line, and `tests/test_brightness.py` runs everything
  against a fake monitor. The first mic run of these phrases then found the
  embedding band's dead zone: "turn brightness down to 50" and "decrease the
  brightness to 50" scored 0.77/0.78 for system_brightness — clearly the
  right intent, but under the 0.82 threshold with no verbatim example inside
  the phrase — so `IntentRouter` gained `BRIGHTNESS_WORDS`, the same outright
  0.75 claim `VOLUME_WORDS` already had. Parse-side needed no change: an
  explicit level beats the direction, so "down to 50" sets 50 (the volume
  contract too).
- **Round-6 (field): shortcuts actually launch** — `open microsoft edge` and
  `open file explorer` routed correctly (`open_app` 0.75) and then died with
  `WinError 193 "%1 is not a valid Win32 application"`: the resolver returns
  Start Menu `.lnk` files for both — and for `chrome` on this machine, which
  has no `chrome.exe` in Program Files — but `open_path` handed them to
  `subprocess.Popen`, and CreateProcess cannot execute shortcuts. Shortcuts
  and folders now launch through `os.startfile` (ShellExecute, which also
  keeps a shortcut's own arguments), executables keep Popen with the shell as
  an OSError fallback, and a double failure answers "I couldn't open X."
  instead of raising into the listen loop. Dry-run had hidden this: every
  earlier field round stopped at "Would open chrome".
- **Round-7 (field): compound commands + long names** — `increase brightness
  and open microsoft edge` raised the brightness and silently dropped the
  launch: the `BRIGHTNESS_WORDS` claim took the whole sentence. `observe` now
  splits on conjunctions (`split_compound`, capped at 3 clauses) and matches
  clause by clause; only *two* matching clauses stand as a compound, otherwise
  the whole phrase routes exactly as before. Follow-ups queue in
  `pending_compound` and run in turn through `run_observation`, pausing behind
  any spoken confirmation (a rejection drops them with the pending action);
  `join_replies` merges the per-clause answers into one spoken response, and
  `--stats` sums their action time. The same log's "you need to add a long
  name" landed as multi-word Start Menu names (`multi_word_shortcut_names`)
  biased into the command slot's `stt_hotwords()` — long names are the ones
  Whisper mangles; single-word stems stay out of that prompt budget.

Verification: `python -m pytest` 244 passed (+7 in `tests/test_compound.py`,
+16 in `tests/test_brightness.py`, +5 launch tests in
`tests/test_app_resolution.py`; the round-5 follow-up before it added
`tests/test_volume.py` and the pronoun/recache coverage), the intent
count/membership asserted at 18 in `tests/test_phase0.py`, `python -m ruff
check .` clean.
