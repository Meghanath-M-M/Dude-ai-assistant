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

- One real-mic validation session (`python -m nova_agent --listen --stats`) of
  10 consecutive commands on the LOQ (RTX 3050 6 GB): STT ≤ 1200 ms, intent ≤
  100 ms, cached TTS ≤ 50 ms, action ≤ 500 ms, **total ≤ 2 s** — the tooling now
  proves the budget with data instead of vibes.

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
