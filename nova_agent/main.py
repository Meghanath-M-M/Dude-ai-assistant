import argparse
import importlib.util
import os
import queue
import re
import subprocess
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from nova_agent.config.settings import (
    CACHE_DIR,
    COMMAND_DIR,
    WAKE_WORD_MODEL_PATH,
    Settings,
    ensure_directories,
)
from nova_agent.core.barge import BargeGate
from nova_agent.core.capabilities import CapabilityRegistry
from nova_agent.core.context_engine import ContextEngine
from nova_agent.core.intent_router import IntentRouter
from nova_agent.core.query_extractor import (
    APP_WORDS,
    extract_app_phrase,
    extract_project_name,
    extract_search_query,
    humanize_delay,
    is_pronoun_target,
    looks_like_open_command,
    parse_reminder,
    parse_set_browser,
    parse_set_preference,
    parse_set_project,
    split_compound,
)
from nova_agent.core.runtime_monitor import RuntimeMonitor
from nova_agent.core.safety import (
    CONFIRM,
    REJECT,
    confirmation_decision,
    confirmation_prompt,
    requires_confirmation,
)
from nova_agent.core.vad import VADRecorder
from nova_agent.core.voice_gate import VoiceGate, enroll_voice
from nova_agent.core.wake_phrase import wake_phrase_hits
from nova_agent.core.wake_word import WakeWordEngine
from nova_agent.tools.app_controller import (
    APP_PATHS,
    RESOLVE_ALIASES,
    multi_word_shortcut_names,
    open_app,
    open_path,
    resolve_app,
)
from nova_agent.tools.browser import search_web
from nova_agent.tools.screen_reader import TESSERACT_MISSING_MESSAGE
from nova_agent.ui.overlay import NovaHUD

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional at runtime for tests.
    sd = None


def time_of_day_greeting(now: datetime | None = None) -> str:
    """Pick a greeting that matches the cached Kokoro phrases."""
    hour = (now or datetime.now()).hour  # noqa: DTZ005 -- local wall clock
    if 5 <= hour < 12:
        return "Good morning"
    if 12 <= hour < 17:
        return "Good afternoon"
    return "Good evening"


def build_router(settings: Settings, capabilities=None) -> IntentRouter:
    """Create the Tier 1 router, optionally with the Tier 2 LLM fallback.

    The Wave 2 brain (``NOVA_LLM_BRAIN=1``) supersedes this string-parsing
    fallback: both would compete for the same below-threshold band, and the
    brain's native tool-calling is strictly the better of the two.
    """
    router = IntentRouter(settings.intent_threshold, fallback_floor=settings.llm_min_score)
    if settings.llm_fallback and not settings.llm_brain:
        from nova_agent.core.llm_fallback import LLMFallback

        router.fallback = LLMFallback(
            model=settings.llm_model,
            timeout=settings.llm_timeout,
            capabilities=capabilities,
        )
    return router


# Every action the dispatcher can actually perform. Anything outside this set is
# reported as unavailable instead of quietly doing nothing.
IMPLEMENTED_ACTIONS = (
    "open_app",
    "browser_search",
    "screen_ocr",
    "context_open",
    "system_control",
    "greet",
    "time_check",
    "repeat_last",
    "identity",
    "help",
    "set_project",
    "set_browser",
    "set_preference",
    "set_reminder",
    "manage_reminders",
    "close_window",
    "tidy_downloads",
    "lock_workstation",
)

# Spoken descriptions for the "help" intent, keyed by action. Prose-sized on
# purpose: TTS reads every word of the reply, and the list is capped.
ACTION_LABELS = {
    "open_app": "open apps",
    "browser_search": "search the web",
    "screen_ocr": "read the screen",
    "context_open": "open saved projects",
    "system_control": "control the volume and brightness",
    "greet": "say hello",
    "time_check": "tell the time",
    "repeat_last": "repeat the last command",
    "identity": "tell you who I am",
    "help": "list what I can do",
    "set_project": "remember projects",
    "set_browser": "switch browsers",
    "set_preference": "change my settings",
    "set_reminder": "set reminders",
    "manage_reminders": "list or cancel reminders",
    "close_window": "close windows",
    "tidy_downloads": "tidy downloads",
    "lock_workstation": "lock the workstation",
}
MAX_HELP_ITEMS = 7

# The assistant's spoken identity, and what it says when asked what it can do.
IDENTITY_RESPONSE = "I'm Dude, your local assistant. Nothing you say leaves this machine."

# Actions that change or destroy something. They stay disabled unless
# NOVA_ALLOW_DESTRUCTIVE=1, and always require a spoken confirmation.
DESTRUCTIVE_ACTIONS = ("close_window", "tidy_downloads", "lock_workstation")

# How many times a confirmation reply may be swapped for a new command before
# the exchange is abandoned: two destructive prompts must not ping-pong forever.
MAX_COMMAND_SWAPS = 2


def build_registry(settings: Settings | None = None) -> CapabilityRegistry:
    """Describe what the assistant may do, and what needs confirming."""
    settings = settings or Settings()
    return CapabilityRegistry.from_actions(
        IMPLEMENTED_ACTIONS,
        DESTRUCTIVE_ACTIONS,
        allow_destructive=settings.allow_destructive,
    )


class CommandProcessor:
    """Turn a transcript into an action, with safety and context applied.

    Split into three steps -- ``transcribe``, ``observe``, ``run_observation`` --
    because a destructive action needs a spoken confirmation *between* choosing
    the action and running it.
    """

    def __init__(
        self,
        stt,
        router,
        tts,
        dry_run: bool = True,
        context=None,
        settings=None,
        capabilities=None,
        monitor=None,
        wake_engine=None,
    ):
        self.stt = stt
        self.router = router
        self.tts = tts
        self.dry_run = dry_run
        self.context = context
        self.settings = settings or Settings()
        self.capabilities = capabilities
        self.monitor = monitor
        # Optional: lets "set wake threshold to 0.6" take effect immediately.
        self.wake_engine = wake_engine
        self.pending_confirmation: dict | None = None
        self.pending_text: str = ""
        # Set when a launch verb named no app ("open."): the next capture is
        # read as the missing target instead of as a fresh command.
        self.pending_clarification: str | None = None
        # Follow-up clauses of a compound command ("...and open microsoft
        # edge"), queued by observe() and run by run_observation() after the
        # head clause — or after the head's spoken confirmation, if the head
        # needs one. Dropped together with a rejected/pending turn.
        self.pending_compound: list[dict] = []
        # Long Start Menu names for decode biasing: scanned once per session
        # (the aliases below must take effect immediately, these need not).
        self._shortcut_names: set[str] | None = None
        self.last_intent: dict | None = None
        self.last_text: str = ""
        # The referent for a follow-up "open it"/"open that": the last app or
        # project actually opened. Kept separate from last_text, which the
        # pronoun command itself would otherwise overwrite (losing the target).
        self.last_open: dict | None = None
        self.timings: dict[str, float] = {}
        # Wave 2 brain: built only when NOVA_LLM_BRAIN=1 (default off, pending
        # the Tier 1 baseline). Its tools come from this router's intents
        # crossed with the dispatcher's runnable actions; the capability
        # registry gates whatever the model proposes.
        self.llm_brain = None
        if self.settings.llm_brain:
            from nova_agent.core.llm_brain import LLMBrain

            self.llm_brain = LLMBrain(
                model=self.settings.llm_model,
                timeout=self.settings.llm_timeout,
                capabilities=self.capabilities,
                intents=getattr(self.router, "intents", {}),
                implemented=IMPLEMENTED_ACTIONS,
            )

    def transcribe(
        self,
        audio_path: str | Path,
        prompt: str | None = None,
        *,
        command: bool = False,
    ) -> str:
        """Transcribe one capture.

        ``command`` marks the command slot and adds the app-name vocabulary as
        a decode hint. Wake segments and confirmation replies must not get it:
        biasing a "yes" toward "notepad" is how a confirmation gets rejected.
        """
        started = time.perf_counter()
        text = self.stt.transcribe(
            str(audio_path),
            prompt=prompt,
            hotwords=self.stt_hotwords() if command else None,
        )
        self.timings["stt"] = time.perf_counter() - started
        return text

    def stt_hotwords(self) -> str | None:
        """Spoken app names to bias the command decode toward, or ``None``.

        Field evidence: Whisper-small turned clear, loud command audio into
        "open by 10 projects" and "we hope the volume". Names are the worst
        offenders because the model has never seen them, so the vocabulary is
        the set of names *this* install understands: the built-in app words,
        the resolver aliases, and whatever the user taught with "set browser
        to ...". Cheap enough to rebuild per command (one small SELECT), which
        is what lets "set browser to edge" take effect immediately. Long names
        ("microsoft edge", "file explorer") are the ones Whisper mangles, so
        they join from a once-per-session Start Menu scan — too much work to
        repeat for every command.
        """
        names = set(APP_WORDS) | set(RESOLVE_ALIASES)
        if self._shortcut_names is None:
            self._shortcut_names = multi_word_shortcut_names()
        names |= self._shortcut_names
        if self.context is not None:
            for alias, app_key in self.context.list_aliases():
                names.update({alias, app_key})
        names.discard("")
        return ", ".join(sorted(names)) or None

    def observe(self, text: str) -> dict:
        """Decide what was asked, without doing anything about it.

        A compound utterance ("increase brightness and open microsoft edge")
        is matched clause by clause and kept as a compound only when *two*
        clauses claim an intent; one match falls back to routing the whole
        phrase, so a conjunction attached to a real command stays one
        command. The follow-up clauses wait in ``pending_compound`` for
        ``run_observation``.
        """
        # A fresh observation replaces any queue left from an older command.
        self.pending_compound = []
        text = (text or "").strip()
        started = time.perf_counter()
        matched: list[dict] = []
        clauses = split_compound(text)
        if len(clauses) > 1:
            for clause in clauses:
                intent, score = self.router.match(clause)
                if intent is not None:
                    matched.append({"text": clause, "intent": intent, "score": score})
        if len(matched) >= 2:
            head, *follow_ups = matched
            self.pending_compound = follow_ups
        else:
            # One matching clause is not a compound: route the whole phrase
            # exactly as before (any partial match above is discarded).
            intent, score = self.router.match(text)
            head = {"text": text, "intent": intent, "score": score}
        # Tier 1's time only — the brain below gets its own `llm` stage, so a
        # slow model can never masquerade as a slow embedding match.
        self.timings["intent"] = time.perf_counter() - started
        if head["intent"] is None and self.llm_brain is not None:
            self._brain_decide(head)
        return head

    def _brain_decide(self, head: dict) -> None:
        """Let the Wave 2 brain resolve a phrase Tier 1 could not (mutates ``head``).

        A tool call becomes that intent, executed normally downstream. A
        conversational reply is streamed sentence by sentence through the
        chunked, bargeable TTS *here*, and ``head`` is marked ``llm_reply`` so
        ``run_observation`` won't speak it a second time.
        """
        text = head.get("text", "")
        speak_seconds = 0.0

        def _speak(sentence: str) -> None:
            nonlocal speak_seconds
            call_started = time.perf_counter()
            self.tts.speak(speakable(sentence))
            # The same synthesis-vs-playback rule _respond uses: an engine that
            # self-reports wins, fakes (and engines without the marker) keep the
            # whole — short — call.
            synth = getattr(self.tts, "last_synthesis_seconds", None)
            speak_seconds += (
                (time.perf_counter() - call_started) if synth is None else synth
            )

        started = time.perf_counter()
        decision = self.llm_brain.decide(text, on_sentence=_speak)
        self.timings["llm"] = time.perf_counter() - started
        if decision.tool_intent is not None:
            head["intent"] = decision.tool_intent
            head["score"] = decision.score
        elif decision.reply_text is not None:
            head["intent"] = {
                "action": "llm_reply",
                "safe": True,
                "reply": decision.reply_text,
            }
            head["score"] = decision.score
            self.timings["tts"] = speak_seconds
        # else: leave the miss alone — the honest "I didn't catch that".

    def respond(self, message: str) -> str:
        """Speak a message without executing anything (used for cancellations)."""
        return self._respond(message)

    def process(self, audio_path: str | Path, confirmed: bool = False) -> str:
        """Transcribe and run one command (kept for the one-shot path)."""
        observation = self.observe(
            self.transcribe(audio_path, prompt=COMMAND_PROMPT, command=True)
        )
        return self.run_observation(observation, confirmed=confirmed)

    def run_observation(self, observation: dict, confirmed: bool = False) -> str:
        """Run one observed command, then any queued compound follow-ups.

        Clauses run in turn; the moment one needs a spoken confirmation the
        rest wait in ``pending_compound`` — the confirm turn picks the queue
        back up (the confirmed clause runs first), while a rejection or a
        fresh command drops it along with the pending action. Action time is
        summed across clauses so ``--stats`` still sees the whole utterance
        against the budget. Exactly one ``_respond`` per call, so a compound
        is spoken as a single joined reply.
        """
        total_action = 0.0
        response = self._run_one(observation, confirmed)
        total_action += self.timings.pop("action", 0.0)
        while self.pending_compound and self.pending_confirmation is None:
            follow_up = self.pending_compound.pop(0)
            response = join_replies(response, self._run_one(follow_up))
            total_action += self.timings.pop("action", 0.0)
        if total_action:
            self.timings["action"] = total_action
        head_intent = observation.get("intent") or {}
        if head_intent.get("action") == "llm_reply":
            # The brain streamed and spoke this answer sentence by sentence
            # while it generated; re-speaking it here would play it twice. The
            # console still gets the text (timings["tts"] was set by the
            # streaming path in _brain_decide).
            print(f"Response: {response}")
            return response
        return self._respond(response)

    def _run_one(self, observation: dict, confirmed: bool = False) -> str:
        text = observation.get("text", "")
        intent = observation.get("intent")
        score = observation.get("score", 0.0)
        # What dispatch and history use. Only the "Open what?" answer differs:
        # it is the target of a verb the user already said, so it is dispatched
        # (and recorded) as the completed command rather than as a bare name.
        dispatch_text = text

        # Misses used to print nothing at all: the console looked dead while
        # the reply was spoken, and a *cached* reply emits no synthesis
        # warnings either — indistinguishable from a broken loop.
        if not text:
            print("Heard: (nothing intelligible)")
            return MISS_RESPONSE

        if intent is None and self.pending_clarification == "open_app":
            # Answer to "Open what?" — the *name* of the app was the question,
            # so a bare "notepad" is the target, not a new command. It has to
            # be dispatched as the command it completes ("open notepad"):
            # phrase extraction is anchored on the launch verb, so the bare
            # name would only ask the question again.
            self.pending_clarification = None
            intent = {"action": "open_app", "safe": True}
            dispatch_text = f"open {text}"
            score = 0.75
        elif intent is not None:
            # A recognised command answers nothing: the old question is stale.
            self.pending_clarification = None

        if intent is None:
            # Name the runner-up intent: "0.51 below the band" is not
            # actionable, "0.51 for time_check" tells the user the pipeline
            # heard the right thing and the threshold is what dropped it.
            best = getattr(self.router, "last_best_label", None)
            runner_up = f" for {best}" if best else ""
            print(f"Heard: {text}")
            print(
                f"Intent: none (best score {score:.2f}{runner_up}, "
                "below the confidence band)"
            )
            return MISS_RESPONSE

        action = intent["action"]
        print(f"Heard: {text}")
        print(f"Intent: {action} ({score:.2f})")

        if action == "llm_reply":
            # The brain already streamed and spoke this answer (observe did the
            # talking); just carry the text back so console and stats see it.
            return intent.get("reply", "")

        # Confirmation comes first: a risky action should be announced before
        # anything else is decided about it.
        if requires_confirmation(intent) and not confirmed:
            self.pending_confirmation = intent
            self.pending_text = text
            return confirmation_prompt(intent)

        if self.capabilities is not None and not self.capabilities.is_enabled(action):
            self.pending_confirmation = None
            return f"The {action.replace('_', ' ')} action is not enabled yet"

        started = time.perf_counter()
        response = self.execute(action, intent, dispatch_text)
        self.timings["action"] = time.perf_counter() - started

        self.pending_confirmation = None
        self.last_intent = intent
        self.last_text = dispatch_text
        if self.context is not None:
            self.context.log_command(action, dispatch_text)
        return response

    def execute(self, action: str, intent: dict, text: str) -> str:
        """Run one dispatched action and return the phrase to speak."""
        if action == "open_app":
            return self._open_app(intent, text)

        if action == "browser_search":
            query = intent.get("query") or extract_search_query(text)
            if not query:
                return "What should I search for?"
            return search_web(query, dry_run=self.dry_run)

        if action == "screen_ocr":
            return self._read_screen()

        if action == "context_open":
            return self._open_project(text)

        # intents.json declares this task as "system_control"; accept the task
        # names too so a rename cannot silently disable them. The intent's
        # target then picks the device: volume or brightness.
        if action in {"system_control", "system_volume", "system_brightness"}:
            return self._control_system(text, intent)

        if action == "greet":
            # "bye" is a greeting-shaped command; answering it with "How can I
            # help?" sounded like the assistant had not understood.
            if set(re.findall(r"[a-z']+", text.lower())) & {"bye", "goodbye"}:
                return "Goodbye. Talk to you soon."
            return f"{time_of_day_greeting()}. How can I help?"

        if action == "time_check":
            return f"The current time is {datetime.now().strftime('%I:%M %p')}."  # noqa: DTZ005

        if action == "identity":
            return IDENTITY_RESPONSE

        if action == "help":
            return self._describe_capabilities()

        if action == "repeat_last":
            return self._repeat_last()

        if action == "set_project":
            return self._set_project(text)

        if action == "set_browser":
            return self._set_browser(text)

        if action == "set_preference":
            return self._set_preference(text)

        if action == "close_window":
            from nova_agent.tools.desktop import close_active_window

            return close_active_window(dry_run=self.dry_run)

        if action == "tidy_downloads":
            from nova_agent.tools.desktop import move_latest_download_to_recycle_bin

            return move_latest_download_to_recycle_bin(dry_run=self.dry_run)

        if action == "lock_workstation":
            from nova_agent.tools.desktop import lock_workstation

            return lock_workstation(dry_run=self.dry_run)

        if action == "set_reminder":
            return self._set_reminder(text)

        if action == "manage_reminders":
            return self._manage_reminders(text)

        return f"The {action.replace('_', ' ')} action is not enabled yet"

    def _set_reminder(self, text: str) -> str:
        """Schedule a spoken reminder.

        Stored in **both** dry-run and live mode, deliberately: like
        ``set_project``/``set_browser`` this is a SQLite memory write plus a
        later spoken reply, and neither is a world action dry-run exists to
        suppress — so the reminder works identically in the default soak.
        """
        if self.context is None:
            return "I can't set reminders without a memory store."
        parsed = parse_reminder(text)
        if parsed is None:
            return "Tell me when — like: remind me to stretch in 20 minutes."
        what, delay = parsed
        self.context.add_reminder(what, int(time.time()) + delay)
        return f"Okay — I'll remind you to {what} in {humanize_delay(delay)}."

    def _manage_reminders(self, text: str) -> str:
        """List pending reminders, or drop them all on a cancel word."""
        if self.context is None:
            return "I can't keep reminders without a memory store."
        lowered = text.lower()
        if any(word in lowered for word in ("cancel", "clear", "delete")):
            removed = self.context.clear_reminders()
            if removed == 0:
                return "You have no reminders to clear."
            plural = "s" if removed != 1 else ""
            return f"Cleared {removed} reminder{plural}."
        pending = self.context.pending_reminders(int(time.time()))
        if not pending:
            return "You have no reminders pending."
        names = ", ".join(what for what, _ in pending)
        plural = "s" if len(pending) != 1 else ""
        return f"You have {len(pending)} reminder{plural} pending: {names}."

    def _respond(self, response: str) -> str:
        started = time.perf_counter()
        # Every spoken line is also shown: TTS-only replies made a missed
        # command indistinguishable from a dead loop. The console gets the
        # full response; TTS gets the speakable form (no path recitation).
        print(f"Response: {response}")
        self.tts.speak(speakable(response))
        elapsed = time.perf_counter() - started
        # Playback lasts as long as the reply *is* — content, not performance.
        # An engine that self-reports time-to-speech wins; test fakes (and
        # engines without the marker) keep the whole call.
        synth = getattr(self.tts, "last_synthesis_seconds", None)
        self.timings["tts"] = elapsed if synth is None else synth
        return response

    def _read_screen(self) -> str:
        from nova_agent.tools.screen_reader import read_screen

        try:
            return read_screen(
                tesseract_path=self.settings.tesseract_path,
                max_characters=self.settings.ocr_max_characters,
            )
        except RuntimeError as exc:
            return f"Screen reading is unavailable. {exc}"
        except Exception:  # noqa: BLE001 -- pragma: no cover, Tesseract-dependent
            return "Screen reading needs Tesseract OCR to be installed."

    def _open_project(self, text: str) -> str:
        if self.context is None:
            return "I don't have any projects saved yet."

        name = extract_project_name(text)
        path = self.context.find_project(name) if name else None
        if path is None and not name:
            path = self.context.get_preference("default_project")
        if path is None:
            if name:
                return f"I don't know where {name} is. Say set project {name} to a folder path."
            return "I don't know where that is. Say set project ml to a folder path."

        label = name or "default"
        # Record as the referent so a later "open it" reopens this project.
        self.last_open = {"action": "context_open", "text": text}
        if self.dry_run:
            return f"Would open project {label} at {path}"

        startfile = getattr(os, "startfile", None)
        if startfile is None:  # pragma: no cover - keeps non-Windows machines working.
            subprocess.Popen(["xdg-open", path])
        else:
            startfile(path)
        return f"Opening project {label}"

    def _control_system(self, text: str, intent: dict | None = None) -> str:
        """Volume or brightness, picked by the intent's target (volume by default)."""
        from nova_agent.tools.system import (
            apply_brightness,
            apply_volume,
            parse_brightness_command,
            parse_volume_command,
        )

        if (intent or {}).get("target") == "brightness":
            command = parse_brightness_command(text)
            if command is None:
                return "Do you want the brightness up, down, or to a percent?"
            return apply_brightness(command, dry_run=self.dry_run)
        command = parse_volume_command(text)
        if command is None:
            return "Do you want the volume up, down, or muted?"
        return apply_volume(command, dry_run=self.dry_run)

    def _open_app(self, intent: dict, text: str) -> str:
        """Open the app that was named, or say honestly that we cannot."""
        phrase = extract_app_phrase(text)
        if phrase is not None and is_pronoun_target(phrase):
            # "open it" / "open that": resolve against what we opened last so
            # a follow-up never repeats the name. Only real history answers;
            # with none, fall back to asking — exactly like a bare verb.
            reopened = self._resolve_open_pronoun()
            if reopened is not None:
                return reopened
            self.pending_clarification = "open_app"
            return OPEN_WHAT_RESPONSE
        if phrase is None:
            if looks_like_open_command(text):
                return "I don't know how to open that yet."
            target = intent.get("target")
            if not target:
                # Nothing was named at all ("open.", "launch"). Defaulting to
                # chrome here turned every mangled launch verb into a browser
                # launch (field log: `Heard: open.` -> "Would open chrome"),
                # which is worse than admitting the sentence was incomplete.
                # Ask — and remember asking, so the answer ("notepad") is read
                # as the missing target instead of as a fresh command.
                self.pending_clarification = "open_app"
                return OPEN_WHAT_RESPONSE
            # No app named at all (for example a bare "chrome"): trust the intent.
            phrase = str(target)

        return self._launch_phrase(phrase)

    def _launch_phrase(self, phrase: str) -> str:
        """Open one concrete, already-named app phrase (the resolution tail).

        Also records the opened app as the referent for a later "open it".
        """
        alias = self.context.get_alias(phrase) if self.context is not None else None
        key = alias or APP_WORDS.get(phrase, phrase)
        if key not in APP_PATHS:
            # Out-of-the-box app: resolve through PATH (System32 tools) and
            # the Start Menu, or say honestly that we do not know it.
            resolved = resolve_app(key)
            if resolved is None:
                return f"I don't know how to open {key} yet."
            self.last_open = {"action": "open_app", "text": phrase}
            return open_path(key, resolved, dry_run=self.dry_run)
        self.last_open = {"action": "open_app", "text": phrase}
        return open_app(key, dry_run=self.dry_run)

    def _resolve_open_pronoun(self) -> str | None:
        """Reopen whatever "it"/"that" points at: the last thing we opened.

        ``last_open`` — not ``last_text`` — is the referent, so the pronoun is
        reusable and survives an unrelated command in between. ``last_text``
        would become the pronoun itself after one use and lose the target.
        Returns ``None`` when history names nothing to reopen.
        """
        referent = self.last_open
        if not referent:
            return None
        action = referent.get("action")
        if action == "context_open":
            return self._open_project(referent["text"])
        if action == "open_app":
            return self._launch_phrase(referent["text"])
        return None

    def _set_project(self, text: str) -> str:
        if self.context is None:
            return "I can't remember projects without a context store."
        name, path = parse_set_project(text)
        if not name or not path:
            return "Say it like: set project ml to a folder path."
        self.context.set_project(name, path)
        if self.context.get_preference("default_project") is None:
            self.context.set_preference("default_project", path)
        return f"Remembered {name}."

    def _set_browser(self, text: str) -> str:
        app = parse_set_browser(text)
        if not app:
            return "Say it like: set browser to chrome."
        if self.context is None:
            return "I can't remember preferences without a context store."
        if app not in APP_PATHS and resolve_app(app) is None:
            # Remembering an alias only makes sense if opening can resolve it
            # later (APP_PATHS, PATH, or a Start Menu shortcut).
            return f"I don't know how to open {app} yet."
        self.context.set_alias("browser", app)
        return f"Browser set to {app}."

    def _set_preference(self, text: str) -> str:
        """Apply a spoken settings change now, and persist it for next start."""
        if self.context is None:
            return "I can't change preferences without a context store."
        parsed = parse_set_preference(text)
        if parsed is None:
            return "I couldn't tell which setting to change."
        key, value = parsed

        if key == "wake_threshold":
            threshold = float(value)  # the parser only yields digits
            if not 0.0 < threshold <= 1.0:
                return "The wake threshold must be above 0 and at most 1."
            self.context.set_preference(key, value)
            if self.wake_engine is None:
                return (
                    f"Wake threshold saved as {threshold:g}; it applies on the next start."
                )
            self.wake_engine.threshold = threshold
            return f"Wake threshold set to {threshold:g}."

        if key == "tts_voice":
            if not _looks_like_voice(value):
                return "I don't know that voice. Say a name like: af heart."
            self.tts.voice = value
            self.context.set_preference(key, value)
            self._recache_replies()
            return f"Voice set to {value}."

        enabled = value == "1"
        self.dry_run = enabled
        self.context.set_preference("dry_run", "1" if enabled else "0")
        if enabled:
            return "Dry run is on; I'll only report what I would do."
        return "Dry run is off; I will act for real."

    def _recache_replies(self) -> None:
        """Re-record the fixed replies in the new voice, off the command path.

        The TTS cache is keyed by phrase, not by voice, so without this every
        cached clip — including the preloaded phrasebook — would keep playing in
        the previous voice after "set voice to ...". Rebuilding inline would
        stall the very reply that reports the change, so it runs on a daemon
        thread; the synthesis lock serialises it with any other reply.
        """
        clear = getattr(self.tts, "clear_cache", None)
        preload = getattr(self.tts, "preload", None)
        if clear is None or preload is None:
            return  # a test double, or an engine without the cache API
        clear()
        phrases = [speakable(reply) for reply in self.canned_replies()]
        threading.Thread(
            target=lambda: preload(phrases), name="tts-recache", daemon=True
        ).start()

    def _repeat_last(self) -> str:
        """Re-run the previous command, re-asking for confirmation if needed."""
        intent = self.last_intent
        text = self.last_text
        if intent is None and self.context is not None:
            # Fall back to the stored history so this survives a restart.
            row = self.context.get_last_command()
            if row:
                intent = self._intent_for_action(row[0])
                text = row[1]

        if intent is None or not text:
            return "I haven't run anything yet."

        if requires_confirmation(intent):
            # Never silently repeat something destructive.
            self.pending_confirmation = intent
            self.pending_text = text
            return confirmation_prompt(intent)
        return self.execute(intent["action"], intent, text)

    def _describe_capabilities(self) -> str:
        """Answer "what can you do?" from the capability registry.

        Deliberately reads the registry and not intents.json: an action that is
        switched off (destructive ones without NOVA_ALLOW_DESTRUCTIVE=1) must
        not be advertised. Capped and prose-sized because every word is spoken.
        """
        if self.capabilities is None:
            return "I can open apps and answer questions about my commands."
        labels = [
            ACTION_LABELS[action]
            for action in self.capabilities.list_enabled()
            if action in ACTION_LABELS
        ]
        if not labels:
            return "I have no actions enabled right now."
        return f"I can {', '.join(labels[:MAX_HELP_ITEMS])}."

    def canned_replies(self) -> tuple[str, ...]:
        """The fixed replies this processor can answer with.

        The TTS warm-up preloads these into the phrase cache so no user command
        ever pays their synthesis — the longest of them, the screen-reading
        failure, measured 2669ms of synthesis against a 2500ms tts budget and
        the cache policy (short, digit-free phrases) would never store it. The
        greeting and the help list are computed, but they stay the same for the
        life of the process, which is all a cache needs.
        """
        return (
            MISS_RESPONSE,
            OPEN_WHAT_RESPONSE,
            IDENTITY_RESPONSE,
            CANCELLED_RESPONSE,
            self._describe_capabilities(),
            "Good morning. How can I help?",
            "Good afternoon. How can I help?",
            "Good evening. How can I help?",
            "Goodbye. Talk to you soon.",
            f"Screen reading is unavailable. {TESSERACT_MISSING_MESSAGE}",
            "Screen reading needs Tesseract OCR to be installed.",
            "What should I search for?",
            "Do you want the volume up, down, or muted?",
            "Do you want the brightness up, down, or to a percent?",
            "Turned the volume up",
            "Turned the volume down",
            "Turned the brightness up",
            "Turned the brightness down",
            "Muted the volume.",
            "Unmuted the volume.",
            "Toggled mute",  # the media-key fallback, which can only toggle
            "I can't change the brightness on this display.",
            "I don't know how to open that yet.",
            "I don't know where that is. Say set project ml to a folder path.",
            "I haven't run anything yet.",
        )

    def _intent_for_action(self, action: str) -> dict | None:
        for config in self.router.intents.values():
            if config.get("action") == action:
                return config
        return None


# Phase 3 latency budget, seconds. The verdict uses the worst command in the
# run: every stage must hold on every command, not just on average.
# Field-calibrated over three rounds (2-command probe, 13 commands, 11
# commands). Commands handled while the background warm-up thread holds the
# lazy loads are excluded from stats (see NovaAgent.warm_done) — they block
# by design, and one boot-window command would otherwise pin the verdict to a
# number no steady-state command sees. tts counts synthesis-to-audio only;
# total is wall clock including *speaking* the reply (paths are never spoken,
# see speakable()), so it is content-bound and deliberately generous;
# stages carry the performance signal.
# `llm` (Wave 2) is model-dependent and only present when the brain engages —
# format_stats skips stages with no samples, so the default brain-off soak is
# unaffected. 4.0s covers a local phi4-mini/qwen tool round-trip.
LATENCY_BUDGETS = {
    "stt": 3.0,
    "intent": 0.2,
    "llm": 4.0,
    "tts": 2.5,
    "action": 0.5,
    "total": 15.0,
}


def format_stats(history: list[dict[str, float]]) -> str:
    """Render per-command stage latencies against the latency budget.

    ``history`` is a list of ``{stage: seconds}`` snapshots, one per handled
    command (stages present depend on how the command flowed).
    """
    if not history:
        return "No commands were handled; no latency stats to report."

    rows = [
        f"Latency over {len(history)} command(s)",
        f"{'stage':<8}{'min':>11}{'median':>11}{'mean':>11}{'max':>11}{'budget':>11}  verdict",
    ]
    over: list[str] = []
    for name, budget in LATENCY_BUDGETS.items():
        values = sorted(row[name] for row in history if name in row)
        if not values:
            continue
        count = len(values)
        median = (
            values[count // 2] if count % 2 else (values[count // 2 - 1] + values[count // 2]) / 2
        )
        mean = sum(values) / count
        worst = values[-1]
        if worst <= budget:
            verdict = "ok"
        else:
            verdict = f"over by {(worst - budget) * 1000:.0f}ms"
            over.append(name)
        rows.append(
            f"{name:<8}{values[0] * 1000:9.0f}ms{median * 1000:9.0f}ms"
            f"{mean * 1000:9.0f}ms{worst * 1000:9.0f}ms{budget * 1000:9.0f}ms  {verdict}"
        )
    rows.append(
        f"Budget EXCEEDED on: {', '.join(over)}" if over else "Budget met on every stage."
    )
    return "\n".join(rows)


def stored_preference_float(context, key: str) -> float | None:
    """Read a stored preference as a float, tolerating missing/garbage values."""
    if context is None:
        return None
    raw = context.get_preference(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def resolve_dry_run(explicit: bool | None, context, settings: Settings) -> bool:
    """Precedence for dry-run: CLI flag > spoken preference > env default."""
    if explicit is not None:
        return explicit
    if context is not None:
        stored = context.get_preference("dry_run")
        if stored is not None:
            return stored == "1"
    return bool(settings.dry_run)


def _looks_like_voice(name: str) -> bool:
    """Kokoro voice ids look like ``af_heart`` / ``am_michael``."""
    parts = name.split("_", 1)
    return len(parts) == 2 and len(parts[0]) == 2 and parts[0].isalpha() and bool(parts[1])


# What run_observation returns when nothing was understood. Doubles as the
# retry signal: a miss inside the wake window keeps the command turn alive.
MISS_RESPONSE = "I didn't catch that"

# Asked when a launch verb named no app at all ("open."). Retries like a miss —
# the turn stays open for the answer, which pending_clarification turns into the
# missing target.
OPEN_WHAT_RESPONSE = "Open what?"
RETRY_RESPONSES = (MISS_RESPONSE, OPEN_WHAT_RESPONSE)

# Spoken when a confirmation is rejected, times out, or its turn is abandoned.
CANCELLED_RESPONSE = "Cancelled"


class NovaAgent:
    """Coordinate wake-word detection, command capture, and processing."""

    def __init__(
        self,
        wake_engine=None,
        vad_reader=None,
        processor=None,
        hud=None,
        debug: bool = False,
        wake_word: str | None = None,
        wake_threshold: float | None = None,
        context=None,
        dry_run: bool | None = None,
        queue_size: int = 64,
        capabilities=None,
        monitor=None,
        hud_enabled: bool | None = None,
        stt_device: str | None = None,
        stats: bool = False,
        transcript_wake: bool | None = None,
        voice_gate=None,
    ):
        settings = Settings()
        self.settings = settings
        self.stt_device = stt_device
        # The phrase to answer to: --wake-word flag, else the settings default.
        self.wake_word = wake_word or settings.wake_word
        self.transcript_wake = (
            settings.transcript_wake if transcript_wake is None else transcript_wake
        )
        self.context = context or ContextEngine()
        if wake_threshold is None:
            # A threshold spoken once ("set wake threshold to 0.6") persists
            # across restarts; the --wake-threshold flag still overrides it.
            wake_threshold = stored_preference_float(self.context, "wake_threshold")
        self.wake_engine = wake_engine or WakeWordEngine(
            model_path=str(WAKE_WORD_MODEL_PATH),
            threshold=wake_threshold,
            wake_word=self.wake_word,
        )
        self.vad_reader = vad_reader or VADRecorder(
            sample_rate=settings.sample_rate,
            max_silence=settings.vad_silence_frames,
            min_speech=settings.vad_min_speech_frames,
            speech_threshold=settings.vad_speech_threshold,
            vad_model_path=settings.vad_model_path,
            max_seconds=settings.vad_max_seconds,
        )
        self.processor = processor
        # Voice gate: which voice may wake us (core/voice_gate.py). The default
        # instance is open unless a voiceprint exists; tests inject their own.
        self.voice_gate = voice_gate or VoiceGate(settings)
        # Rolling ~1.5s window of mic frames — enough voice to compare a
        # speaker when the audio model (not a transcript) fires the wake.
        self._recent_audio: deque = deque(
            maxlen=max(1, int(1.5 * settings.sample_rate / settings.sample_block))
        )
        self.capabilities = capabilities or build_registry(settings)
        self.monitor = monitor or RuntimeMonitor()
        self.hud = hud or NovaHUD(
            enabled=settings.hud_enabled if hud_enabled is None else hud_enabled
        )
        self.listening = False
        self.debug = debug
        self.stats = stats
        self.stats_history: list[dict[str, float]] = []
        # Cleared by run() while the background model warm-up is in flight,
        # set again when it finishes (set by default so direct test harnesses
        # that never run() still record). Commands handled mid-warm block on
        # the lazy loads by design — boot artifacts that would otherwise pin
        # the worst-case budget verdict to numbers no steady-state command
        # ever sees (field: intent 1986ms, total 24843ms on command #1).
        self.warm_done = threading.Event()
        self.warm_done.set()
        self.dry_run = dry_run
        self.confirmation_timeout = settings.confirmation_timeout
        # Microphone frames are handed to a worker thread so the PortAudio
        # callback never blocks on Whisper or Kokoro playback.
        self.audio_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        # Barge-in: an adaptive energy gate, fed only while Kokoro plays. It
        # learns the steady playback bleed from real frames, so a person
        # talking over the reply is a step change above that floor while the
        # reply's own echo is not (core/barge.py explains why not the VAD).
        self.barge_gate = BargeGate(sample_rate=settings.sample_rate)
        # Frames the PortAudio callback dropped because the worker fell behind:
        # a gap in what the VAD assembles shows up as a low count here, printed
        # alongside idle segments when debugging.
        self._dropped_frames = 0
        # Set by _begin_listening: when the current command turn began.
        # Misses may keep the turn alive only up to command_max_turn past it.
        self._turn_started = 0.0
        self.is_speaking = False
        # When the last reply stopped (distant past at boot: the reverb-tail
        # guard must not deafen the mic before anything has ever played).
        self._speaking_ended_at = float("-inf")
        self._worker: threading.Thread | None = None
        self._stopping = False

    def process_chunk(self, chunk):
        """Return a status token or the recorded audio path for the next stage."""
        # Echo guard: never start (or continue) a capture from audio that is us
        # talking — otherwise Kokoro playback wakes the assistant on its own voice.
        if self.is_speaking:
            return None

        energy = float(
            np.sqrt(np.mean(np.square(np.asarray(chunk, dtype=np.float32)))) if chunk.size else 0.0
        )
        if self.debug:
            print(f"[debug] energy={energy:.4f} listening={self.listening}")

        if not self.listening:
            if self.wake_engine.process_chunk(chunk):
                # Voice gate on the rolling window: the sound model crossed on
                # *something* — confirm it was the enrolled voice before acting.
                if self.voice_gate.verify_samples(self._recent_audio_window()) is False:
                    self.wake_engine.reset()
                    print(
                        f"[wake] ignored (voice mismatch {self.voice_gate.last_score:.2f} < "
                        f"{self.voice_gate.threshold:g}, audio model)"
                    )
                    return None
                self._begin_listening("audio model")
                return "wake"
            if self._transcript_wake_active():
                # Idle probes are ambient: cap them here so background chatter
                # cannot buffer toward vad_max_seconds and cost seconds of STT
                # (field round 8: an 11.5s transcription from one noise segment).
                self.vad_reader.max_seconds = self.settings.idle_segment_max
                segment = self.vad_reader.process_chunk(chunk)
                if segment is not None:
                    return self._wake_from_transcript(segment)
            return None

        self.hud.set_state("thinking", "Thinking")
        result = self.vad_reader.process_chunk(chunk)
        if result is None:
            return None

        # The turn closes only once the outcome is known: a miss inside the
        # wake window keeps listening (background audio often wins the first
        # capture while the real command is still coming).
        response = self._run_captured(result)
        self._finish_turn(response)
        if self.debug:
            print(f"[debug] captured command: {result}")
        return str(result)

    def _begin_listening(self, how: str) -> None:
        """Enter command-listening mode from either wake detector."""
        self.listening = True
        self._turn_started = time.monotonic()
        # Clear detection history so the tail of the wake phrase cannot
        # immediately re-trigger, and reset the ribbon for the new command.
        self.wake_engine.reset()
        self.vad_reader.reset()
        # Command captures keep the full segment budget; the shorter idle cap
        # (idle_segment_max) is re-applied by the idle branch on the next frame.
        self.vad_reader.max_seconds = self.settings.vad_max_seconds
        self.hud.show_transcript("")
        self.hud.set_state("listening", "Listening")
        # Always say it: without this line a successful wake in normal mode is
        # indistinguishable from a dead loop (the HUD alone is easy to miss).
        # The source names which detector fired: field round 8 traced false
        # wakes to the audio model, which this line alone never showed.
        source = "transcript" if how.startswith("transcript") else how
        print(f"Wake detected ({source}) - listening for your command.")
        if self.debug:
            print(f"[debug] wake detected ({how})")

    def _finish_turn(self, response: str) -> None:
        """Close (or keep) the command turn after one capture.

        A miss — empty transcript or nothing above the confidence band, e.g.
        background video winning the capture while the real command is still
        being spoken — must not end the turn. The window is re-armed from the
        miss itself, not from the wake: the pipeline's own latency used to
        consume it (command #1 took 13.33s end to end against an 8s window,
        so the retry never fired). ``command_max_turn`` still bounds the whole
        turn so background audio cannot hold it hostage.

        The clarification question retries too: "Open what?" is only useful if
        the answer can still be captured without saying the wake word again.
        """
        if response not in RETRY_RESPONSES:
            self.listening = False
            self.hud.set_state("idle", "idle")
            return
        now = time.monotonic()
        deadline = min(
            now + self.settings.command_window,
            self._turn_started + self.settings.command_max_turn,
        )
        if now < deadline:
            print("Still listening for your command.")
            self.hud.set_state("listening", "Listening")
            return
        self.listening = False
        self.hud.set_state("idle", "idle")

    def _transcript_wake_active(self) -> bool:
        """Whether idle speech should be segmented and checked for the phrase."""
        return bool(
            self.transcript_wake
            and self.processor is not None
            and getattr(self.processor, "stt", None) is not None
        )

    def _wake_from_transcript(self, segment_path) -> str | None:
        """Transcribe one idle utterance; wake when it holds the wake phrase.

        This is the "hey dude" path: the ONNX sound model only ever fires on
        the phrase it was trained for, so any other wording is recognized from
        what was actually *said*. Whatever follows the phrase is treated as
        the command, so "hey dude, open chrome" still runs in one breath.
        """
        try:
            rms, seconds = _wav_stats(segment_path)
            if rms < self.settings.wake_segment_min_rms:
                if self.debug:
                    print(
                        f"[debug] idle segment too quiet for STT "
                        f"({seconds:.2f}s rms={rms:.4f}, "
                        f"dropped frames: {self._dropped_frames})"
                    )
                Path(segment_path).unlink(missing_ok=True)
                return None
            if self.debug:
                # The live fingerprint: does the segment hold speech, and were
                # mic frames lost before it reached the VAD?
                print(
                    f"[debug] idle segment: {seconds:.2f}s rms={rms:.4f} "
                    f"(dropped frames: {self._dropped_frames})"
                )
            text = self.processor.transcribe(
                segment_path, prompt=wake_prompt_for(self.wake_word)
            )
        except Exception as exc:  # noqa: BLE001 - one bad segment must not kill the loop
            # Always loud: a silently failing STT would look like a dead loop.
            print(f"Wake transcription failed: {exc}")
            Path(segment_path).unlink(missing_ok=True)
            return None

        matched, remainder = wake_phrase_hits(text, self.wake_word)
        if not matched:
            if self.debug:
                # Keep the evidence under a distinct name: replaying what the
                # live loop actually heard is the only way to debug a miss.
                original = Path(segment_path)
                kept = original
                renamed = original.with_name(f"idle_{original.name}")
                if not renamed.exists():
                    original.rename(renamed)
                    kept = renamed
                print(f"[debug] idle transcript (no wake): {text!r} [kept: {kept}]")
                return None
            # Idle segments are probes, not command records.
            Path(segment_path).unlink(missing_ok=True)
            return None

        # Voice gate: text cannot separate a real wake from a Whisper prompt
        # echo on noise (both read "hey dude") — the speaker can. A mismatch
        # stays silent: nobody addressed us after all.
        if self.voice_gate.verify_file(segment_path) is False:
            print(
                f"[wake] ignored (voice mismatch {self.voice_gate.last_score:.2f} < "
                f"{self.voice_gate.threshold:g}): {text!r}"
            )
            Path(segment_path).unlink(missing_ok=True)
            return None
        score = self.voice_gate.last_score
        suffix = f" (voice {score:.2f})" if score is not None else ""
        print(f"[wake] matched: {text!r}{suffix}")
        Path(segment_path).unlink(missing_ok=True)
        self._begin_listening(f"transcript: {text!r}")
        if not remainder:
            return "wake"

        # One-shot: the words after the wake phrase are the command.
        self._finish_turn(self._run_captured(text=remainder))
        return "wake"

    def _run_captured(self, audio_path=None, *, text: str | None = None) -> str:
        """Run one captured command (from a file, or pre-transcribed text).

        Returns the response so the caller can decide whether a miss should
        keep the command turn alive (see ``_finish_turn``).
        """
        if self.processor is None:
            return ""
        # Decide the stats verdict up front: a command that *started* while the
        # background warm-up held the model loads is a boot artifact however it
        # ends. Field round 4 proved the end check alone is racy — command #1
        # blocked on the encoder lock, then its TTS blocked on the warm-up's
        # synthesis lock, so it finished after warm_done.set() and got counted
        # (intent 19088ms, total 33010ms pinned the verdict).
        warm_at_start = self.warm_done.is_set()
        # Per-command stage timings: the wake-segment transcription also runs
        # through processor.transcribe, and a text-path command (transcript-wake
        # remainder) never transcribes at all — without a reset, its snapshot
        # inherited the wake segment's stt (field round 4: a 10888ms idle
        # segment leaked into a command row).
        self.processor.timings.clear()
        self.hud.set_state("speaking", "Processing")
        started = time.perf_counter()
        # Mute the pipeline while it works: Kokoro playback would otherwise
        # be captured by the open input stream and treated as a new wake word.
        self.is_speaking = True
        try:
            self._discard_pending_audio()
            if text is None:
                response = self.handle_command(audio_path)
            else:
                response = self.handle_command(text=text)
        finally:
            self._finish_speaking()
            self._discard_pending_audio()
        elapsed = time.perf_counter() - started
        if self.debug:
            print(f"[debug] command handled in {elapsed:.2f}s")
        self._record_metrics(elapsed)
        if self.stats:
            if not warm_at_start or not self.warm_done.is_set():
                print("[stats] command handled during model warm-up; not counted")
            else:
                snapshot = dict(self.processor.timings)
                snapshot["total"] = elapsed
                self.stats_history.append(snapshot)
        return response

    def handle_command(
        self, audio_path=None, confirmed: bool = False, text: str | None = None
    ) -> str:
        """Run one captured command, including any spoken confirmation.

        ``text`` skips transcription — transcript wake already transcribed the
        utterance it woke on, so the tail after the phrase arrives as text.

        The confirmation window is multi-turn: a clear "no" or silence cancels,
        but a reply that is itself a confident new command *replaces* the
        pending one ("close the window" -> "no, open chrome"), bounded by
        MAX_COMMAND_SWAPS so destructive prompts cannot ping-pong forever.
        """
        if text is None:
            text = self.processor.transcribe(
                audio_path, prompt=COMMAND_PROMPT, command=True
            )
        self.hud.show_transcript(text)
        observation = self.processor.observe(text)
        response = self.processor.run_observation(observation, confirmed=confirmed)
        self._debug_stages()
        if self.processor.pending_confirmation is None:
            return response  # nothing to confirm: the ordinary happy path

        swaps = 0
        while self.processor.pending_confirmation is not None:
            reply = self._await_confirmation()
            if reply is None:
                break  # silence or no capture: fall through to cancel
            decision = confirmation_decision(reply)
            if decision == CONFIRM:
                return self.processor.run_observation(
                    {
                        "text": self.processor.pending_text,
                        "intent": self.processor.pending_confirmation,
                    },
                    confirmed=True,
                )
            if decision == REJECT:
                break
            # Not a yes/no: try to read it as a fresh command instead of just
            # giving up on it — the user may be replacing, not answering.
            swapped = self.processor.observe(reply)
            if swapped["intent"] is None or swaps >= MAX_COMMAND_SWAPS:
                break
            swaps += 1
            self.hud.show_transcript(reply)
            response = self.processor.run_observation(swapped)
            self._debug_stages()
            if self.processor.pending_confirmation is None:
                return response  # safe swap completed; nothing pending anymore
            # The swap asked for its own confirmation: loop and ask again.

        self.processor.pending_confirmation = None
        # A rejected/pending turn also drops a compound's queued follow-ups.
        self.processor.pending_compound = []
        return self.processor.respond(CANCELLED_RESPONSE)

    def _debug_stages(self) -> None:
        if self.debug and self.processor.timings:
            stages = " ".join(
                f"{name}={value * 1000:.0f}ms" for name, value in self.processor.timings.items()
            )
            print(f"[debug] stages: {stages}")

    def _await_confirmation(self) -> str | None:
        """Return the spoken reply, or None on silence or timeout.

        The caller decides what the reply *means*: confirm, reject, or a new
        command spoken in place of the yes/no that was asked for.
        """
        if sd is None or self.vad_reader is None:
            return None

        self.hud.set_state("confirming", "Waiting for confirmation")
        self.vad_reader.reset()
        started = time.perf_counter()

        while time.perf_counter() - started < self.confirmation_timeout:
            try:
                chunk = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            result = self.vad_reader.process_chunk(chunk)
            if result is None:
                continue

            self.is_speaking = True
            try:
                reply = self.processor.transcribe(result, prompt=COMMAND_PROMPT)
            finally:
                self._finish_speaking()
                self._discard_pending_audio()
            if self.debug:
                print(f"[debug] confirmation reply: {reply!r}")
            return reply or None

        return None

    def _record_metrics(self, elapsed: float) -> None:
        """Feed the runtime monitor so resource health stays observable."""
        self.monitor.record_metric("last_command_seconds", elapsed)
        try:
            import psutil

            self.monitor.record_metric("cpu", psutil.cpu_percent(interval=None))
            self.monitor.record_metric("memory", psutil.virtual_memory().percent)
        except ImportError:  # pragma: no cover - psutil is optional.
            pass

    def _barge_begin(self) -> None:
        """Playback is opening: forget the past, calibrate on real bleed.

        Frames queued before Kokoro opened his mouth are the tail of the user's
        command or room noise. Feeding them to the gate would seed its baseline
        wrong — a silent gap right before we speak makes the first echo syllable
        look like a shout and trips us instantly. Drop them, reset the gate, and
        let it learn the echo floor from actual playback.
        """
        self.barge_gate.reset()
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def _barge_check(self) -> bool:
        """Feed the frames that arrived during the last chunk; True = stop.

        Runs on the worker thread between ~50ms playback chunks, so while
        speak() holds the floor this is the only consumer of audio_queue — the
        echo guard that drops frames elsewhere has nobody to drop them here.
        Returns whether the user talked over the reply long enough to count.
        """
        tripped = False
        while True:
            try:
                frame = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            if self.barge_gate.feed(frame):
                tripped = True
        if tripped:
            stop = getattr(self.processor.tts, "stop", None)
            if stop is not None:
                stop()
            print("\n[user] talked over the reply — playback stopped")
        return tripped

    def _finish_speaking(self) -> None:
        """Reopen the mic and start the reverb-tail cooldown.

        The echo guard only covers playback itself; the room keeps ringing for
        a few hundred ms after it ends (field round 8: wakes fed by the reverb
        of our own "Opening chrome"). Every speaking site ends here, so this is
        the one place that knows when we actually stopped — ``_on_audio`` then
        drops the tail before it ever reaches the queue.
        """
        self.is_speaking = False
        self._speaking_ended_at = time.monotonic()

    def _recent_audio_window(self) -> np.ndarray:
        """The last ~1.5s of mic audio — enough voice to compare a speaker."""
        if not self._recent_audio:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._recent_audio)

    def _on_audio(self, indata, _frames, _time, _status) -> None:
        """PortAudio callback: queue the chunk and return immediately.

        Transcription, synthesis, and window automation must never run here;
        blocking this thread makes PortAudio drop microphone frames.

        ``indata`` is PortAudio's *reusable* ring-buffer memory: a view of it
        is stale by the time the worker reads it — the next callback overwrites
        the speech, so segments the VAD holds until finalization would be
        written from later (silent) audio. Copy before queueing.
        """
        # Tail guard: for wake_cooldown seconds after we stop talking the room
        # is still ringing with us — drop frames at the source so neither the
        # wake model nor the VAD ever sees our own reverb.
        if time.monotonic() - self._speaking_ended_at < self.settings.wake_cooldown:
            return
        chunk = np.array(indata[:, 0], dtype=np.float32)
        # Rolling window for the voice gate (audio-model path): an append of a
        # reference — zero copy, nothing here may block.
        self._recent_audio.append(chunk)
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            # Never print here (the callback must not block): count only, so
            # debug output can show that live audio is being lost.
            self._dropped_frames += 1

    def process_pending(self, limit: int | None = None) -> int:
        """Drain queued microphone frames. Returns the number handled."""
        handled = 0
        while limit is None or handled < limit:
            try:
                chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            if self.is_speaking:
                continue
            self.process_chunk(chunk)
            handled += 1
        return handled

    def _discard_pending_audio(self) -> int:
        """Drop frames captured while the assistant was talking to itself."""
        dropped = 0
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
            dropped += 1
        return dropped

    def _fire_due_reminders(self) -> None:
        """Speak every reminder whose time has come.

        Runs on the worker thread *between* command turns — the same thread
        that speaks commands — so ``is_speaking`` is never toggled from two
        places and a reminder can never overlap a command. Going through
        ``tts.speak`` means the barge gate applies too: you can talk over a
        reminder and cut it off like any other reply. A reminder is marked
        fired *before* speaking so a synthesis failure can never make it loop.
        """
        if self.context is None or self.processor is None:
            return
        if self.is_speaking:  # belt-and-suspenders: never talk over ourselves
            return
        due = self.context.due_reminders(int(time.time()))
        if not due:
            return
        for reminder_id, what in due:
            self.context.mark_reminder_fired(reminder_id)
            message = f"Reminder: {what}"
            self.is_speaking = True
            self.hud.set_state("speaking", "Reminder")
            try:
                self._discard_pending_audio()
                print(message)
                self.processor.tts.speak(speakable(message))
            finally:
                self._finish_speaking()
                self._discard_pending_audio()
                self.hud.set_state("listening", "Listening")

    def _consume_loop(self) -> None:
        next_reminder_check = 0.0
        while not self._stopping:
            if time.monotonic() >= next_reminder_check:
                next_reminder_check = time.monotonic() + 1.0
                try:
                    self._fire_due_reminders()
                except Exception as exc:  # noqa: BLE001 - a dead reminder must not kill the loop
                    print(f"Reminder check failed ({type(exc).__name__}): {exc}")
            try:
                chunk = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.is_speaking:
                continue
            try:
                self.process_chunk(chunk)
            except Exception as exc:  # noqa: BLE001 - a dead consumer looks like a dead loop
                # One bad command used to kill this thread outright: the
                # assistant kept "listening" but never responded again. Fail
                # the command loudly and keep the loop alive.
                print(f"Command failed ({type(exc).__name__}): {exc}")
                traceback.print_exc()

    def run(self):
        if sd is None:
            raise RuntimeError("Microphone support requires sounddevice.")

        settings = self.settings
        ensure_directories()
        dry_run = resolve_dry_run(self.dry_run, self.context, settings)

        if self.processor is None:
            from nova_agent.core.stt import STTEngine
            from nova_agent.core.tts import TTSEngine

            stt = STTEngine(
                settings.stt_model,
                self.stt_device or settings.stt_device,
                settings.stt_compute_type,
            )
            if settings.stt_warmup:
                warm = stt.warm_up()
                print(f"STT warm-up: {warm:.2f}s on {stt.device}")
            router = build_router(settings, self.capabilities)
            self.processor = CommandProcessor(
                stt,
                router,
                TTSEngine(
                    voice=self.context.get_preference("tts_voice") or settings.tts_voice,
                    # settings is the documented home of the cache policy; the
                    # engine used to fall back to its own identical defaults.
                    max_cache_words=settings.tts_max_cache_words,
                    max_cache_characters=settings.tts_max_cache_characters,
                ),
                dry_run=dry_run,
                context=self.context,
                settings=settings,
                capabilities=self.capabilities,
                monitor=self.monitor,
                wake_engine=self.wake_engine,
            )
            if settings.intent_warmup or settings.tts_warmup:
                # Warm off the boot path: cold loads measured 21.3s (intent)
                # and 21.5s (first Kokoro synthesis) live, and an always-on
                # assistant must reach "listening" immediately. A command
                # arriving mid-warm blocks on the same lazy load it used to
                # pay anyway — the re-armed retry window covers even that.
                self.warm_done.clear()

                def _warm_models() -> None:
                    try:
                        if settings.intent_warmup:
                            print(f"Intent warm-up: {router.warm_up():.2f}s (background)")
                        warm_tts = getattr(self.processor.tts, "warm_up", None)
                        if settings.tts_warmup and warm_tts is not None:
                            # Warm the model *and* preload the phrasebook: the
                            # fixed replies are spoken from the cache, so their
                            # synthesis never lands inside a user command (the
                            # long ones are not cacheable by policy, which is
                            # exactly why they were re-synthesised every time).
                            phrases = [
                                speakable(reply)
                                for reply in self.processor.canned_replies()
                            ]
                            seconds = warm_tts(phrases=phrases)
                            print(
                                f"TTS warm-up: {seconds:.2f}s (background, "
                                f"{self.processor.tts.last_preload_count} replies preloaded)"
                            )
                    finally:
                        self.warm_done.set()

                threading.Thread(
                    target=_warm_models, name="model-warm", daemon=True
                ).start()

        # Barge-in: while Kokoro speaks, the worker thread blocked in playback
        # is the only reader of audio_queue — so speak() calls these back
        # between ~50ms chunks. begin() flushes the pre-playback backlog and
        # resets the gate; check() feeds the frames that arrived during the
        # last chunk and reports whether the user talked over the reply.
        self.processor.tts.barge_begin = self._barge_begin
        self.processor.tts.barge_check = self._barge_check

        mode = "dry-run" if dry_run else "live"
        detector = getattr(self.wake_engine, "backend", "custom")
        print(
            f"DUDE starting (execution: {mode}, wake word: {self.wake_word!r}, "
            f"detector: {detector}, vad: {getattr(self.vad_reader, 'backend', 'custom')})"
        )
        transcript_mode = "on" if self._transcript_wake_active() else "off"
        print(
            f"Transcript wake: {transcript_mode} "
            f"(idle speech is matched for {self.wake_word!r})"
        )
        print(f"Voice gate: {self.voice_gate.status()}")
        load_error = getattr(self.wake_engine, "load_error", None)
        if detector != "openwakeword" and load_error:
            print(f"Wake word note: {load_error}")
        enabled = self.capabilities.list_enabled()
        print(f"Actions enabled: {len(enabled)}")
        blocked = [
            action
            for action in DESTRUCTIVE_ACTIONS
            if not self.capabilities.is_enabled(action)
        ]
        if blocked:
            print(
                "Destructive actions disabled "
                f"({', '.join(blocked)}); set NOVA_ALLOW_DESTRUCTIVE=1 to enable."
            )

        self.hud.start()
        self._stopping = False
        self._worker = threading.Thread(target=self._consume_loop, daemon=True)
        self._worker.start()

        try:
            with sd.InputStream(
                samplerate=settings.sample_rate,
                blocksize=settings.sample_block,
                dtype="float32",
                channels=1,
                callback=self._on_audio,
            ):
                print("DUDE is listening. Say the wake word to begin.")
                try:
                    while True:
                        sd.sleep(100)
                except KeyboardInterrupt:
                    # Ctrl+C is how this loop normally ends. Swallow it here so
                    # the finally block below still runs (stats + HUD shutdown)
                    # instead of a traceback being the last thing on screen.
                    print("\nStopping.")
        finally:
            self._stopping = True
            self.hud.set_state("idle", "Stopped")
            if self._worker is not None:
                self._worker.join(timeout=1.0)
            self.hud.stop()
            if self.stats:
                print(format_stats(self.stats_history))


def run_once(dry_run: bool = True) -> int:
    from nova_agent.core.recorder import record_command
    from nova_agent.core.stt import STTEngine
    from nova_agent.core.tts import TTSEngine

    settings = Settings()
    ensure_directories()
    recording = COMMAND_DIR / "command.wav"
    mode = "dry-run" if dry_run else "live"
    print(f"Press Enter, then speak your command... (execution: {mode})")
    input()
    record_command(recording, settings.command_seconds, settings.sample_rate)
    registry = build_registry(settings)
    processor = CommandProcessor(
        STTEngine(settings.stt_model, settings.stt_device, settings.stt_compute_type),
        build_router(settings, registry),
        TTSEngine(voice=settings.tts_voice),
        dry_run=dry_run,
        context=ContextEngine(),
        settings=settings,
        capabilities=registry,
    )
    observation = processor.observe(processor.transcribe(recording))
    response = processor.run_observation(observation)
    if processor.pending_confirmation is not None:
        # One-shot mode has no second recording pass, so ask for a typed answer.
        print(response)
        answer = input("Type 'confirm' to proceed: ").strip().lower()
        if confirmation_decision(answer) == CONFIRM:
            processor.run_observation(observation, confirmed=True)
        else:
            processor.respond("Cancelled")
    return 0


def check_environment() -> int:
    from nova_agent.core.tts import TTSEngine

    ensure_directories()
    settings = Settings()
    registry = build_registry(settings)
    router = build_router(settings, registry)
    print("Dude environment check")
    print(f"Project root: {router.project_root}")
    print(f"Intents loaded: {len(router.intents)}")
    print(f"Sample rate: {settings.sample_rate} Hz")
    print(f"STT: {settings.stt_model} on {settings.stt_device} ({settings.stt_compute_type})")
    print(f"Execution mode: {'dry-run' if settings.dry_run else 'live'}")
    print(
        "Tier 2 fallback: "
        f"{'on' if settings.llm_fallback and not settings.llm_brain else 'off'}"
    )
    print(f"LLM brain: {'on' if settings.llm_brain else 'off'}")
    tesseract = settings.tesseract_path or "default (PATH)"
    print(f"Tesseract: {tesseract}")

    detector = WakeWordEngine(
        model_path=str(WAKE_WORD_MODEL_PATH),
        threshold=settings.wake_threshold,
        wake_word=settings.wake_word,
    )
    print(f"Wake detector: {detector.backend} (threshold {detector.threshold:g})")
    if detector.load_error:
        print(f"Wake detector note: {detector.load_error}")
    print(
        f"Wake phrase: {settings.wake_word!r} "
        f"(transcript wake: {'on' if settings.transcript_wake else 'off'})"
    )
    print(f"Voice gate: {VoiceGate(settings).status()}")

    recorder = VADRecorder(
        max_silence=settings.vad_silence_frames,
        min_speech=settings.vad_min_speech_frames,
        speech_threshold=settings.vad_speech_threshold,
        vad_model_path=settings.vad_model_path,
    )
    print(f"Voice detector: {recorder.backend}")
    if recorder.vad is not None and recorder.vad.load_error:
        print(f"Voice detector note: {recorder.vad.load_error}")

    from nova_agent.tools.system import brightness_backend, volume_backend

    # Which path volume commands take: levels and mute states need Core Audio,
    # the media keys can only nudge and toggle. Brightness needs a controllable
    # display behind WMI; desktops driving external monitors only report here.
    print(f"Volume backend: {volume_backend()}")
    print(f"Brightness backend: {brightness_backend()}")

    enabled = registry.list_enabled()
    print(f"Actions enabled: {len(enabled)}")
    print(f"Destructive actions: {', '.join(registry.list_destructive()) or 'disabled'}")
    cached = len(list(CACHE_DIR.glob('*.wav'))) if CACHE_DIR.exists() else 0
    print(f"Cached TTS phrases: {cached}")
    print(f"TTS cache policy: <= {settings.tts_max_cache_words} words, no digits")

    for package in (
        "numpy",
        "scipy",
        "sounddevice",
        "faster_whisper",
        "kokoro",
        "openwakeword",
        "onnxruntime",
        "psutil",
        "speechbrain",  # voice gate (missing = --enroll-voice refuses politely)
    ):
        status = "installed" if importlib.util.find_spec(package) else "missing"
        print(f"{package}: {status}")

    tts = TTSEngine(cache_dir=CACHE_DIR)
    stale = [
        path
        for path in CACHE_DIR.glob("*.wav")
        if path.stem != tts._cache_path(path.stem).stem
        or any(character.isdigit() for character in path.stem)
    ]
    if stale:
        print(f"Stale cache entries: {len(stale)} (run --prune-cache)")
    return 0


def calibrate_microphone(seconds: float = 3.0) -> float:
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - depends on runtime environment.
        raise RuntimeError("Microphone calibration requires sounddevice.") from exc

    sample_rate = Settings().sample_rate
    frames = int(sample_rate * seconds)
    recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    audio = np.asarray(recording[:, 0], dtype=np.float32)
    recommended = WakeWordEngine.estimate_threshold(audio)
    print(f"Ambient energy sample: {float(np.abs(audio).mean()):.4f}")
    print(f"Recommended energy-fallback threshold: {recommended:.4f}")
    return recommended


def calibrate_wizard(wake_word: str | None = None) -> int:
    """Two-step calibration: ambient noise floor, then real wake attempts.

    Step (a) calibrates the loudness fallback; step (b) records actual wake
    phrase attempts and reports transcript hits (the primary trigger) plus
    the sound model's score distribution.
    """
    settings = Settings()
    wake_word = wake_word or settings.wake_word
    print("Step 1/2: ambient energy (calibrates the energy fallback)")
    calibrate_microphone()
    print(f"\nStep 2/2: wake phrase attempts (does {wake_word!r} wake it?)")
    return wake_probe(wake_word=wake_word, attempts=5)


def join_replies(*parts: str) -> str:
    """Join the per-clause replies of a compound command into one response.

    Each clause answers with its own sentence ("Turned the brightness up",
    "Would open chrome (C:\\...)"); the stop between clauses is normalized so
    a reply that ended without one doesn't run into the next clause, while
    the final clause keeps whatever punctuation it already had.
    """
    live = [part.strip() for part in parts if part and part.strip()]
    if not live:
        return ""
    body = ". ".join(part.rstrip(". ") for part in live[:-1])
    if not body:
        return live[-1]
    return f"{body}. {live[-1]}"


def speakable(response: str) -> str:
    """What TTS should say: the console gets full detail, speech gets prose.

    Dry-run replies carry resolved paths ("Would open chrome (C:\\...)" or
    "... at C:/work/ml"); reading those aloud made one field command run
    24.8s of path recitation — content, but it blew the wall-clock budget
    and nobody wants to hear a Windows path spelled out.
    """
    spoken = re.sub(r"\s*\([^)]*[\\/][^)]*\)", "", response)  # (C:\...)
    spoken = re.sub(r"\s+\bat\s+[A-Za-z]:.*$", "", spoken)  # at C:/work/ml
    return spoken.strip() or response


def wake_prompt_for(wake_word: str) -> str | None:
    """Whisper decoding context for wake detection: the phrase as a prior.

    Short two-word utterances otherwise fall into Whisper's generic priors
    on some voices ("what are you doing?" for "hey dude"). Priming fixes the
    miss without forcing the phrase onto unrelated audio (verified against
    silence, tone, noise, and non-wake speech).
    """
    phrase = (wake_word or "").strip().capitalize()
    return f"{phrase}. {phrase}." if phrase else None


# Command-slot Whisper context: the same priming trick that made wake
# transcripts reliable, pointed at the intent vocabulary instead. Field runs
# showed unprompted Whisper-small mangling clear, loud command audio
# ("mute the volume" -> "we hope the volume", "open python projects" ->
# "open by 10 projects") while primed wake segments transcribed 3/3. Keep the
# phrases aligned with config/intents.json; do not add filler the intents
# would not want matched.
COMMAND_PROMPT = (
    "Open Chrome. Open my project. What time is it? "
    "Mute the volume. Read the screen. Search the web. Set browser to Chrome."
)


def _wav_stats(path) -> tuple[float, float]:
    """RMS (0..1) and duration of a wav; unreadable files return (1.0, 0.0).

    An unreadable file passes the quiet gate on purpose: STT should produce
    the loud error instead of a silent skip.
    """
    from scipy.io import wavfile

    try:
        sample_rate, data = wavfile.read(path)
    except (OSError, ValueError):
        return 1.0, 0.0
    audio = np.asarray(data)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / 32768.0
    audio = audio.astype(np.float32)
    if audio.size == 0:
        return 0.0, 0.0
    rms = float(np.sqrt(np.mean(np.square(audio))))
    seconds = audio.shape[0] / float(sample_rate or 1)
    return rms, seconds


def _wav_rms(path) -> float:
    """RMS of a wav as 0..1 float; unreadable files return 1.0 so STT errors."""
    return _wav_stats(path)[0]


def wake_status(wake_word: str | None = None) -> int:
    """Report which wake detector is live and what it currently scores."""
    settings = Settings()
    wake_word = wake_word or settings.wake_word
    engine = WakeWordEngine(
        model_path=str(WAKE_WORD_MODEL_PATH),
        threshold=settings.wake_threshold,
        wake_word=wake_word,
    )
    status = engine.status()
    backend = "onnx" if status["backend"] == "openwakeword" else status["backend"]
    score = status["last_score"]
    transcript_mode = "on" if settings.transcript_wake else "off"
    print(f"Wake phrase: {wake_word!r} (transcript wake: {transcript_mode})")
    print(f"Wake backend: {backend}")
    print(f"Model path: {status['model_path']}")
    print(f"Threshold: {status['threshold']:g}")
    print(f"Last score: {'none yet' if score is None else f'{score:.3f}'}")
    if status["load_error"]:
        print(f"Note: {status['load_error']}")
    return 0


def wake_probe(wake_word: str | None = None, attempts: int = 3) -> int:
    """Record real wake attempts and report what would actually wake it.

    Two detectors answer differently: transcript wake (the primary trigger)
    needs the spoken phrase to survive transcription; the ONNX sound model
    scores whatever phrase it was trained for. Each attempt reports both, and
    a threshold is suggested only when the sound model showed real signal.
    """
    if sd is None:
        raise RuntimeError("Wake probing requires sounddevice.")

    settings = Settings()
    wake_word = wake_word or settings.wake_word
    ensure_directories()
    engine = WakeWordEngine(
        model_path=str(WAKE_WORD_MODEL_PATH),
        threshold=settings.wake_threshold,
        wake_word=wake_word,
    )
    if engine.backend != "openwakeword":
        print(f"Note: the wake word model is not loaded ({engine.load_error or 'unknown reason'}).")
        print("Scores below come from the loudness fallback and are not comparable.")

    stt = None
    try:
        from nova_agent.core.stt import STTEngine

        stt = STTEngine(settings.stt_model, settings.stt_device, settings.stt_compute_type)
    except Exception as exc:  # noqa: BLE001 - probe still measures the model backend
        print(f"Note: transcript wake check unavailable ({exc}).")

    print(f"Say {wake_word!r} {attempts} times, pausing between attempts.")
    peaks: list[float] = []
    hits = 0
    block = settings.sample_block
    per_attempt = int(4.0 * settings.sample_rate / block)

    with sd.InputStream(
        samplerate=settings.sample_rate,
        blocksize=block,
        dtype="float32",
        channels=1,
    ) as stream:
        for attempt in range(attempts):
            print(f"  attempt {attempt + 1}/{attempts} -- speak now")
            peak = 0.0
            frames = []
            for _ in range(per_attempt):
                chunk, _overflow = stream.read(block)
                audio = np.asarray(chunk[:, 0], dtype=np.float32)
                frames.append(audio)
                peak = max(peak, engine.observe(audio))
            peaks.append(peak)
            print(f"    audio-model peak {peak:.3f}")
            if stt is None:
                continue
            probe_path = _write_probe_capture(np.concatenate(frames), settings)
            try:
                rms = _wav_rms(probe_path)
                if rms < settings.wake_segment_min_rms:
                    print(f"    too quiet to trust (rms={rms:.4f}) -> miss")
                    continue
                heard = stt.transcribe(
                    str(probe_path), prompt=wake_prompt_for(wake_word)
                )
            finally:
                probe_path.unlink(missing_ok=True)
            matched, _remainder = wake_phrase_hits(heard, wake_word)
            hits += int(matched)
            print(f"    heard {heard!r} -> transcript wake {'HIT' if matched else 'miss'}")

    return _probe_verdict(peaks, hits, attempts, stt is not None)


def _write_probe_capture(audio: np.ndarray, settings: Settings) -> Path:
    """Write one probe attempt to a uniquely named wav for transcription."""
    from uuid import uuid4

    from scipy.io import wavfile

    target = COMMAND_DIR / f"probe_{uuid4().hex[:8]}.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(target, settings.sample_rate, (audio * 32767).astype(np.int16))
    return target


def _probe_verdict(peaks: list[float], hits: int, attempts: int, have_stt: bool) -> int:
    """Summarize the probe: transcript hit rate first, threshold only if real."""
    worked = False
    if have_stt:
        print(f"Transcript wake: {hits}/{attempts} attempts contained the phrase.")
        if hits == attempts:
            print("Transcript wake is working; it needs no threshold.")
            worked = True
        elif hits:
            print("Partial detection: sit closer to the mic, or speak a bit louder.")
            worked = True
        else:
            print(
                "No attempt contained the phrase. Check the 'heard' lines above: "
                "empty/garbled text points at the mic, wrong words at the STT model."
            )

    peak_max = max(peaks) if peaks else 0.0
    suggested = WakeWordEngine.suggest_threshold(peaks) if peak_max >= 0.2 else None
    if suggested is not None:
        ordered = sorted(peaks)
        median = ordered[len(ordered) // 2]
        print(f"Peak scores: {', '.join(f'{value:.3f}' for value in peaks)}")
        print(
            f"Score distribution: min {min(peaks):.3f} / median {median:.3f} / "
            f"max {max(peaks):.3f} over {len(peaks)} attempts"
        )
        print(f"Suggested audio-model threshold: {suggested}")
        print(f"Use it with: python -m nova_agent --listen --wake-threshold {suggested}")
        worked = True
    elif not have_stt:
        print(
            "No usable wake-word signal was recorded and transcript wake is "
            "unavailable. Try again somewhere quieter."
        )
    else:
        print(
            "The sound model gave no usable score for this phrase (expected: it "
            "was trained for different words). Transcript wake needs no threshold."
        )
    return 0 if worked else 1


def prune_cache() -> int:
    """Delete TTS cache entries that can never be reused."""
    from nova_agent.core.tts import TTSEngine

    ensure_directories()
    removed = TTSEngine(cache_dir=CACHE_DIR).prune_cache()
    if removed:
        print(f"Removed {len(removed)} stale cache entries:")
        for name in removed:
            print(f"  {name}")
    else:
        print("No stale cache entries found.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Dude local voice agent")
    parser.add_argument(
        "--check", action="store_true", help="report the environment, detectors, and capabilities"
    )
    parser.add_argument(
        "--enroll-voice",
        action="store_true",
        help="record a short session and store the voiceprint for the wake voice gate",
    )
    parser.add_argument("--once", action="store_true", help="record and process one command")
    parser.add_argument("--listen", action="store_true", help="start the hands-free wake-word loop")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "two-step wizard: (a) ambient energy for the fallback threshold, "
            "(b) five wake-phrase attempts for transcript hits and model scores"
        ),
    )
    parser.add_argument(
        "--wake-status",
        action="store_true",
        help="print the wake backend, model path, threshold, and last score",
    )
    parser.add_argument(
        "--wake-probe",
        action="store_true",
        help="record wake attempts: report transcript hits, suggest a model threshold",
    )
    parser.add_argument(
        "--prune-cache",
        action="store_true",
        help="delete TTS cache entries that can never be reused",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="actually open apps and folders instead of printing a dry run",
    )
    parser.add_argument(
        "--no-hud", action="store_true", help="run without the on-screen status overlay"
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="override the speech-to-text device",
    )
    parser.add_argument(
        "--debug", action="store_true", help="print microphone energy, timings, and states"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="print a per-stage latency summary (min/median/mean/max vs budget) on exit",
    )
    parser.add_argument(
        "--wake-word",
        default=None,
        help="wake phrase to listen for (default: from settings)",
    )
    parser.add_argument(
        "--wake-threshold",
        type=float,
        default=None,
        help=(
            "override the wake trigger score (default: 0.65 with the openWakeWord "
            "model, 0.13 with the energy fallback)"
        ),
    )
    args = parser.parse_args()
    dry_run = False if args.live else Settings().dry_run
    if args.enroll_voice:
        raise SystemExit(0 if enroll_voice() else 1)
    if args.check:
        raise SystemExit(check_environment())
    if args.once:
        raise SystemExit(run_once(dry_run=dry_run))
    if args.calibrate:
        raise SystemExit(calibrate_wizard(wake_word=args.wake_word))
    if args.wake_status:
        raise SystemExit(wake_status(wake_word=args.wake_word))
    if args.wake_probe:
        raise SystemExit(wake_probe(wake_word=args.wake_word))
    if args.prune_cache:
        raise SystemExit(prune_cache())
    if args.listen:
        try:
            code = NovaAgent(
                debug=args.debug,
                stats=args.stats,
                wake_word=args.wake_word,
                wake_threshold=args.wake_threshold,
                # --live forces live; otherwise None lets run() resolve
                # spoken preference > NOVA_DRY_RUN env default.
                dry_run=False if args.live else None,
                hud_enabled=not args.no_hud,
                stt_device=args.device,
            ).run()
        except KeyboardInterrupt:
            # Backstop for a Ctrl+C that lands before the listen loop is up
            # (STT model load, microphone open): never a traceback on exit.
            print("\nStopped.")
            code = 0
        raise SystemExit(code)
    parser.print_help()
    print("\nCommon commands:")
    print("  python -m nova_agent --listen --live        talk to Dude, really act")
    print("  python -m nova_agent --wake-probe           tune the wake word to your voice")
    print("  python -m nova_agent --check                environment and capabilities")


if __name__ == "__main__":
    main()
