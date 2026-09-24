"""Reminders (Wave 1c): a spoken nudge, stored in SQLite, fired from the worker.

The pieces, bottom-up:

* ``parse_reminder`` — "remind me to stretch in 20 minutes" -> ("stretch", 1200),
  including spoken numbers ("ten minutes"), "an hour", "half an hour", and a
  reminder that itself mentions a place (split on the *last* "in").
* ``ContextEngine`` — the row store: add, list pending, pop due, clear.
* ``CommandProcessor`` — the ``set_reminder`` / ``manage_reminders`` actions.
* ``NovaAgent._fire_due_reminders`` — speaks what is due and marks it fired,
  once, through the same playback path commands use (so barge-in applies).

Like ``set_project``/``set_browser``, the set action is a SQLite memory write
plus a later spoken reply — neither is a world action, so it behaves the same
in dry-run and live mode (the default soak can exercise it).
"""

import time

from nova_agent.core.context_engine import ContextEngine
from nova_agent.core.query_extractor import humanize_delay, parse_reminder
from nova_agent.main import CommandProcessor, NovaAgent

# --- parsing -----------------------------------------------------------------


def test_the_common_shapes_parse():
    assert parse_reminder("remind me to take medicine in 10 minutes") == (
        "take medicine",
        600,
    )
    assert parse_reminder("remind me to stretch in an hour") == ("stretch", 3600)
    assert parse_reminder("remind me to drink water in 30 minutes") == (
        "drink water",
        1800,
    )
    assert parse_reminder("reminder to call mom in 5 minutes") == ("call mom", 300)


def test_spoken_numbers_and_small_units_parse():
    assert parse_reminder("remind me to blink in ten minutes") == ("blink", 600)
    assert parse_reminder("remind me to check the oven in 30 seconds") == (
        "check the oven",
        30,
    )
    assert parse_reminder("remind me to pause in half an hour") == ("pause", 1800)


def test_it_splits_on_the_last_in():
    # The reminder itself mentions a place; the duration is still the last "in".
    assert parse_reminder("remind me to meet john in the lab in 10 minutes") == (
        "meet john in the lab",
        600,
    )


def test_non_reminders_and_missing_durations_are_rejected():
    assert parse_reminder("open chrome") is None
    assert parse_reminder("remind me to stretch") is None  # no duration
    assert parse_reminder("remind me to stretch in a while") is None  # unparseable
    assert parse_reminder("remind me in 10 minutes") is None  # nothing to remind


def test_delay_phrases_read_naturally():
    assert humanize_delay(3600) == "an hour"
    assert humanize_delay(7200) == "2 hours"
    assert humanize_delay(60) == "a minute"
    assert humanize_delay(300) == "5 minutes"
    assert humanize_delay(30) == "30 seconds"


# --- storage -----------------------------------------------------------------


def test_a_reminder_round_trips_from_add_to_fired(tmp_path):
    context = ContextEngine(tmp_path / "nova.db")
    now = int(time.time())

    reminder_id = context.add_reminder("stretch", now + 60)

    assert context.due_reminders(now) == []  # not yet due
    assert context.pending_reminders(now) == [("stretch", now + 60)]
    assert context.due_reminders(now + 60) == [(reminder_id, "stretch")]

    context.mark_reminder_fired(reminder_id)

    assert context.due_reminders(now + 60) == []
    assert context.pending_reminders(now) == []


def test_clearing_drops_every_pending_reminder(tmp_path):
    context = ContextEngine(tmp_path / "nova.db")
    now = int(time.time())
    context.add_reminder("first", now + 10)
    context.add_reminder("second", now + 20)

    assert context.clear_reminders() == 2
    assert context.pending_reminders(now) == []
    assert context.clear_reminders() == 0


# --- the actions -------------------------------------------------------------


class FakeSTT:
    def transcribe(self, _audio_path, prompt=None, hotwords=None):
        return ""


class FakeRouter:
    def match(self, _text):
        return {"action": "set_reminder", "safe": True}, 0.9


class FakeTTS:
    def __init__(self):
        self.spoken = []

    def speak(self, message):
        self.spoken.append(message)


def _processor(tmp_path):
    return CommandProcessor(
        FakeSTT(),
        FakeRouter(),
        FakeTTS(),
        context=ContextEngine(tmp_path / "nova.db"),
    )


def _dispatch(processor, text, action):
    return processor.run_observation(
        {"text": text, "intent": {"action": action, "safe": True}, "score": 0.9}
    )


def test_set_reminder_stores_it_and_confirms(tmp_path):
    processor = _processor(tmp_path)
    before = int(time.time())

    reply = _dispatch(processor, "remind me to stretch in 10 minutes", "set_reminder")

    assert reply == "Okay — I'll remind you to stretch in 10 minutes."
    pending = processor.context.pending_reminders(int(time.time()))
    assert len(pending) == 1
    what, fire_at = pending[0]
    assert what == "stretch"
    assert before + 600 <= fire_at <= before + 610


def test_set_reminder_without_a_duration_asks_for_one(tmp_path):
    processor = _processor(tmp_path)

    reply = _dispatch(processor, "remind me to stretch", "set_reminder")

    assert reply == "Tell me when — like: remind me to stretch in 20 minutes."
    assert processor.context.pending_reminders(int(time.time())) == []


def test_listing_reads_back_what_is_pending(tmp_path):
    processor = _processor(tmp_path)
    _dispatch(processor, "remind me to stretch in 10 minutes", "set_reminder")
    _dispatch(processor, "remind me to drink water in 30 minutes", "set_reminder")

    reply = _dispatch(processor, "what are my reminders", "manage_reminders")

    assert reply == "You have 2 reminders pending: stretch, drink water."


def test_listing_when_empty_says_so(tmp_path):
    processor = _processor(tmp_path)

    reply = _dispatch(processor, "what are my reminders", "manage_reminders")

    assert reply == "You have no reminders pending."


def test_cancelling_clears_them(tmp_path):
    processor = _processor(tmp_path)
    _dispatch(processor, "remind me to stretch in 10 minutes", "set_reminder")

    reply = _dispatch(processor, "cancel my reminders", "manage_reminders")

    assert reply == "Cleared 1 reminder."
    assert processor.context.pending_reminders(int(time.time())) == []


def test_cancelling_with_nothing_to_clear_says_so(tmp_path):
    processor = _processor(tmp_path)

    reply = _dispatch(processor, "clear my reminders", "manage_reminders")

    assert reply == "You have no reminders to clear."


# --- firing ------------------------------------------------------------------


class _FireTTS:
    def __init__(self):
        self.spoken = []

    def speak(self, message):
        self.spoken.append(message)


class _FireProcessor:
    def __init__(self):
        self.tts = _FireTTS()


class _FireHud:
    def __init__(self):
        self.states = []

    def set_state(self, state, detail=""):
        self.states.append((state, detail))


def _agent(tmp_path):
    return NovaAgent(
        wake_engine=object(),
        vad_reader=object(),
        processor=_FireProcessor(),
        hud=_FireHud(),
        monitor=object(),
        context=ContextEngine(tmp_path / "nova.db"),
    )


def test_a_due_reminder_is_spoken_exactly_once(tmp_path):
    agent = _agent(tmp_path)
    agent.context.add_reminder("stretch", int(time.time()) - 1)  # already due

    agent._fire_due_reminders()

    assert agent.processor.tts.spoken == ["Reminder: stretch"]
    assert ("speaking", "Reminder") in agent.hud.states
    # Marked fired, so a later check never repeats it.
    assert agent.context.due_reminders(int(time.time())) == []
    agent._fire_due_reminders()
    assert agent.processor.tts.spoken == ["Reminder: stretch"]


def test_nothing_due_says_nothing(tmp_path):
    agent = _agent(tmp_path)
    agent.context.add_reminder("stretch", int(time.time()) + 600)  # future

    agent._fire_due_reminders()

    assert agent.processor.tts.spoken == []
