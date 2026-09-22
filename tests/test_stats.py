"""--stats summarizes per-stage latency against the Phase 3 budget."""

import numpy as np

from nova_agent.main import LATENCY_BUDGETS, NovaAgent, format_stats


def test_format_stats_reports_counts_and_budget_verdicts():
    history = [
        {"stt": 1.0, "intent": 0.05, "tts": 0.04, "action": 0.3, "total": 1.5},
        {"stt": 1.4, "intent": 0.06, "tts": 0.04, "action": 0.4, "total": 2.4},
    ]

    report = format_stats(history)

    assert "over 2 command(s)" in report
    assert "over by 200ms" in report  # worst stt 1.4s vs 1.2s budget
    assert "over by 400ms" in report  # worst total 2.4s vs 2.0s budget
    assert "Budget EXCEEDED on: stt, total" in report
    assert "intent  " in report and " ok" in report


def test_format_stats_accepts_a_run_that_meets_the_budget():
    history = [{"stt": 0.9, "intent": 0.04, "tts": 0.03, "action": 0.2, "total": 1.3}]

    report = format_stats(history)

    assert "Budget met on every stage." in report
    assert "EXCEEDED" not in report


def test_format_stats_uses_worst_case_not_average_for_the_verdict():
    # Mean total (2.0s) sits exactly at the budget; the worst command is 3.0s.
    history = [
        {"stt": 0.2, "total": 1.0},
        {"stt": 0.8, "total": 3.0},
    ]

    report = format_stats(history)

    # The verdict follows the max, not the mean.
    assert "Budget EXCEEDED on: total" in report
    assert "stt" in report and " ok" in report  # stt worst 0.8s is within budget


def test_format_stats_without_commands_says_so():
    assert "no latency stats" in format_stats([])


def test_budgets_match_the_roadmap_targets():
    assert LATENCY_BUDGETS == {
        "stt": 1.2,
        "intent": 0.1,
        "tts": 0.05,
        "action": 0.5,
        "total": 2.0,
    }


class FakeWake:
    def process_chunk(self, _chunk):
        return True

    def reset(self):
        pass


class FakeVAD:
    def __init__(self):
        self.calls = 0

    def process_chunk(self, _chunk):
        self.calls += 1
        return "command.wav" if self.calls == 1 else None

    def reset(self):
        pass


class FakeProcessor:
    def __init__(self):
        self.pending_confirmation = None
        self.timings = {}

    def transcribe(self, _audio_path, prompt=None):
        self.timings["stt"] = 0.01
        return "open chrome"

    def observe(self, text):
        self.timings["intent"] = 0.01
        return {
            "text": text,
            "intent": {"action": "open_app", "target": "chrome"},
            "score": 0.9,
        }

    def run_observation(self, _observation, confirmed=False):
        self.timings["tts"] = 0.01
        self.timings["action"] = 0.01
        return "Would open chrome"


class FakeHUD:
    def set_state(self, _state, _message=""):
        pass

    def show_transcript(self, _text):
        pass


def test_agent_records_a_latency_snapshot_per_command():
    agent = NovaAgent(
        wake_engine=FakeWake(),
        vad_reader=FakeVAD(),
        processor=FakeProcessor(),
        hud=FakeHUD(),
        stats=True,
    )

    agent.process_chunk(np.full(160, 0.5, dtype=np.float32))  # wake word
    agent.process_chunk(np.full(160, 0.5, dtype=np.float32))  # the command

    assert len(agent.stats_history) == 1
    snapshot = agent.stats_history[0]
    assert snapshot["stt"] == 0.01
    assert snapshot["intent"] == 0.01
    assert snapshot["action"] == 0.01
    assert snapshot["total"] > 0


def test_stats_history_stays_empty_when_stats_is_off():
    agent = NovaAgent(
        wake_engine=FakeWake(),
        vad_reader=FakeVAD(),
        processor=FakeProcessor(),
        hud=FakeHUD(),
        stats=False,
    )

    agent.process_chunk(np.full(160, 0.5, dtype=np.float32))
    agent.process_chunk(np.full(160, 0.5, dtype=np.float32))

    assert agent.stats_history == []
