"""--stats summarizes per-stage latency against the Phase 3 budget."""

import numpy as np

from nova_agent.main import LATENCY_BUDGETS, NovaAgent, format_stats


def test_format_stats_reports_counts_and_budget_verdicts():
    history = [
        {"stt": 1.7, "intent": 0.03, "tts": 1.0, "action": 0.01, "total": 3.9},
        {"stt": 3.5, "intent": 0.15, "tts": 2.0, "action": 0.4, "total": 16.0},
    ]

    report = format_stats(history)

    assert "over 2 command(s)" in report
    assert "over by 500ms" in report  # worst stt 3.5s vs 3.0s budget
    assert "over by 1000ms" in report  # worst total 16.0s vs 15.0s budget
    assert "Budget EXCEEDED on: stt, total" in report
    assert "intent  " in report and " ok" in report


def test_format_stats_accepts_a_run_that_meets_the_budget():
    history = [{"stt": 0.9, "intent": 0.04, "tts": 0.03, "action": 0.2, "total": 1.3}]

    report = format_stats(history)

    assert "Budget met on every stage." in report
    assert "EXCEEDED" not in report


def test_format_stats_uses_worst_case_not_average_for_the_verdict():
    # Mean total (9.5s) sits under the budget; the worst command is 16.0s.
    history = [
        {"stt": 0.2, "total": 3.0},
        {"stt": 0.8, "total": 16.0},
    ]

    report = format_stats(history)

    # The verdict follows the max, not the mean.
    assert "Budget EXCEEDED on: total" in report
    assert "stt" in report and " ok" in report  # stt worst 0.8s is within budget


def test_format_stats_without_commands_says_so():
    assert "no latency stats" in format_stats([])


def test_budgets_match_field_measurements():
    # Two field rounds: a 2-command probe, then 13 real commands (stt max
    # 2.8s, intent max 126ms warm, tts counts synthesis-to-audio only, total
    # is wall clock including speaking the reply — max 13.8s on the longest).
    assert LATENCY_BUDGETS == {
        "stt": 3.0,
        "intent": 0.2,
        "tts": 2.5,
        "action": 0.5,
        "total": 15.0,
    }


class _NeverRouter:
    def match(self, _text):
        return None, 0.0


class _SilentSTT:
    def transcribe(self, _audio_path, prompt=None, hotwords=None):
        return "hmm"


def test_tts_stage_counts_synthesis_not_playback():
    from nova_agent.main import CommandProcessor

    class ReportingTTS:
        last_synthesis_seconds = 0.42

        def speak(self, _message):
            pass

    processor = CommandProcessor(_SilentSTT(), _NeverRouter(), ReportingTTS())

    processor._respond("hi")

    assert processor.timings["tts"] == 0.42


def test_tts_stage_falls_back_to_wall_time_without_self_report():
    from nova_agent.main import CommandProcessor

    class PlainTTS:
        def speak(self, _message):
            pass

    processor = CommandProcessor(_SilentSTT(), _NeverRouter(), PlainTTS())

    processor._respond("hi")

    assert processor.timings["tts"] >= 0.0


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

    def transcribe(self, _audio_path, prompt=None, hotwords=None, command=False):
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


def test_commands_during_model_warm_up_are_not_counted(capsys):
    """A mid-warm command blocks on the lazy loads by design; counting it
    would pin the worst-case verdict to a boot artifact (field: intent
    1986ms, total 24843ms on the first command of the session)."""
    agent = NovaAgent(
        wake_engine=FakeWake(),
        vad_reader=FakeVAD(),
        processor=FakeProcessor(),
        hud=FakeHUD(),
        stats=True,
    )
    agent.warm_done.clear()

    agent.process_chunk(np.full(160, 0.5, dtype=np.float32))  # wake word
    agent.process_chunk(np.full(160, 0.5, dtype=np.float32))  # the command

    assert agent.stats_history == []
    assert "during model warm-up" in capsys.readouterr().out


def test_a_command_outliving_warm_on_the_tts_lock_is_still_skipped(capsys):
    """Field round 4: command #1 blocked on the encoder lock, then its TTS
    blocked on the warm-up's synthesis lock, so it *finished* after
    warm_done.set() — an end-only check counted it (intent 19088ms, total
    33010ms pinned the verdict). The verdict is taken at command start."""

    class WarmFinishesMidCommand(FakeProcessor):
        on_run = None

        def run_observation(self, observation, confirmed=False):
            if self.on_run is not None:
                self.on_run()  # warm-up completes while the command runs
            return super().run_observation(observation, confirmed)

    processor = WarmFinishesMidCommand()
    agent = NovaAgent(
        wake_engine=FakeWake(),
        vad_reader=FakeVAD(),
        processor=processor,
        hud=FakeHUD(),
        stats=True,
    )
    agent.warm_done.clear()
    processor.on_run = lambda: agent.warm_done.set()

    agent.process_chunk(np.full(160, 0.5, dtype=np.float32))  # wake word
    agent.process_chunk(np.full(160, 0.5, dtype=np.float32))  # the command

    assert agent.stats_history == []
    assert "during model warm-up" in capsys.readouterr().out


def test_wake_segment_transcription_does_not_leak_into_a_text_command():
    """_wake_from_transcript runs idle segments through processor.transcribe,
    writing timings["stt"]; a text-path command (transcript-wake remainder)
    never transcribes, so without a per-command reset its snapshot inherited
    the wake segment's stt (field round 4: a 10888ms idle segment in a
    command row)."""
    processor = FakeProcessor()
    agent = NovaAgent(
        wake_engine=FakeWake(),
        vad_reader=FakeVAD(),
        processor=processor,
        hud=FakeHUD(),
        stats=True,
    )
    processor.transcribe("idle_segment.wav")  # wake-phase transcription

    agent._run_captured(text="time")  # text path: this command never transcribes

    snapshot = agent.stats_history[0]
    assert "stt" not in snapshot
    assert snapshot["intent"] == 0.01
    assert snapshot["total"] > 0
