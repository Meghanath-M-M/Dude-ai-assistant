"""Wave 2: the LLM tool-calling brain (``NOVA_LLM_BRAIN=1``).

Bottom-up: the schema builder, then ``LLMBrain.decide`` against a faked Ollama
stream (tool call / conversational reply / every failure mode), then the
``CommandProcessor`` integration — Tier 1 stays the untouched fast path, the
brain only runs when Tier 1 resolved nothing, its time lands in a separate
``llm`` stage, and a streamed reply is spoken **once** (never re-spoken by
``_respond``).

No Ollama and no model are required: the client is faked via ``monkeypatch``.
"""

import json
from pathlib import Path

from nova_agent.config.settings import Settings
from nova_agent.core import llm_brain
from nova_agent.core.capabilities import CapabilityRegistry
from nova_agent.core.llm_brain import LLMBrain, build_tool_schemas, take_sentences
from nova_agent.main import IMPLEMENTED_ACTIONS, CommandProcessor

INTENTS = json.loads(
    (Path(__file__).resolve().parents[1] / "nova_agent" / "config" / "intents.json").read_text(
        encoding="utf-8"
    )
)


# --- schemas -----------------------------------------------------------------


def test_every_implemented_intent_becomes_a_tool():
    schemas = build_tool_schemas(INTENTS, IMPLEMENTED_ACTIONS)

    assert len(schemas) == len(INTENTS)  # all 20 map to a runnable action
    names = [schema["function"]["name"] for schema in schemas]
    assert "system_volume" in names and "remind_me" in names
    # Parameters are deliberately empty: the transcript carries the arguments.
    assert all(schema["function"]["parameters"]["properties"] == {} for schema in schemas)


def test_an_unrunnable_action_is_never_offered_as_a_tool():
    schemas = build_tool_schemas({"bad": {"action": "not_a_real_action"}}, IMPLEMENTED_ACTIONS)

    assert schemas == []


# --- sentence flushing -------------------------------------------------------


def test_flushes_on_sentence_boundaries_and_keeps_the_remainder():
    sentences, rest = take_sentences("Hello there. And now ", force=False)

    assert sentences == ["Hello there."]
    assert rest == "And now "


def test_force_flushes_a_trailing_fragment_with_no_period():
    assert take_sentences("no period here", force=True) == (["no period here"], "")


def test_a_punctuation_free_run_is_cut_so_speech_is_never_starved():
    long_run = "word " * 60  # 240 chars, past MAX_SENTENCE_CHARS, no period
    sentences, rest = take_sentences(long_run, force=False)

    assert sentences  # something was spoken
    assert len(sentences[0]) <= llm_brain.MAX_SENTENCE_CHARS
    assert rest.strip()  # the tail is carried, not dropped


# --- decide (faked Ollama) ---------------------------------------------------


def _brain(capabilities=None):
    return LLMBrain(
        capabilities=capabilities, intents=INTENTS, implemented=IMPLEMENTED_ACTIONS
    )


class _ToolOllama:
    """A model that calls exactly one tool."""

    class Client:
        def __init__(self, **_kwargs):
            pass

        def chat(self, **_kwargs):
            return iter(
                [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {"function": {"name": "system_volume", "arguments": {}}}
                            ],
                        },
                        "done": True,
                    }
                ]
            )


class _ReplyOllama:
    """A model that streams a conversational answer instead of calling a tool."""

    class Client:
        def __init__(self, **_kwargs):
            pass

        def chat(self, **_kwargs):
            return iter(
                [
                    {"message": {"content": "Sure thing. "}, "done": False},
                    {"message": {"content": "That is set now."}, "done": True},
                ]
            )


def test_a_tool_call_resolves_to_that_intents_config(monkeypatch):
    monkeypatch.setattr(llm_brain, "ollama", _ToolOllama)

    decision = _brain().decide("turn it down")

    assert decision.tool_intent is not None
    assert decision.tool_intent["action"] == "system_control"
    assert decision.tool_intent["target"] == "volume"
    assert decision.tool_intent["source"] == "brain"
    assert decision.reply_text is None


def test_a_disabled_action_stays_a_miss(monkeypatch):
    monkeypatch.setattr(llm_brain, "ollama", _ToolOllama)
    registry = CapabilityRegistry()
    registry.register("system_control", enabled=False, safe=True)

    decision = _brain(capabilities=registry).decide("turn it down")

    assert decision.is_miss


def test_a_hallucinated_tool_name_is_a_miss(monkeypatch):
    class _HallucinatingOllama:
        class Client:
            def __init__(self, **_kwargs):
                pass

            def chat(self, **_kwargs):
                return iter(
                    [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {"function": {"name": "make_sandwich", "arguments": {}}}
                                ],
                            },
                            "done": True,
                        }
                    ]
                )

    monkeypatch.setattr(llm_brain, "ollama", _HallucinatingOllama)

    assert _brain().decide("hi").is_miss


def test_a_conversational_reply_streams_sentence_by_sentence(monkeypatch):
    monkeypatch.setattr(llm_brain, "ollama", _ReplyOllama)
    spoken = []

    decision = _brain().decide("what just happened", on_sentence=spoken.append)

    assert decision.reply_text == "Sure thing. That is set now."
    assert decision.tool_intent is None
    assert spoken == ["Sure thing.", "That is set now."]


def test_no_ollama_installed_is_a_miss_not_a_crash(monkeypatch):
    monkeypatch.setattr(llm_brain, "ollama", None)

    assert _brain().decide("hello").is_miss


def test_a_blowing_up_client_is_a_miss_not_a_crash(monkeypatch):
    class _Boom:
        class Client:
            def __init__(self, **_kwargs):
                pass

            def chat(self, **_kwargs):
                raise RuntimeError("ollama is down")

    monkeypatch.setattr(llm_brain, "ollama", _Boom)

    assert _brain().decide("hello").is_miss


# --- CommandProcessor integration -------------------------------------------


class FakeSTT:
    def __init__(self, text=""):
        self.text = text

    def transcribe(self, _audio_path, prompt=None, hotwords=None):
        return self.text


class FakeTTS:
    def __init__(self):
        self.messages = []

    def speak(self, message):
        self.messages.append(message)


class MissRouter:
    """Tier 1 resolves nothing — the brain's cue to take over."""

    def __init__(self):
        self.intents = INTENTS
        self.last_best_label = "help"
        self.last_best_score = 0.5

    def match(self, _text):
        return None, 0.5


class HitRouter:
    """Tier 1 is confident — the brain must never run."""

    def __init__(self):
        self.intents = INTENTS
        self.calls = 0

    def match(self, _text):
        self.calls += 1
        return {"action": "greet", "safe": True}, 0.95


def _processor(router, monkeypatch, ollama_module):
    monkeypatch.setattr(llm_brain, "ollama", ollama_module)
    return CommandProcessor(
        FakeSTT(),
        router,
        FakeTTS(),
        settings=Settings(llm_brain=True),
    )


def test_the_brain_rescues_a_tier1_miss_with_a_tool_call(monkeypatch):
    processor = _processor(MissRouter(), monkeypatch, _ToolOllama)

    observation = processor.observe("turn it down")

    assert observation["intent"]["action"] == "system_control"
    assert observation["intent"]["source"] == "brain"
    # The brain's latency is its own stage, separate from the Tier 1 intent stage.
    assert "llm" in processor.timings
    assert "intent" in processor.timings


def test_a_streamed_reply_is_spoken_once_not_twice(monkeypatch):
    processor = _processor(MissRouter(), monkeypatch, _ReplyOllama)

    observation = processor.observe("what just happened")
    response = processor.run_observation(observation)

    assert observation["intent"]["action"] == "llm_reply"
    # Spoken sentence by sentence during observe; _respond must not repeat it.
    assert processor.tts.messages == ["Sure thing.", "That is set now."]
    assert response == "Sure thing. That is set now."


def test_a_confident_tier1_match_never_reaches_the_brain(monkeypatch):
    router = HitRouter()
    processor = _processor(router, monkeypatch, _ToolOllama)

    observation = processor.observe("hello")

    assert observation["intent"]["action"] == "greet"
    assert "llm" not in processor.timings  # brain never engaged
    assert router.calls == 1


def test_a_below_band_miss_still_asks_honestly_when_the_brain_misses(monkeypatch):
    class _SilentOllama:
        class Client:
            def __init__(self, **_kwargs):
                pass

            def chat(self, **_kwargs):
                return iter([{"message": {"content": ""}, "done": True}])

    processor = _processor(MissRouter(), monkeypatch, _SilentOllama)

    observation = processor.observe("wibble wobble")

    assert observation["intent"] is None
    response = processor.run_observation(observation)
    assert response == "I didn't catch that"


def test_the_brain_is_off_unless_enabled():
    plain = CommandProcessor(FakeSTT(), MissRouter(), FakeTTS())
    enabled = CommandProcessor(
        FakeSTT(), MissRouter(), FakeTTS(), settings=Settings(llm_brain=True)
    )

    assert plain.llm_brain is None
    assert enabled.llm_brain is not None
    assert enabled.llm_brain.tools  # built from intents x IMPLEMENTED_ACTIONS


def test_build_router_skips_the_tier2_fallback_when_the_brain_is_on():
    from nova_agent.main import build_router

    settings = Settings(llm_fallback=True, llm_brain=True)

    router = build_router(settings, CapabilityRegistry())

    assert router.fallback is None  # brain supersedes the string-parsing fallback
