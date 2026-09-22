from nova_agent.core.audio_session import AudioSession


def test_audio_session_transitions_cleanly():
    session = AudioSession()

    session.start_listening()
    assert session.state == "listening"

    session.start_capture()
    assert session.state == "capturing"

    session.finish_capture()
    assert session.state == "processing"

    session.reset()
    assert session.state == "idle"


def test_audio_session_tracks_event_history():
    session = AudioSession()
    session.start_listening()
    session.start_capture()

    assert session.events[-2:] == ["listening", "capturing"]
