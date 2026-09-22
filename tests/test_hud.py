from nova_agent.ui.overlay import NovaHUD


def test_hud_state_sets_label_and_color():
    hud = NovaHUD()
    hud.set_state("listening", "Listening")

    assert hud.current_state == "listening"
    assert hud.status_text == "Listening"
    assert hud.color == "#00d4ff"


def test_hud_defaults_to_idle():
    hud = NovaHUD()

    assert hud.current_state == "idle"
    assert hud.status_text == "idle"
    assert hud.color == "#333333"
