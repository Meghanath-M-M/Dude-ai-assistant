class NovaHUD:
    """Simple HUD state machine for the local assistant."""

    COLORS = {
        "idle": "#333333",
        "listening": "#00d4ff",
        "thinking": "#ffaa00",
        "speaking": "#00ff88",
        "error": "#ff4444",
    }

    def __init__(self):
        self.current_state = "idle"
        self.status_text = "idle"
        self.color = self.COLORS["idle"]

    def set_state(self, state: str, message: str = "") -> None:
        self.current_state = state
        self.status_text = message or state
        self.color = self.COLORS.get(state, "#ffffff")
        print(f"HUD [{state}] {self.status_text}".rstrip())
