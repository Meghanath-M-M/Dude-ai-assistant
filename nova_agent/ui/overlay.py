class NovaHUD:
    """Placeholder UI contract; PyQt implementation belongs to the HUD phase."""

    def set_state(self, state: str, message: str = "") -> None:
        print(f"HUD [{state}] {message}".rstrip())
