from nova_agent.core.capabilities import CapabilityRegistry


def test_capability_registry_tracks_enabled_actions():
    registry = CapabilityRegistry()

    registry.register("open_app", enabled=True, safe=True)
    registry.register("delete_file", enabled=True, safe=False)

    assert registry.is_enabled("open_app") is True
    assert registry.is_safe("delete_file") is False
    assert registry.list_enabled() == ["open_app", "delete_file"]


def test_capability_registry_rejects_unknown_action():
    registry = CapabilityRegistry()

    assert registry.is_enabled("unknown_action") is False
    assert registry.is_safe("unknown_action") is True
