from nova_agent.core.runtime_monitor import RuntimeMonitor


def test_runtime_monitor_tracks_health_and_metrics():
    monitor = RuntimeMonitor()

    monitor.record_metric("cpu", 30)
    monitor.record_metric("memory", 55)
    monitor.record_metric("gpu", 5)

    assert monitor.health() == "healthy"
    assert monitor.snapshot()["cpu"] == 30


def test_runtime_monitor_detects_unhealthy_range():
    monitor = RuntimeMonitor()
    monitor.record_metric("cpu", 95)
    monitor.record_metric("memory", 90)

    assert monitor.health() == "warning"
