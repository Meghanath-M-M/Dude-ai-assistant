from __future__ import annotations


class RuntimeMonitor:
    """Tracks live resource health for the local assistant runtime."""

    def __init__(self):
        self.metrics: dict[str, float] = {}

    def record_metric(self, name: str, value: float) -> None:
        self.metrics[name] = float(value)

    def snapshot(self) -> dict[str, float]:
        return dict(self.metrics)

    def health(self) -> str:
        if not self.metrics:
            return "healthy"

        cpu = self.metrics.get("cpu", 0)
        memory = self.metrics.get("memory", 0)
        gpu = self.metrics.get("gpu", 0)

        if cpu >= 90 or memory >= 90 or gpu >= 90:
            return "warning"
        return "healthy"
