from app.models.probe import Probe


def failing_probe_names(probes: list[Probe]) -> list[str]:
    return [probe.name for probe in probes if probe.failing]
