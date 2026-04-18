from app.models.probe import Probe
from app.services.uptime_service import failing_probe_names
from app.utils.text import join_names


def render_rollup(probes: list[Probe]) -> str:
    return f"visible failing probes: {join_names(failing_probe_names(probes))}"
