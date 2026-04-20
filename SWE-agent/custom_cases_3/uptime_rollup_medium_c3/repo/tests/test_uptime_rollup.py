from app.models.probe import Probe
from app.presenters.rollup_presenter import render_rollup
from app.services.uptime_service import failing_probe_names


def test_failing_probe_names_excludes_suppressed_probes() -> None:
    probes = [
        Probe("api", failing=True, suppressed=False),
        Probe("legacy", failing=True, suppressed=True),
    ]
    assert failing_probe_names(probes) == ["api"]


def test_presenter_uses_service_output() -> None:
    assert render_rollup([Probe("api", failing=True, suppressed=False)]) == "visible failing probes: api"


def test_non_matching_rows_do_not_count() -> None:
    assert failing_probe_names([Probe("docs", failing=False, suppressed=False)]) == []
