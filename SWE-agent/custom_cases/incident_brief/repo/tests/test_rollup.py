from app.models.incident import Incident
from app.services.rollup import count_open_incidents, count_urgent_incidents


def test_count_open_incidents_counts_all_open_items() -> None:
    incidents = [
        Incident(title="api", severity="critical", state="open", muted=False),
        Incident(title="disk", severity="low", state="open", muted=True),
        Incident(title="audit", severity="warning", state="closed", muted=False),
    ]
    assert count_open_incidents(incidents) == 2


def test_count_urgent_incidents_counts_warning_incidents() -> None:
    incidents = [
        Incident(title="api", severity="warning", state="open", muted=False),
        Incident(title="audit", severity="info", state="open", muted=False),
    ]
    assert count_urgent_incidents(incidents) == 1


def test_count_urgent_incidents_ignores_closed_incidents() -> None:
    incidents = [
        Incident(title="api", severity="critical", state="closed", muted=False),
    ]
    assert count_urgent_incidents(incidents) == 0
