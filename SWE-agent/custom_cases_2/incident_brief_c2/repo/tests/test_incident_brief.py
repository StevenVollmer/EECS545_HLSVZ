from app.models.incident import Incident
from app.presenters.brief_presenter import render_incident_brief
from app.services.incident_service import visible_incidents


def test_visible_incidents_excludes_silenced() -> None:
    incidents = [
        Incident(title="API latency", severity=2, is_open=True, silenced=True),
        Incident(title="Queue backlog", severity=3, is_open=True, silenced=False),
    ]
    assert [incident.title for incident in visible_incidents(incidents)] == ["Queue backlog"]


def test_brief_renders_visible_alert_count() -> None:
    incidents = [
        Incident(title="API latency", severity=2, is_open=True, silenced=False),
        Incident(title="DB errors", severity=3, is_open=False, silenced=False),
    ]
    assert render_incident_brief(incidents) == "open alerts: 1"


def test_closed_incidents_not_visible() -> None:
    incidents = [Incident(title="Cache miss", severity=1, is_open=False, silenced=False)]
    assert visible_incidents(incidents) == []
