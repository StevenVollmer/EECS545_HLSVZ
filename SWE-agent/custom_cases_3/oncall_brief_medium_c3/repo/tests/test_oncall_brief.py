from app.models.incident import Incident
from app.presenters.brief_presenter import render_brief
from app.services.incident_service import active_unacked_count


def test_active_unacked_count_excludes_acknowledged_incidents() -> None:
    incidents = [
        Incident("INC-7", active=True, acknowledged=False),
        Incident("INC-8", active=True, acknowledged=True),
    ]
    assert active_unacked_count(incidents) == 1


def test_presenter_uses_service_output() -> None:
    assert render_brief([Incident("INC-7", active=True, acknowledged=False)]) == "unacked incidents: 1 incident"


def test_non_matching_rows_do_not_count() -> None:
    assert active_unacked_count([Incident("INC-9", active=False, acknowledged=False)]) == 0
