from app.models.incident import Incident
from app.services.incident_service import active_unacked_count
from app.utils.labels import render_count_label


def render_brief(incidents: list[Incident]) -> str:
    return render_count_label("unacked incidents", active_unacked_count(incidents), "incident")
