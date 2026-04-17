from app.models.incident import Incident
from app.services.incident_service import visible_count


def render_incident_brief(incidents: list[Incident]) -> str:
    return f"open alerts: {visible_count(incidents)}"
