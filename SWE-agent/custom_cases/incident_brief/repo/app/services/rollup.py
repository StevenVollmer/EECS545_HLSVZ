from app.models.incident import Incident
from app.utils.severity import is_urgent


def count_open_incidents(incidents: list[Incident]) -> int:
    return sum(1 for incident in incidents if incident.state == "open")


def count_urgent_incidents(incidents: list[Incident]) -> int:
    return sum(1 for incident in incidents if incident.state == "open" and is_urgent(incident.severity))
