from app.models.incident import Incident


def visible_incidents(incidents: list[Incident]) -> list[Incident]:
    return [incident for incident in incidents if incident.is_open]


def visible_count(incidents: list[Incident]) -> int:
    return len(visible_incidents(incidents))
