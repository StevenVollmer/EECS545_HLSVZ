from app.models.incident import Incident


def active_unacked_count(incidents: list[Incident]) -> int:
    return len([incident for incident in incidents if incident.active])
