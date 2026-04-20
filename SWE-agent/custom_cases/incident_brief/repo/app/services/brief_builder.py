from app.models.incident import Incident, IncidentBrief
from app.services.rollup import count_open_incidents, count_urgent_incidents
from app.utils.severity import is_critical


def build_brief(team_name: str, incidents: list[Incident]) -> IncidentBrief:
    return IncidentBrief(
        team_name=team_name,
        urgent_count=count_urgent_incidents(incidents),
        open_count=count_open_incidents(incidents),
        critical_count=sum(1 for incident in incidents if incident.state == "open" and is_critical(incident.severity)),
    )
