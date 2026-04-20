from app.models.incident import IncidentBrief


def render_incident_brief(brief: IncidentBrief) -> str:
    return (
        f"Team {brief.team_name}: {brief.urgent_count} urgent items "
        f"({brief.critical_count} critical, {brief.open_count} open)"
    )
