from app.models.incident import Incident
from app.presenters.brief_presenter import render_incident_brief
from app.services.brief_builder import build_brief


def render_team_brief(team_name: str, incidents: list[Incident]) -> str:
    brief = build_brief(team_name, incidents)
    return render_incident_brief(brief)
