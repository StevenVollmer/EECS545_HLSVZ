from app.utils.names import lead_display_name


def roster_signature(team: str, lead: str) -> str:
    return f"{team}: lead {lead_display_name(lead)}"


def export_team_code(team: str) -> str:
    return f"team={team.strip().upper()}"
