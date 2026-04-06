from app.utils.text import punctuated_title


def render_contact_card(name: str, team: str) -> str:
    return f"Primary Contact: {punctuated_title(name)} ({team})"
