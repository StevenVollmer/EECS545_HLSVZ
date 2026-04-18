from app.models.change import Change
from app.services.release_service import public_change_titles
from app.utils.text import join_titles


def render_notes(changes: list[Change]) -> str:
    return f"public highlights: {join_titles(public_change_titles(changes))}"
