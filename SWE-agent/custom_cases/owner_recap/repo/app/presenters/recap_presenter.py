from app.models.owner import OwnerRecap
from app.utils.text import normalize_owner_code


def render_owner_preview(recap: OwnerRecap) -> str:
    display_name = normalize_owner_code(recap.owner_name)
    return f"Owner: {display_name} | Projects: {recap.total_projects}"
