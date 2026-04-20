from app.models.owner import OwnerRecap
from app.utils.text import normalize_owner_code


def export_owner_row(recap: OwnerRecap) -> str:
    owner_code = normalize_owner_code(recap.owner_name)
    return f"owner_code={owner_code},projects={recap.total_projects}"
