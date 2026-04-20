from app.models.owner import OwnerRecap


def build_owner_recap(owner_name: str, total_projects: int) -> OwnerRecap:
    return OwnerRecap(owner_name=owner_name, total_projects=total_projects)
