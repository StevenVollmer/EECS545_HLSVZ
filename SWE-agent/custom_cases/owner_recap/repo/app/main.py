from app.exports.csv_writer import export_owner_row
from app.models.owner import OwnerRecap
from app.presenters.recap_presenter import render_owner_preview


def preview_owner_recap(owner_name: str, total_projects: int) -> str:
    recap = OwnerRecap(owner_name=owner_name, total_projects=total_projects)
    return render_owner_preview(recap)


def export_owner(owner_name: str, total_projects: int) -> str:
    recap = OwnerRecap(owner_name=owner_name, total_projects=total_projects)
    return export_owner_row(recap)
