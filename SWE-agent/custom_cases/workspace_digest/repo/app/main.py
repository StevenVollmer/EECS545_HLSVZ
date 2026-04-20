from app.models.workspace import AlertItem, Workspace
from app.presenters.dashboard_presenter import render_dashboard
from app.services.digest_builder import build_workspace_digest


def render_workspace_dashboard(name: str, alerts: list[AlertItem]) -> str:
    workspace = Workspace(name=name, alerts=alerts)
    digest = build_workspace_digest(workspace)
    return render_dashboard(digest)
