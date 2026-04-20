from app.models.workspace import Workspace, WorkspaceDigest
from app.services.board_rollup import alert_rollup
from app.utils.filters import actionable_alerts


def build_workspace_digest(workspace: Workspace) -> WorkspaceDigest:
    rollup = alert_rollup(workspace.alerts)
    actionable = actionable_alerts(workspace.alerts)
    return WorkspaceDigest(
        name=workspace.name,
        visible_alert_count=len(workspace.alerts),
        open_alert_count=rollup["open"],
        critical_alert_count=len([alert for alert in actionable if alert.severity == "critical"]),
    )
