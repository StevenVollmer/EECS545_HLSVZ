from app.models.workspace import WorkspaceDigest


def render_dashboard(digest: WorkspaceDigest) -> str:
    return (
        f"Workspace {digest.name}: "
        f"{digest.visible_alert_count} items need attention "
        f"({digest.critical_alert_count} critical, {digest.open_alert_count} open)"
    )
