from app.models.workspace import AlertItem, Workspace
from app.services.digest_builder import build_workspace_digest


def test_digest_builder_preserves_open_count() -> None:
    workspace = Workspace(
        name="ops east",
        alerts=[
            AlertItem(title="cpu", state="open", snoozed=False, severity="critical"),
            AlertItem(title="disk", state="open", snoozed=True, severity="warning"),
            AlertItem(title="audit", state="closed", snoozed=False, severity="warning"),
        ],
    )

    digest = build_workspace_digest(workspace)
    assert digest.open_alert_count == 2


def test_digest_builder_counts_critical_actionable_alerts() -> None:
    workspace = Workspace(
        name="ops east",
        alerts=[
            AlertItem(title="cpu", state="open", snoozed=False, severity="critical"),
            AlertItem(title="disk", state="open", snoozed=False, severity="warning"),
        ],
    )

    digest = build_workspace_digest(workspace)
    assert digest.critical_alert_count == 1
