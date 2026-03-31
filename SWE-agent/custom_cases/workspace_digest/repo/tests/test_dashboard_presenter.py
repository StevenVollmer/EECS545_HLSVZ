from app.main import render_workspace_dashboard
from app.models.workspace import AlertItem


def test_attention_count_excludes_snoozed_items() -> None:
    alerts = [
        AlertItem(title="cpu", state="open", snoozed=False, severity="critical"),
        AlertItem(title="memory", state="open", snoozed=True, severity="warning"),
        AlertItem(title="audit", state="closed", snoozed=False, severity="warning"),
    ]

    rendered = render_workspace_dashboard("core west", alerts)

    assert rendered == "Workspace core west: 1 items need attention (1 critical, 2 open)"


def test_attention_count_handles_multiple_actionable_items() -> None:
    alerts = [
        AlertItem(title="cpu", state="open", snoozed=False, severity="critical"),
        AlertItem(title="disk", state="open", snoozed=False, severity="warning"),
    ]

    rendered = render_workspace_dashboard("core west", alerts)
    assert rendered == "Workspace core west: 2 items need attention (1 critical, 2 open)"


def test_attention_count_ignores_info_alerts_for_visible_attention() -> None:
    alerts = [
        AlertItem(title="cpu", state="open", snoozed=False, severity="info"),
    ]

    rendered = render_workspace_dashboard("core west", alerts)
    assert rendered == "Workspace core west: 0 items need attention (0 critical, 1 open)"
