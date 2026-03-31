from app.main import render_workspace_dashboard
from app.models.workspace import AlertItem


def sample_output() -> str:
    alerts = [
        AlertItem(title="cpu", state="open", snoozed=False, severity="critical"),
        AlertItem(title="disk", state="open", snoozed=True, severity="warning"),
    ]
    return render_workspace_dashboard("ops alpha", alerts)
