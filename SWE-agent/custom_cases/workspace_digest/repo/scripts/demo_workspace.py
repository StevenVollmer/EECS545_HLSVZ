from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import render_workspace_dashboard
from app.models.workspace import AlertItem


def main() -> None:
    alerts = [
        AlertItem(title="cpu", state="open", snoozed=False, severity="critical"),
        AlertItem(title="memory", state="open", snoozed=True, severity="warning"),
        AlertItem(title="audit", state="closed", snoozed=False, severity="warning"),
    ]
    print(render_workspace_dashboard("core west", alerts))


if __name__ == "__main__":
    main()
