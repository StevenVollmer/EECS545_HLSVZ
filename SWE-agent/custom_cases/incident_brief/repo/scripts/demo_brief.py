from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import render_team_brief
from app.models.incident import Incident


def main() -> None:
    incidents = [
        Incident(title="api", severity="critical", state="open", muted=False),
        Incident(title="disk", severity="low", state="open", muted=True),
        Incident(title="audit", severity="warning", state="closed", muted=False),
    ]
    print(render_team_brief("east platform", incidents))


if __name__ == "__main__":
    main()
