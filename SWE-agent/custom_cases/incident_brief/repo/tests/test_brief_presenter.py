from app.main import render_team_brief
from app.models.incident import Incident


def test_urgent_count_excludes_muted_low_severity_items() -> None:
    incidents = [
        Incident(title="api", severity="critical", state="open", muted=False),
        Incident(title="disk", severity="low", state="open", muted=True),
        Incident(title="audit", severity="warning", state="closed", muted=False),
    ]

    rendered = render_team_brief("east platform", incidents)
    assert rendered == "Team east platform: 1 urgent items (1 critical, 2 open)"


def test_open_count_still_includes_muted_open_items() -> None:
    incidents = [
        Incident(title="disk", severity="low", state="open", muted=True),
        Incident(title="queue", severity="info", state="open", muted=False),
    ]

    rendered = render_team_brief("east platform", incidents)
    assert rendered.endswith("(0 critical, 2 open)")
