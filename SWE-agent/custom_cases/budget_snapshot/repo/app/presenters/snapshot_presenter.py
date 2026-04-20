from app.config import SNAPSHOT_TITLE
from app.models.snapshot import SnapshotViewModel
from app.views.console import join_lines


def render_snapshot(view_model: SnapshotViewModel) -> str:
    lines = [
        SNAPSHOT_TITLE,
        "=" * len(SNAPSHOT_TITLE),
        view_model.heading,
        f"Owner: {view_model.owner}",
        view_model.summary_line,
        f"Total planned spend: {view_model.metrics.total_spend_label}",
        f"Variance: {view_model.metrics.variance_label}",
        f"Notes: {view_model.metrics.note_count}",
    ]
    for note in view_model.notes:
        lines.append(f"- {note}")
    return join_lines(lines)
