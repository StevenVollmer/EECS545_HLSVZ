from app.models.snapshot import SnapshotMetrics, SnapshotRequest, SnapshotViewModel
from app.services.note_service import build_notes
from app.utils.currency import abbreviate_currency
from app.utils.variance import variance_label


def build_snapshot(request: SnapshotRequest) -> SnapshotViewModel:
    metrics = SnapshotMetrics(
        total_spend_label=abbreviate_currency(request.total_spend),
        variance_label=variance_label(request.variance),
        note_count=request.note_count,
    )
    return SnapshotViewModel(
        heading="Budget health overview",
        owner=request.owner,
        summary_line="This snapshot is intended for finance preview use.",
        metrics=metrics,
        notes=build_notes(request.owner, request.note_count),
    )
