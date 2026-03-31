from app.models.snapshot import SnapshotRequest
from app.presenters.snapshot_presenter import render_snapshot
from app.services.snapshot_service import build_snapshot


def render_budget_snapshot(owner: str, total_spend: float, variance: float, note_count: int) -> str:
    request = SnapshotRequest(
        owner=owner,
        total_spend=total_spend,
        variance=variance,
        note_count=note_count,
    )
    snapshot = build_snapshot(request)
    return render_snapshot(snapshot)
