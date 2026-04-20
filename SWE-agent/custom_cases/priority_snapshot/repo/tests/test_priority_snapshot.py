from app.models.item import SnapshotItem
from app.presenters.snapshot_presenter import render_snapshot
from app.services.snapshot_service import attention_count


def test_attention_count_excludes_low_severity_items() -> None:
    items = [
        SnapshotItem(title="CPU", severity="critical"),
        SnapshotItem(title="Memory", severity="warning"),
        SnapshotItem(title="Tip", severity="low"),
    ]
    assert attention_count(items) == 2


def test_attention_count_ignores_archived_items() -> None:
    items = [
        SnapshotItem(title="CPU", severity="critical", archived=True),
        SnapshotItem(title="Memory", severity="warning"),
    ]
    assert attention_count(items) == 1


def test_render_snapshot_uses_attention_count() -> None:
    items = [
        SnapshotItem(title="CPU", severity="critical"),
        SnapshotItem(title="Memory", severity="warning"),
        SnapshotItem(title="Tip", severity="low"),
    ]
    assert render_snapshot(items) == "2 items need attention"
