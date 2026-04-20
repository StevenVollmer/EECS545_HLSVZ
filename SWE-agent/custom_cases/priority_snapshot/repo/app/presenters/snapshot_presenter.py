from app.models.item import SnapshotItem
from app.services.snapshot_service import attention_count


def render_snapshot(items: list[SnapshotItem]) -> str:
    return f"{attention_count(items)} items need attention"
