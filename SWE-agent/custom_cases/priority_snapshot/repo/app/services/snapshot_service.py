from app.models.item import SnapshotItem


def attention_count(items: list[SnapshotItem]) -> int:
    active_items = [item for item in items if not item.archived]
    return len(active_items)
