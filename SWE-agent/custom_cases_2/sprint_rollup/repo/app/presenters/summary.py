from app.models.item import Item
from app.services.rollup import blocker_count


def sprint_summary(items: list[Item]) -> str:
    return f"open blockers: {blocker_count(items)}"

