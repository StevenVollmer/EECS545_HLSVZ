from app.models.item import Item


def blocker_count(items: list[Item]) -> int:
    return len([item for item in items if item.blocked])

