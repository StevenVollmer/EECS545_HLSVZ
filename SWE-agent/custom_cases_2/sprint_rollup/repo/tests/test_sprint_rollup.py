from app.models.item import Item
from app.presenters.summary import sprint_summary
from app.services.rollup import blocker_count


def test_blockers_exclude_resolved_items() -> None:
    items = [
        Item("Queue", blocked=True, resolved=False),
        Item("Billing", blocked=True, resolved=True),
        Item("Docs", blocked=False, resolved=False),
    ]
    assert blocker_count(items) == 1


def test_summary_uses_rollup_count() -> None:
    items = [Item("Queue", blocked=True, resolved=False)]
    assert sprint_summary(items) == "open blockers: 1"


def test_non_blocked_items_do_not_count() -> None:
    assert blocker_count([Item("Docs", blocked=False, resolved=False)]) == 0

