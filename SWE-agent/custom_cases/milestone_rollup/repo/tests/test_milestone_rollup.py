from app.models.milestone import Milestone
from app.services.milestone_rollup import blocker_count


def test_blocker_count_ignores_closed_milestones() -> None:
    milestones = [
        Milestone(name="API", blocked=True),
        Milestone(name="Billing", blocked=True, closed=True),
    ]
    assert blocker_count(milestones) == 1


def test_blocker_count_handles_zero_blockers() -> None:
    assert blocker_count([Milestone(name="API", blocked=False)]) == 0


def test_blocker_count_counts_open_blockers() -> None:
    milestones = [
        Milestone(name="API", blocked=True),
        Milestone(name="Billing", blocked=True),
    ]
    assert blocker_count(milestones) == 2
