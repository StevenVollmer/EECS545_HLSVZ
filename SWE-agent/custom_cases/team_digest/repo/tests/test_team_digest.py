from app.models.task import Task
from app.services.team_digest import build_team_digest


def test_action_items_exclude_snoozed_tasks() -> None:
    tasks = [
        Task(title="CPU", owner="Ava"),
        Task(title="Billing", owner="Mina", snoozed=True),
        Task(title="Queue", owner="Ava"),
    ]
    digest = build_team_digest(tasks)
    assert digest.open_tasks == 3
    assert digest.action_items == 2


def test_owners_are_unique_and_sorted() -> None:
    tasks = [
        Task(title="CPU", owner="Mina"),
        Task(title="Queue", owner="Ava"),
        Task(title="Billing", owner="Ava"),
    ]
    assert build_team_digest(tasks).owners == ["Ava", "Mina"]


def test_done_tasks_are_not_open() -> None:
    tasks = [Task(title="CPU", owner="Ava", done=True)]
    assert build_team_digest(tasks).open_tasks == 0
