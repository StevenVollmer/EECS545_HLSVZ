from app.models.task import Task
from app.presenters.health_presenter import render_health
from app.services.backlog_service import blocked_task_count


def test_blocked_task_count_excludes_archived_tasks() -> None:
    tasks = [
        Task("Ops cleanup", blocked=True, archived=False),
        Task("Retired epic", blocked=True, archived=True),
    ]
    assert blocked_task_count(tasks) == 1


def test_presenter_uses_service_output() -> None:
    assert render_health([Task("Ops cleanup", blocked=True, archived=False)]) == "active blockers: 1 task"


def test_non_matching_rows_do_not_count() -> None:
    assert blocked_task_count([Task("Docs", blocked=False, archived=False)]) == 0
