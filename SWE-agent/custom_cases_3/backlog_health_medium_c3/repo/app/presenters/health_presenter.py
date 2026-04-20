from app.models.task import Task
from app.services.backlog_service import blocked_task_count
from app.utils.labels import render_count_label


def render_health(tasks: list[Task]) -> str:
    return render_count_label("active blockers", blocked_task_count(tasks), "task")
