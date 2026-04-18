from app.models.task import Task


def blocked_task_count(tasks: list[Task]) -> int:
    return len([task for task in tasks if task.blocked])
