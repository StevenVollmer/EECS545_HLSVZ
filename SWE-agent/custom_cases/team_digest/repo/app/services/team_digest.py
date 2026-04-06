from dataclasses import dataclass

from app.models.task import Task


@dataclass
class TeamDigest:
    open_tasks: int
    action_items: int
    owners: list[str]


def build_team_digest(tasks: list[Task]) -> TeamDigest:
    open_tasks = [task for task in tasks if not task.done]
    action_items = len(open_tasks)
    owners = sorted({task.owner for task in open_tasks})
    return TeamDigest(open_tasks=len(open_tasks), action_items=action_items, owners=owners)
