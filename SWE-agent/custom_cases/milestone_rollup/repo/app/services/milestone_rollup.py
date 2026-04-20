from app.models.milestone import Milestone


def blocker_count(milestones: list[Milestone]) -> int:
    blockers = [milestone for milestone in milestones if milestone.blocked]
    return len(blockers)
