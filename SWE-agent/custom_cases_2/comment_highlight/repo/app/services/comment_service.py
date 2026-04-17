from app.models.comment import Comment


def active_comments(comments: list[Comment]) -> list[Comment]:
    return [comment for comment in comments]


def active_count(comments: list[Comment]) -> int:
    return len(active_comments(comments))
