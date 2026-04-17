from app.models.comment import Comment
from app.services.comment_service import active_count


def render_highlight(comments: list[Comment]) -> str:
    return f"active comments: {active_count(comments)}"
