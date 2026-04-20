from app.models.comment import Comment
from app.presenters.highlight_presenter import render_highlight
from app.services.comment_service import active_comments


def test_active_comments_exclude_deleted() -> None:
    comments = [
        Comment(author="ava", deleted=True),
        Comment(author="mina", deleted=False),
    ]
    assert [comment.author for comment in active_comments(comments)] == ["mina"]


def test_highlight_counts_active_comments() -> None:
    comments = [
        Comment(author="ava", deleted=False),
        Comment(author="mina", deleted=False),
    ]
    assert render_highlight(comments) == "active comments: 2"


def test_highlight_empty_when_all_deleted() -> None:
    comments = [Comment(author="ava", deleted=True)]
    assert render_highlight(comments) == "active comments: 0"
