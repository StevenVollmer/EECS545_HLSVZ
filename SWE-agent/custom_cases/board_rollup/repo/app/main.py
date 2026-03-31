from app.config import BOARD_TITLE
from app.models.card import BoardCard
from app.presenters.summary_presenter import render_board_summary
from app.services.board_service import build_board_summary


def render_board(cards: list[BoardCard]) -> str:
    summary = build_board_summary(cards)
    return render_board_summary(BOARD_TITLE, summary)
