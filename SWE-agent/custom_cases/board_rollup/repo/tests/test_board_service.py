from app.models.card import BoardCard
from app.services.board_service import build_board_summary


def test_active_alert_total_excludes_archived_cards() -> None:
    cards = [
        BoardCard(title="Cache error", owner="Ava", severity="high"),
        BoardCard(title="Queue lag", owner="Mina", severity="medium"),
        BoardCard(title="Closed incident", owner="Ava", severity="high", archived=True),
    ]
    summary = build_board_summary(cards)
    assert summary.active_cards == 2
    assert summary.alert_cards == 2


def test_unique_owners_only_come_from_active_cards() -> None:
    cards = [
        BoardCard(title="Cache error", owner="Ava", severity="high"),
        BoardCard(title="Closed incident", owner="Ava", severity="high", archived=True),
    ]
    summary = build_board_summary(cards)
    assert summary.owners == ["Ava"]
