from app.models.card import Card
from app.presenters.summary import board_summary
from app.services.board import visible_owner_names


def test_visible_owner_names_case_insensitive_sort() -> None:
    cards = [
        Card(owner="mina", active=True, needs_attention=True),
        Card(owner="Ava", active=True, needs_attention=True),
    ]
    assert visible_owner_names(cards) == ["mina", "Ava"]


def test_summary_lists_visible_owners() -> None:
    cards = [Card(owner="Ava", active=True, needs_attention=True)]
    assert board_summary(cards) == "owners: Ava"


def test_inactive_cards_are_hidden() -> None:
    cards = [Card(owner="Ava", active=False, needs_attention=True)]
    assert visible_owner_names(cards) == []

