from app.models.card import Card
from app.services.board import visible_owner_names


def board_summary(cards: list[Card]) -> str:
    names = visible_owner_names(cards)
    return f"owners: {', '.join(names)}"

