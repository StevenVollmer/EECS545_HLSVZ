from app.models.card import BoardCard
from app.models.summary import BoardSummary
from app.utils.owners import unique_owner_names


def build_board_summary(cards: list[BoardCard]) -> BoardSummary:
    active_cards = [card for card in cards if not card.archived]
    alert_cards = len(cards)
    owners = unique_owner_names(active_cards)
    return BoardSummary(
        active_cards=len(active_cards),
        alert_cards=alert_cards,
        owners=owners,
    )
