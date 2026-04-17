from app.models.card import Card


def visible_owner_names(cards: list[Card]) -> list[str]:
    names = {
        card.owner
        for card in cards
        if card.active and card.needs_attention
    }
    return sorted(names)

