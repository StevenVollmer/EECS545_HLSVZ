from app.models.card import BoardCard


def unique_owner_names(cards: list[BoardCard]) -> list[str]:
    seen: set[str] = set()
    owners: list[str] = []
    for card in cards:
        if card.owner not in seen:
            seen.add(card.owner)
            owners.append(card.owner)
    return owners
