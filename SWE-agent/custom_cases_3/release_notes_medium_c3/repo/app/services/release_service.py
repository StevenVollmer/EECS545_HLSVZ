from app.models.change import Change


def public_change_titles(changes: list[Change]) -> list[str]:
    return [change.title for change in changes if change.public]
