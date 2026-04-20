from app.models.summary import BoardSummary
from app.utils.labels import pluralize


def render_board_summary(title: str, summary: BoardSummary) -> str:
    owners = ", ".join(summary.owners)
    return (
        f"{title}\n"
        f"Active cards: {summary.active_cards}\n"
        f"Alerts: {summary.alert_cards} {pluralize('card', summary.alert_cards)}\n"
        f"Owners: {owners}"
    )
