from app.models.summary import BoardSummary
from app.presenters.summary_presenter import render_board_summary


def test_summary_rendering() -> None:
    rendered = render_board_summary(
        "Ops Board",
        BoardSummary(active_cards=2, alert_cards=2, owners=["Ava", "Mina"]),
    )
    assert "Alerts: 2 cards" in rendered
    assert "Owners: Ava, Mina" in rendered
